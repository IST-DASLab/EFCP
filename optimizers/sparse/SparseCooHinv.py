import os
import sys

import torch
from line_profiler import LineProfiler
from torch.nn.parallel import parallel_apply
from concurrent.futures import ThreadPoolExecutor

from helpers.mylogger import MyLogger

# from pytorch_memlab import profile
# from memory_profiler import profile

# Disable tensor cores as they can mess with precision
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# Use custom CUDA implementation for computing $B$ / `coef` if installed
USE_CUDA = True
try:
    import hinv_cuda
except Exception as e:
    USE_CUDA = False

# USE_CUDA = False


class SparseCooHinv:

    # `dev` is the device where all the coefficient calculation happens
    # `grads` ... initial $m \times d$ gradient matrix $G$; assumed to be on CPU (can be deleted afterwards)
    # `dev` ... device where coefficient calculation happens
    # `gpus` ... list of GPUs across which to split stored gradients
    # `damp` ... dampening constant $\lambda$
    # @profile
    def __init__(self, m, d, nnz, dev, gpus, damp=1e-5):
        if USE_CUDA and m % 32 != 0 or m > 1024:
            raise ValueError('CUDA implementation currently on supports $m$ < 1024 and divisible by 32.')
        print('USING COO')
        self.m = m
        self.d = d
        self.nnz = nnz
        self.dev = dev
        self.gpus = gpus
        self.dtype = torch.float
        self.lambd = 1. / damp
        self.buffer_full_size = self.nnz * self.m
        self.dper = self.d // len(gpus) + 1
        self.grads = []
        self.grads_count = 0 # counts how many gradients were introduced so far

        self.buffer_m = torch.zeros(self.m, dtype=torch.float, device=self.dev)
        self.zeros_d = torch.zeros((1, self.d), dtype=torch.float, device=self.dev)
        # self.temp_buffers = []
        # for numel in range(1, self.m):
        #     self.temp_buffers.append(torch.zeros((self.m - numel,), dtype=torch.float, device=self.dev))

        self.buffer_d = torch.zeros((1, self.d), device=self.dev, dtype=self.dtype)
        self.dots = torch.zeros((self.m, self.m), device=self.dev, dtype=self.dtype)  # matrix $GG^T$
        self.last = 0  # ringbuffer index
        self.giHig = None # matrix $D$
        self.denom = torch.zeros(self.m, device=self.dev, dtype=self.dtype)  # $D_ii + m$
        self.coef = self.lambd * torch.eye(self.m, device=self.dev, dtype=self.dtype)  # matrix $B$
        self.setup()

    # @profile
    # Calculate $D$ / `giHig` and $B$ / `coef`
    def setup(self):
        self.giHig = self.lambd * self.dots
        diag_m = torch.diag(torch.full([self.m], self.m, device=self.dev, dtype=self.dtype))
        self.giHig = torch.lu(self.giHig + diag_m, pivot=False)[0]
        self.giHig = torch.triu(self.giHig - diag_m)
        self.denom = self.m + torch.diagonal(self.giHig)
        tmp = -self.giHig.t().contiguous() / self.denom.reshape((1, -1))

        if USE_CUDA:
            diag_lambd = torch.diag(torch.full([self.m], self.lambd, device=self.dev, dtype=self.dtype))
            self.coef = hinv_cuda.hinv_setup(tmp, diag_lambd)
        else:
            for i in range(max(self.last, 1), self.m):
                self.coef[i, :i] = tmp[i, :i].matmul(self.coef[:i, :i])

    # @profile
    # Replace oldest gradient with `g` and then calculate the IHVP with `g`
    def update_mul(self, g):
        """
        returns update inv(F) * g = 1/lambda * g - linear_combination_of_gradients (tmp contains linear comb params)
        :param g: tensor with zeros, but in dense format
        :param indices:
        :return:
        """
        # set a sparse gradient in the ring buffer, but keep g dense with many zeros
        g_sparse = g.view(1, -1).to_sparse_coo() # should be of size (1, d)
        self.set_grad(self.last, g_sparse)
        tmp = self.grads_matmul(g)
        self.dots[self.last, :] = tmp
        self.dots[:, self.last] = tmp
        self.setup()
        res = self.mul(g, tmp)
        self.last = (self.last + 1) % self.m
        return res

    # @profile
    # Distributed `grads[j, :] = g`
    def set_grad(self, row_index, g_sparse):
        if self.grads_count < self.m:
            self.grads.append(g_sparse)
        else:
            self.grads[row_index] = g_sparse
        self.grads_count += 1

    # @profile
    # Distributed `grads.matmul(x)`
    def grads_matmul(self, g):
        """
            Computes G * x, where x is a gradient and G is self.grads
            G: matrix of size (m,d) containing gradients on rows
            g: the gradient, which is of size (d,1) or (d,)
        """

        def dot(index):
            return torch.matmul(self.grads[index], g)[0]

        threads_count = min(self.m, self.grads_count)
        result = parallel_apply(modules=[dot] * threads_count, inputs=range(threads_count))
        r = torch.cat(result).to_dense()
        if self.grads_count < self.m:
            self.buffer_m[:self.grads_count] = r
            r = self.buffer_m
        else:
            del self.buffer_m
            self.buffer_m = None
        return r

    # @profile
    # Product with inverse of dampened empirical Fisher
    def mul(self, g, dots=None):
        """
            Returns the update inv(F) * x
            The matrix M stores the coefficients of the linear combination
            x: usually the gradient
        """
        if dots is None:
            dots = self.grads_matmul(g)
        giHix = self.lambd * dots
        if USE_CUDA:
            giHix = hinv_cuda.hinv_mul(self.giHig, giHix)
        else:
            for i in range(1, self.m):
                giHix[i:].sub_(self.giHig[i - 1, i:], alpha=giHix[i - 1] / self.denom[i - 1])
        M = (giHix / self.denom).matmul(self.coef)
        r = self.matmul_grads(M)
        return self.lambd * g - r.T

    # @profile
    # Distributed `x.matmul(grads)`
    def matmul_grads(self, M):
        """
            Computes G * M
            M: an intermediary result of MFAC of size (m,) (the coefficients of linear combination)
        """
        self.buffer_d.mul_(0)

        # def scalar_vector_mul(index):
        #     # first thread returns the zeros vector in dense format to be able to call sum(result)
        #     if index == -1:
        #         return self.zeros_d
        #     if index < self.grads_count:
        #         return M[index] * self.grads[index]
        #     return torch.sparse_coo_tensor(indices=[[], []], values=[], size=(1, self.d), device=self.dev)

        # # threads_count = min(self.m, self.grads_count)
        # result = parallel_apply(modules=[scalar_vector_mul] * (self.m + 1), inputs=list(range(-1, self.m)))
        # result = sum(result)
        # return result.to_dense()

        def scalar_vector_mul(index):
            return M[index] * self.grads[index]

        threads_count = min(self.m, self.grads_count)
        result = parallel_apply(modules=[scalar_vector_mul] * threads_count, inputs=list(range(threads_count)))

        for r in result:
            self.buffer_d.add_(r)

        return self.buffer_d


