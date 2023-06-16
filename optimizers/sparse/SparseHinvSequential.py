import os
import time

import torch
import wandb
import numpy as np
from line_profiler import LineProfiler
from torch.nn.parallel import parallel_apply

from helpers.mylogger import MyLogger
from helpers.optim import zerorize, aggregate
from helpers.tools import get_gpus

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


class SparseHinvSequential:

    # `dev` is the device where all the coefficient calculation happens
    # `grads` ... initial $m \times d$ gradient matrix $G$; assumed to be on CPU (can be deleted afterwards)
    # `dev` ... device where coefficient calculation happens
    # `gpus` ... list of GPUs across which to split stored gradients
    # `damp` ... dampening constant $\lambda$
    # @profile
    def __init__(self, m, d, nnz, dev, gpus, damp):
        if USE_CUDA and m % 32 != 0 or m > 1024:
            raise ValueError('CUDA implementation currently on supports $m$ < 1024 and divisible by 32.')
        self.cuda_profile = False
        self.m = m
        self.d = d
        self.nnz = nnz
        self.dev = dev
        self.gpus = gpus
        self.dtype = torch.float
        self.buffer_full_size = self.nnz * self.m
        self.dper = self.d // len(gpus) + 1
        self.grads = None
        self.grads_count = 0  # counts how many gradients were introduced so far
        self.wandb_data = dict()

        self.damp = None
        self.lambd = None
        self.set_damp(damp)

        self.gpus = get_gpus(remove_first=False)
        self.gpus_count = len(self.gpus)
        self.gpus_cols = int(self.nnz / self.gpus_count) + 1

        self.result_m = [torch.zeros(self.m, dtype=torch.float, device=gpu) for gpu in self.gpus]
        self.result_d = [torch.zeros(self.d, dtype=torch.float, device=gpu) for gpu in self.gpus]

        self.indices, self.values = [], []
        for index, gpu in enumerate(self.gpus):
            start = index * self.gpus_cols
            end = min((index + 1) * self.gpus_cols, self.nnz)
            cols = end - start
            self.indices.append(torch.zeros(m, cols, dtype=torch.long, device=gpu))
            self.values.append(torch.zeros(m, cols, dtype=torch.float, device=gpu))

        self.dots = torch.zeros((self.m, self.m), device=self.dev, dtype=self.dtype)  # matrix $GG^T$
        self.last = 0  # ringbuffer index
        self.giHig = None # matrix $D$
        self.denom = torch.zeros(self.m, device=self.dev, dtype=self.dtype)  # $D_ii + m$
        self.coef = self.lambd * torch.eye(self.m, device=self.dev, dtype=self.dtype)  # matrix $B$
        self.setup()
        print('USING SparseHinvSequential')


    def set_damp(self, new_damp):
        self.damp = new_damp
        self.lambd = 1. / new_damp

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
    def integrate_gradient_and_precondition(self, g, indices, x):
        """
        returns update inv(F) * g = 1/lambda * g - linear_combination_of_gradients (tmp contains linear comb params)
        :param g: tensor with zeros, but in dense format
        :param indices:
        :param x:
        :return:
        """
        tmp = self.integrate_gradient(g, indices)
        p = self.precondition(x, tmp)
        return p

    def integrate_gradient_and_precondition_twice(self, g, indices, x):
        """
        Returns update inv(F) * g = 1/lambda * g - linear_combination_of_gradients (tmp contains linear comb params)
        :param g: tensor with zeros, but in dense format
        :param indices:
        :return:
        """
        tmp = self.integrate_gradient(g, indices)
        p1 = self.precondition(x, tmp)
        p2 = self.precondition(p1, None) # None means that the dot products with p1 are computed
        # TODO do we have to use the same tmp to obtain p2?
        # I think so, we dont integrate anything in the buffer,
        # just precondition with the current Fisher estimate
        return p2

    def integrate_gradient(self, g, indices):
        self.set_grad(self.last, g[indices], indices)
        tmp = self.compute_scalar_products(g)
        tmp = tmp.squeeze()  # (d, 1) becomes (d,)
        self.dots[self.last, :] = tmp
        self.dots[:, self.last] = tmp
        self.setup()
        self.last = (self.last + 1) % self.m
        return tmp

    # @profile
    def set_grad(self, row_index, values, indices):
        self.grads_count += 1
        for idx_gpu, gpu in enumerate(self.gpus):
            start = idx_gpu * self.gpus_cols
            end = min((idx_gpu + 1) * self.gpus_cols, self.nnz)
            self.indices[idx_gpu][row_index, :] = indices[start:end].to(dtype=torch.long, device=gpu)
            self.values[idx_gpu][row_index, :] = values[start:end].to(dtype=torch.float, device=gpu)

    def compute_scalar_products(self, g):
        """Computes G * g"""
        def f(gpu_index):
            gpu = self.gpus[gpu_index]
            slice_g = g.to(gpu).take(self.indices[gpu_index])
            self.result_m[gpu_index].add_((slice_g * self.values[gpu_index]).sum(axis=1))

        zerorize(self.result_m, self.gpus_count)
        parallel_apply(modules=[f] * self.gpus_count, inputs=range(self.gpus_count))
        aggregate(self.result_m, self.gpus_count)
        return self.result_m[0].to(self.dev)

    # @profile
    def precondition(self, g, dots=None):
        """
            Returns the update inv(F) * x
            The matrix M stores the coefficients of the linear combination
            x: usually the gradient
        """
        # print('[mul]')
        # print(f'\tdots norm = {dots.norm()}')
        if dots is None:
            dots = self.compute_scalar_products(g)
        giHix = self.lambd * dots
        if USE_CUDA:
            giHix = hinv_cuda.hinv_mul(self.m, self.giHig, giHix)
        else:
            for i in range(1, self.m):
                giHix[i:].sub_(self.giHig[i - 1, i:], alpha=giHix[i - 1] / self.denom[i - 1])
        M = (giHix / self.denom).matmul(self.coef) # .view(-1, 1) # view is linked to matmul_grads_sequential_batch

        partA = self.lambd * g
        partB = self.compute_linear_combination(M)
        self.wandb_data.update(dict(norm_partA=partA.norm(p=2), norm_partB=partB.norm(p=2)))
        return partA - partB

    def compute_linear_combination(self, M):
        """Computes the linear combination of gradients, where the coefficients are M: M[i] * grad[i]"""
        def f(gpu_index):
            for row in range(min(self.m, self.grads_count)):
                self.result_d[gpu_index].index_add_(0, self.indices[gpu_index][row, :], M[row] * self.values[gpu_index][row, :])

        M = M.detach().cpu().numpy()

        zerorize(self.result_d, self.gpus_count)
        parallel_apply(modules=[f] * self.gpus_count, inputs=range(self.gpus_count))
        aggregate(self.result_d, self.gpus_count)

        self.wandb_data.update(dict(lin_comb_coef_norm=np.linalg.norm(M)))

        return self.result_d[0].to(self.dev)

        # if self.cuda_profile and self.grads_count >= self.m:
        #     torch.cuda.nvtx.range_push(f'[matmul_grads_sequential][for]step@{self.grads_count}')
        # if self.cuda_profile and self.grads_count >= self.m:
        #     torch.cuda.nvtx.range_pop()


# print(f'[matmul_grads]')
# print(f'\tM.size() = {M.shape}')
# input('Press any key to continue...')

# print(f'[update_mul]')
# print(f'\tdot_prod norm = {tmp.norm()}')

# print(f'[grads_matmul_sequential][gpu-{gpu_index}]')
# print(f'\tslice_g size() = {slice_g.size()}')
# print(f'\tslice_g norm = {slice_g.norm()}')
# print()
# print(f'\tself.indices[{gpu_index}][0, :5] = {self.indices[gpu_index][0,:5]}')
# print(f'\tslice_g[first-5-indices] = {g.to(gpu)[self.indices[gpu_index][0,:5]]}')
# print(f'\tself.values[{gpu_index}][first-5-indices] = {self.values[gpu_index][0, :5]}')
# print()
# print(f'\tself.values[{gpu_index}] size = {slice_g.size()}')
# print(f'\tself.values[{gpu_index}] norm = {slice_g.norm()}')
# print(f'\tslice_g DOT self.values[{gpu_index}] = {(slice_g * self.values[gpu_index]).sum()}')
# print()
# print(f'ASSERTION')
# print(f'\t ==>', self.values[0][0, 0] == g[self.indices[0][0, 0]])
# print()
# print(f'\tself.result_m[{gpu_index}] size: {self.result_m[gpu_index].size()}')
# print(f'\tself.result_m[{gpu_index}] norm before add = {self.result_m[gpu_index].norm().item()}')
# print(f'\ttemp size = {temp.size()}')
# print(f'\ttemp_sum size = {temp_sum.size()}')
# print(f'\ttemp_sum[:10] = {temp_sum[:10]}')
# print(f'\tself.result_m[{gpu_index}] norm after add = {self.result_m[gpu_index].norm().item()}')
