from typing import Tuple, Union, Any

import torch
from torch import Tensor

from ..grad_maker import GradientMaker
import hinv_cuda
import wandb

__all__ = ['SparseMFACGradientMaker']

import numpy as np
from torch.nn.parallel import parallel_apply

def aggregate(vector, size):
    dev0 = vector[0].device
    for idx in range(1, size):
        vector[0].add_(vector[idx].to(dev0))


def zerorize(vector, size):
    for idx in range(size):
        vector[idx].zero_()


USE_CUDA = True
class SparseHinvSequential:
    # `dev` is the device where all the coefficient calculation happens
    # `grads` ... initial $m \times d$ gradient matrix $G$; assumed to be on CPU (can be deleted afterwards)
    # `dev` ... device where coefficient calculation happens
    # `gpus` ... list of GPUs across which to split stored gradients
    # `damp` ... dampening constant $\lambda$
    # @profile
    def __init__(self, m, d, nnz, fix_scaling, dev, gpus, damp):
        if USE_CUDA and m % 32 != 0 or m > 1024:
            raise ValueError('CUDA implementation currently on supports $m$ < 1024 and divisible by 32.')
        self.fix_scaling = fix_scaling
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

        self.gpus = ["cuda:0"]
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

    def setup_fixed(self):
        min_m_grads = min(self.m, self.grads_count)
        self.giHig = self.lambd * self.dots
        diag_m = torch.diag(torch.full(size=[self.m], fill_value=min_m_grads, device=self.dev, dtype=self.dtype))
        self.giHig = torch.lu(self.giHig + diag_m, pivot=False)[0]
        self.giHig = torch.triu(self.giHig - diag_m)
        self.denom = min_m_grads + torch.diagonal(self.giHig)
        tmp = -self.giHig.t().contiguous() / self.denom.reshape((1, -1))

        if USE_CUDA:
            diag_lambd = torch.diag(torch.full(size=[self.m], fill_value=self.lambd, device=self.dev, dtype=self.dtype))
            self.coef = hinv_cuda.hinv_setup(tmp, diag_lambd)
        else:
            for i in range(max(self.last, 1), self.m):
                self.coef[i, :i] = tmp[i, :i].matmul(self.coef[:i, :i])

    # @profile
    # Replace oldest gradient with `g` and then calculate the IHVP with `g`
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
        returns update inv(F) * g = 1/lambda * g - linear_combination_of_gradients (tmp contains linear comb params)
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
    # Distributed `grads[j, :] = g`
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
    # Product with inverse of dampened empirical Fisher
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
            rows = self.m
            if self.fix_scaling:
                rows = min(self.grads_count, self.m)
            # print(f'\tgiHix norm before cuda call = {giHix.norm()}')
            giHix = hinv_cuda.hinv_mul(rows, self.giHig, giHix)
            # print(f'\tgiHix norm after cuda call = {giHix.norm()}')
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


@torch.no_grad()
def get_k(numel, k_init):
    k = numel
    if k_init is not None:
        if k_init <= 1:
            # select top-k% parameters (PERCENT)
            k = int(k_init * numel)
        else:
            # select top-k parameters (CONSTANT)
            k = int(k_init)
    print(f'k={k}')
    return k


@torch.no_grad()
def compute_topk(vector, k, device):
    topk_indices = torch.topk(torch.abs(vector), k=k, sorted=False).indices
    mask = torch.zeros_like(vector).to(device)
    mask[topk_indices] = 1.
    vector_topk = vector * mask
    return vector_topk, topk_indices, mask


@torch.no_grad()
def apply_topk(lr, k, error, vector, use_ef, device):
    acc = error + lr * vector
    acc_topk, topk_indices, mask = compute_topk(vector=acc, k=k, device=device)
    if use_ef:
        error = acc - acc_topk
    return error, acc, acc_topk, mask, topk_indices


class SparseMFACGradientMaker(GradientMaker):
    def __init__(self, model, param_groups, k_init, ngrads, damp, fix):
        super().__init__(model)
        self.fix = fix
        self.ngrads = ngrads
        self.dev = "cuda:0"
        self.model_size = 0
        self.param_groups = param_groups

        with torch.no_grad():
            for group in self.param_groups:
                for p in group['params']:
                    self.model_size += p.numel()
        print(f'Model size: {self.model_size}')

        self.error = torch.zeros(self.model_size).to(self.dev)
        self.k = get_k(numel=self.model_size, k_init=k_init)
        nnz = self.k
        self.hinv = SparseHinvSequential(
            m=ngrads,
            d=self.model_size,
            nnz=nnz,
            fix_scaling=False,
            dev=self.dev,
            gpus=[self.dev],
            damp=damp)

        self.steps = 0

    def forward_and_backward(self) -> Union[Tuple[Any, Tensor], Any]:
        self.steps += 1
        model_output, loss = super().forward_and_backward()

        with torch.no_grad():
            # get all current gradients
            g_dense = torch.empty(self.model_size).to(self.dev)
            x = 0
            for group in self.param_groups:
                for p in group['params']:
                    g_dense[x : x + p.grad.numel()] = p.grad.reshape(-1)
                    x += p.grad.numel()

        if torch.isnan(g_dense).sum() > 0:
            raise ValueError(f'[ELDAR PROBLEM] gradient has NaNs')

        with torch.no_grad():
            self.error, acc, acc_topk, mask, topk_indices = apply_topk(
                lr=1,
                k=self.k,
                error=self.error,
                vector=g_dense,
                use_ef=True,
                device=self.dev,)

        with torch.no_grad():
            if self.fix == 0: # original version
                update = self.hinv.integrate_gradient_and_precondition(g=acc_topk, indices=topk_indices, x=acc_topk)
            elif self.fix == 1: # add grad components to update where update == 0 (only during the first m steps)
                update = self.hinv.integrate_gradient_and_precondition(g=acc_topk, indices=topk_indices, x=acc_topk)
                if self.steps <= self.ngrads:
                    M = (update == 0)
                    update.add_(M * g_dense)
            elif self.fix == 2: # add error components to update where update == 0 (only during first m steps)
                update = self.hinv.integrate_gradient_and_precondition(g=acc_topk, indices=topk_indices, x=acc_topk)
                if self.steps <= self.ngrads:
                    M = (update == 0)
                    q = M * self.error
                    update.add_(q)
                    self.error.sub_(q)
            elif self.fix == 3: # precondition dense gradient (only during first m steps)
                if self.steps <= self.ngrads:
                    update = self.hinv.integrate_gradient_and_precondition(g=acc_topk, indices=topk_indices, x=g_dense)
                else:
                    update = self.hinv.integrate_gradient_and_precondition(g=acc_topk, indices=topk_indices, x=acc_topk)
            elif self.fix == 4: # precondition dense gradient during the entire training
                update = self.hinv.integrate_gradient_and_precondition(g=acc_topk, indices=topk_indices, x=g_dense)

            if self.steps % 20 == 0:
                log_stats(self.steps, g=g_dense, u=update, e=self.error, c=acc_topk)

            x = 0
            for group in self.param_groups:
                for p in group['params']:
                    p.grad.data = update[x : x+p.grad.numel()].reshape(p.grad.shape)
                    x += p.grad.numel()

        return model_output, loss


@torch.no_grad()
def log_stats(step, g, u, e, c):
    norm_g = g.norm(p=2).item()
    norm_u = u.norm(p=2).item()
    norm_e = e.norm(p=2).item()
    sparsity_g = (g == 0).sum() / g.numel()
    sparsity_u = (u == 0).sum() / u.numel()
    sparsity_c = (c == 0).sum() / c.numel()
    cos, _ = get_cos_and_angle(g, u)
    wandb.log(dict(
        step=step,
        norm_g=norm_g,
        norm_u=norm_u,
        norm_e=norm_e,
        sparsity_u=sparsity_u,
        sparsity_g=sparsity_g,
        sparsity_c=sparsity_c,
        cos_g_u=cos,
    ))
    # ratio = torch.abs(u / (g + 1e-8))
    # q25, q50, q75 = torch.quantile(input=ratio, q=torch.tensor([0.25, 0.5, 0.75]), interpolation='midpoint')


def get_cos_and_angle(x, y):
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)(x, y).item()
    angle = map_interval(x=cos, a=-1, b=1, A=180, B=0)
    return cos, angle


def map_interval(x, a, b, A, B):
    """This method maps x in [a, b] to y in [A, B]"""
    return A + (B - A) * (x - a) / (b - a)
