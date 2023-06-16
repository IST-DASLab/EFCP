from typing import Tuple, Union, Any

import torch
from torch import Tensor

from ..grad_maker import GradientMaker
import hinv_cuda
import wandb
import os

__all__ = ['MFACGradientMaker']

import numpy as np
from torch.nn.parallel import parallel_apply

USE_CUDA = True


def get_first_device():
    if not torch.cuda.is_available():
        return torch.device('cpu')
    return torch.device('cuda:0')


def get_gpus(remove_first):
    if not torch.cuda.is_available():
        return ['cpu']
    if torch.cuda.device_count() == 1:
        return [get_first_device()]

    gpus = [torch.device(f'cuda:{i}') for i in range(len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')))]
    print(f'Device: {get_first_device()}')
    if remove_first:
        gpus.remove(get_first_device())
    print(f'GPUs: {gpus}')
    return gpus


class HInvFastUpMulti:
    def __init__(self, grads, fix_scaling, dev, gpus, damp=1e-5, optim_name=None):
        self.m, self.d = grads.shape
        self.dev = dev
        self.gpus = gpus
        self.dtype = grads.dtype
        self.gpus = gpus
        self.fix_scaling = fix_scaling
        self.grads_count = 0
        self.wandb_data = dict()
        self.damp = None
        self.lambd = None
        self.set_damp(damp)
        self.optim_name = optim_name

        if USE_CUDA and self.m % 32 != 0 or self.m > 1024:
            raise ValueError('CUDA implementation currently on supports $m$ < 1024 and divisible by 32.')

        self.dper = self.d // len(gpus) + 1
        self.grads = []  # matrix $G$ in the paper
        for idx in range(len(gpus)):
            start, end = idx * self.dper, (idx + 1) * self.dper
            self.grads.append(grads[:, start:end].to(gpus[idx]))
        self.dots = torch.zeros((self.m, self.m), device=self.dev, dtype=self.dtype)  # matrix $GG^T$
        for idx in range(len(gpus)):
            self.dots += self.grads[idx].matmul(self.grads[idx].t()).to(self.dev)

        self.last = 0  # ringbuffer index
        self.giHig = self.lambd * self.dots  # matrix $D$
        self.denom = torch.zeros(self.m, device=self.dev, dtype=self.dtype)  # $D_ii + m$
        self.coef = self.lambd * torch.eye(self.m, device=self.dev, dtype=self.dtype)  # matrix $B$

        if fix_scaling:
            print('USING setup_fixed')
            self.setup = self.setup_fixed
        else:
            print('USING setup_original')
            self.setup = self.setup_original

        self.setup()

    def set_damp(self, new_damp):
        self.damp = new_damp
        self.lambd = 1. / new_damp

    def reset_optimizer(self):
        self.grads_count = 0
        for idx in range(len(self.gpus)):
            self.grads[idx].zero_()
        self.dots.zero_()
        for idx in range(len(self.gpus)):
            self.dots += self.grads[idx].matmul(self.grads[idx].t()).to(self.dev)
        self.last = 0
        self.giHig = self.lambd * self.dots  # matrix $D$
        self.denom = torch.zeros(self.m, device=self.dev, dtype=self.dtype)  # $D_ii + m$
        self.coef = self.lambd * torch.eye(self.m, device=self.dev, dtype=self.dtype)  # matrix $B$
        self.setup()

    # Calculate $D$ / `giHig` and $B$ / `coef`
    def setup_original(self):
        self.giHig = self.lambd * self.dots
        diag = torch.diag(torch.full(size=[self.m], fill_value=self.m, device=self.dev, dtype=self.dtype))
        self.giHig = torch.lu(self.giHig + diag, pivot=False)[0]
        self.giHig = torch.triu(self.giHig - diag)
        self.denom = self.m + torch.diagonal(self.giHig) # here we should use min(grads_count, m)
        tmp = -self.giHig.t().contiguous() / self.denom.reshape((1, -1))

        if USE_CUDA:
            diag = torch.diag(torch.full(size=[self.m], fill_value=self.lambd, device=self.dev, dtype=self.dtype))
            self.coef = hinv_cuda.hinv_setup(tmp, diag)
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

    # Replace oldest gradient with `g` and then calculate the IHVP with `g`
    def integrate_gradient_and_precondition(self, g, x):
        tmp = self.integrate_gradient(g)
        p = self.precondition(x, tmp)
        return p

    def integrate_gradient_and_precondition_twice(self, g, x):
        tmp = self.integrate_gradient(g)
        p1 = self.precondition(x, tmp)
        p2 = self.precondition(p1, None) # see comment from the method with the same name from SparseHinvSequential

        self.wandb_data.update({
            'SqNewton_cos_x_p1': get_cos_and_angle(x, p1)[0],
            'SqNewton_cos_p1_p2': get_cos_and_angle(p1, p2)[0],
        })

        return p2

    # Replace oldest gradient with `g`
    def integrate_gradient(self, g):
        self.set_grad(self.last, g)
        tmp = self.compute_scalar_products(g)
        self.dots[self.last, :] = tmp
        self.dots[:, self.last] = tmp
        self.setup()
        self.last = (self.last + 1) % self.m
        return tmp

    # Distributed `grads[j, :] = g`
    def set_grad(self, j, g):
        self.grads_count += 1
        def f(i):
            start, end = i * self.dper, (i + 1) * self.dper
            self.grads[i][j, :] = g[start:end]

        parallel_apply(
            [f] * len(self.grads), list(range(len(self.gpus)))
        )

    def compute_scalar_products(self, x):
        def f(i):
            start, end = i * self.dper, (i + 1) * self.dper
            G = self.grads[i]
            return G.matmul(x[start:end].to(G.device)).to(self.dev)

        outputs = parallel_apply(
            [f] * len(self.gpus), list(range(len(self.gpus)))
        )
        return sum(outputs)

    def precondition(self, x, dots=None):
        if dots is None:
            dots = self.compute_scalar_products(x)
        giHix = self.lambd * dots
        if USE_CUDA:
            rows = self.m
            if self.fix_scaling:
                rows = min(self.grads_count, self.m)
            giHix = hinv_cuda.hinv_mul(rows, self.giHig, giHix)
        else:
            for i in range(1, self.m):
                giHix[i:].sub_(self.giHig[i - 1, i:], alpha=giHix[i - 1] / self.denom[i - 1])
        """
            giHix size: 1024
            denom size: 1024
            coef size: 1024x1024
            M size: 1024
            x size: d
        """
        M = (giHix / self.denom).matmul(self.coef)
        partA = self.lambd * x
        partB = self.compute_linear_combination(M)
        prefix = '' if self.optim_name is None else f'{self.optim_name}_'
        self.wandb_data.update({f'{prefix}norm_partA': partA.norm(p=2), f'{prefix}norm_partB': partB.norm(p=2)})
        return partA.to(self.dev) - partB.to(self.dev)

    # Distributed `x.matmul(grads)`
    def compute_linear_combination(self, x):
        def f(G):
            return (x.to(G.device).matmul(G)).to(self.dev)
        outputs = parallel_apply(
            [f] * len(self.grads), self.grads
        )
        """
            x size: 1024
            grads: 1024 x d
        """
        x = x.detach().cpu().numpy()
        norm = np.linalg.norm(x)
        prefix = '' if self.optim_name is None else f'{self.optim_name}_'
        self.wandb_data.update({f'{prefix}lin_comb_coef_norm': norm}) # 'lin_comb_coef_hist': wandb.Histogram(x)})
        return torch.cat(outputs)


class MFACGradientMaker(GradientMaker):
    def __init__(self, model, param_groups, ngrads, damp):
        super().__init__(model)
        self.ngrads = ngrads
        self.dev = "cuda:0"
        self.model_size = 0
        self.param_groups = param_groups

        with torch.no_grad():
            for group in self.param_groups:
                for p in group['params']:
                    self.model_size += p.numel()
        print(f'Model size: {self.model_size}')

        self.hinv = HInvFastUpMulti(
            grads=torch.zeros((ngrads, self.model_size), dtype=torch.float),
            fix_scaling=False,
            dev=get_first_device(),
            gpus=get_gpus(remove_first=False),
            damp=damp,
            optim_name=None)

        self.steps = 0

    def forward_and_backward(self) -> Union[Tuple[Any, Tensor], Any]:
        self.steps += 1
        model_output, loss = super().forward_and_backward()

        with torch.no_grad():
            # get all current gradients
            g = torch.empty(self.model_size).to(self.dev)
            x = 0
            for group in self.param_groups:
                for p in group['params']:
                    g[x : x + p.grad.numel()] = p.grad.reshape(-1)
                    x += p.grad.numel()

            if torch.isnan(g).sum() > 0:
                raise ValueError(f'[IONUT PROBLEM] gradient has NaNs')

            update = self.hinv.integrate_gradient_and_precondition(g=g, x=g)

            if self.steps % 20 == 0:
                log_stats(self.steps, g=g, u=update)

            x = 0
            for group in self.param_groups:
                for p in group['params']:
                    p.grad.data = update[x : x+p.grad.numel()].reshape(p.grad.shape)
                    x += p.grad.numel()

        return model_output, loss


@torch.no_grad()
def log_stats(step, g, u):
    norm_g = g.norm(p=2).item()
    norm_u = u.norm(p=2).item()
    sparsity_g = (g == 0).sum() / g.numel()
    sparsity_u = (u == 0).sum() / u.numel()
    cos, _ = get_cos_and_angle(g, u)
    wandb.log(dict(
        step=step,
        norm_g=norm_g,
        norm_u=norm_u,
        sparsity_u=sparsity_u,
        sparsity_g=sparsity_g,
        cos_g_u=cos,
    ))


def get_cos_and_angle(x, y):
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)(x, y).item()
    angle = map_interval(x=cos, a=-1, b=1, A=180, B=0)
    return cos, angle


def map_interval(x, a, b, A, B):
    """This method maps x in [a, b] to y in [A, B]"""
    return A + (B - A) * (x - a) / (b - a)
