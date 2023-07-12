# Contains a full implementation of the dynamic algorithm and the M-FAC optimizer

import wandb
import torch
import torch.nn as nn
import numpy as np

from helpers.tools import get_first_device, get_gpus
from helpers.mylogger import MyLogger
from helpers.optim import get_weights_and_gradients, update_model

# Disable tensor cores as they can mess with precision
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

USE_CUDA = True
try:
    import hinv_cuda
except Exception as e:
    USE_CUDA = False


class HInvFastUpMulti:

    def __init__(self, grads, dev, gpus, damp=1e-5, optim_name=None):
        self.m, self.d = grads.shape
        self.dev = dev
        self.gpus = gpus
        self.dtype = grads.dtype
        self.gpus = gpus
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

    def set_damp(self, new_damp):
        self.damp = new_damp
        self.lambd = 1. / new_damp

    def setup(self):
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

    def integrate_gradient_and_precondition(self, g, x):
        tmp = self.integrate_gradient(g)
        p = self.precondition(x, tmp)
        return p

    def integrate_gradient(self, g):
        self.set_grad(self.last, g)
        tmp = self.compute_scalar_products(g)
        self.dots[self.last, :] = tmp
        self.dots[:, self.last] = tmp
        self.setup()
        self.last = (self.last + 1) % self.m
        return tmp

    def set_grad(self, j, g):
        self.grads_count += 1
        def f(i):
            start, end = i * self.dper, (i + 1) * self.dper
            self.grads[i][j, :] = g[start:end]

        nn.parallel.parallel_apply(
            [f] * len(self.grads), list(range(len(self.gpus)))
        )

    def compute_scalar_products(self, x):
        def f(i):
            start, end = i * self.dper, (i + 1) * self.dper
            G = self.grads[i]
            return G.matmul(x[start:end].to(G.device)).to(self.dev)

        outputs = nn.parallel.parallel_apply(
            [f] * len(self.gpus), list(range(len(self.gpus)))
        )
        return sum(outputs)

    def precondition(self, x, dots=None):
        if dots is None:
            dots = self.compute_scalar_products(x)
        giHix = self.lambd * dots
        if USE_CUDA:
            giHix = hinv_cuda.hinv_mul(self.m, self.giHig, giHix)
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

    def compute_linear_combination(self, x):
        def f(G):
            return (x.to(G.device).matmul(G)).to(self.dev)
        outputs = nn.parallel.parallel_apply(
            [f] * len(self.grads), self.grads
        )

        """
            x size: 1024
            grads: 1024 x d
        """

        x = x.detach().cpu().numpy()
        norm = np.linalg.norm(x)
        prefix = '' if self.optim_name is None else f'{self.optim_name}_'
        self.wandb_data.update({f'{prefix}lin_comb_coef_norm': norm})
        return torch.cat(outputs)


class DenseMFAC(torch.optim.Optimizer):
    def __init__(self, params, ngrads, lr, damp, wd_type, weight_decay, momentum, sparse=False, dev=None, gpus=None):
        self.wd_type = wd_type if isinstance(wd_type, str) else {0: 'wd', 1: 'reg', 2: 'both'}[wd_type]
        self.m = ngrads
        self.lr = lr
        self.damp = damp
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.dev = dev if dev is not None else get_first_device()
        self.gpus = gpus if gpus is not None else get_gpus(remove_first=False)
        self.model_size = None
        self.sparse = sparse
        self.sparsity_mask = None
        self.sparse_update = None

        self.steps = 0
        self.named_parameters = None
        self.wandb_data = dict()

        print(f'USING DenseMFAC')
        super(DenseMFAC, self).__init__(params, dict(lr=lr))

        with torch.no_grad():
            w = []
            for group in self.param_groups:
                for p in group['params']:
                    w.append(p.reshape(-1))
            w = torch.cat(w).to(self.dev)
            print(f'Full Model Size: {w.numel()}')

            if self.sparse:
                self.sparse_update = torch.zeros_like(w)
                self.sparsity_mask = w != 0
                w = w[self.sparsity_mask]
                print(f'Pruned Model Size: {w.numel()}')
                print(f'Sparsity Mask Size: {self.sparsity_mask.size()}')

            self.model_size = w.numel()

            if self.momentum > 0:
                self.v = torch.zeros(self.model_size, device=self.dev)

            if gpus is None or len(gpus) == 0:
                self.gpus = [self.dev]

            self.hinv = HInvFastUpMulti(
                grads=torch.zeros((ngrads, self.model_size), dtype=torch.float),
                dev=self.dev,
                gpus=self.gpus,
                damp=damp)

        # MyLogger.get('optimizer').log(message=f'{str(self)}').close()

    def set_named_parameters(self, named_parameters):
        self.named_parameters = named_parameters

    def __str__(self):
        return f'm={self.m}\n' \
               f'momentum={self.momentum}\n' \
               f'lr={self.lr}\n' \
               f'weight_decay={self.weight_decay}\n' \
               f'damp={self.damp}\n' \
               f'dev={self.dev}\n' \
               f'model_size={self.model_size}\n'

    @torch.no_grad()
    def step(self, closure=None):
        self.steps += 1

        if self.wd_type == 'wd':
            g = get_weights_and_gradients(self.param_groups, get_weights=False)
            w = None
        elif self.wd_type in ['reg', 'both']:
            w, g = get_weights_and_gradients(self.param_groups, get_weights=True)

        if self.sparse:
            g = g[self.sparsity_mask]
            if w is not None:
                w = w[self.sparsity_mask]

        if self.wd_type == 'wd':
            update = self.hinv.integrate_gradient_and_precondition(g, x=g)
        elif self.wd_type in ['reg', 'both']:
            update = self.hinv.integrate_gradient_and_precondition(g, x=g + self.weight_decay * w)

        if self.momentum > 0:
            self.v = self.momentum * self.v + update
            update = self.v

        update = update.to(self.dev)

        if self.sparse:
            self.sparse_update[self.sparsity_mask] = update
            update = self.sparse_update

        shrinking_factor = update_model(
            params=self.param_groups,
            update=update,
            wd_type=self.wd_type,
            alpha=None)

        lr = self.param_groups[0]['lr']
        self.wandb_data.update({f'norm_upd_w_lr': lr * update.norm(p=2), f'shrinking_factor': shrinking_factor})
        self.wandb_data.update(self.hinv.wandb_data)
        # self.wandb_data.update(quantify_preconditioning(g=g, u=update.to(g.device), return_distribution=False, use_abs=True)
        # self.wandb_data.update(get_different_params_norm(self.named_parameters))
        wandb.log(self.wandb_data)
