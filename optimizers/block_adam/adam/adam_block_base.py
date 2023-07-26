import math

import torch

from helpers.optim import get_weights, get_gradients, update_model
from helpers.tools import get_first_device, get_gpus
from optimizers.block_adam.blocks import SimpleBlock


class BaseBlockAdam(torch.optim.Optimizer):
    """
        Base class for all Block Adam optimizers
    """
    def __init__(self, params, block_size, lr, beta1, beta2, weight_decay, eps, compute_on_gpu, sparse=False):
        super(BaseBlockAdam, self).__init__(
            params,
            defaults=dict(
                lr=lr,
                beta1=beta1,
                beta2=beta2,
                weight_decay=weight_decay,
                eps=eps))

        self.block_size = block_size
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.eps = eps
        self.compute_on_gpu = compute_on_gpu
        self.sparse = sparse

        self.steps = 0
        self.mom1 = None
        self.mom2 = None
        self.block_updates = [] # this will hold the update slices
        self.main_device = get_first_device()
        self.gpus = get_gpus(remove_first=False)

        w = get_weights(self.param_groups)

        if self.sparse:
            self.sparse_update = torch.zeros_like(w)
            self.sparsity_mask = w != 0
            w = w[self.sparsity_mask]
            print(f'Pruned Model Size: {w.numel()}')
            print(f'Sparsity Mask Size: {self.sparsity_mask.size()}')

        self.model_size = w.numel()

        self.block_indices = self._get_block_indices()
        self._init_moments()

    def _init_moments(self):
        """This method initializes the second order momentum blocks with zeros"""
        self.mom1 = torch.zeros(self.model_size, dtype=torch.float, device=self.main_device)
        self.mom2 = []
        n_gpus = len(self.gpus)
        for index, indices in self.block_indices.items():
            block = SimpleBlock(
                start=indices['start'],
                end=indices['end'],
                beta2=self.beta2,
                eps=self.eps,
                main_device=self.main_device if self.compute_on_gpu else 'cpu',
                compute_device=self.gpus[index % n_gpus] if self.compute_on_gpu else 'cpu',
                index=index)
            self.mom2.append(block)

    def _get_block_indices(self):
        """
        This method computes the start and end indices of each block
        :return: a list of DiagonalBlock
        """
        count = 0
        index = 0
        blocks = dict()
        while count < self.model_size:
            right = min(self.model_size, count + self.block_size)
            blocks[index] = dict(start=count, end=right)
            count += self.block_size
            index += 1
        return blocks

    def _get_unbiased_lr(self):
        lr = self.param_groups[0]['lr']
        ub1 = self.beta1 ** self.steps - 1 # this incorporates the minus sign
        ub2 = math.sqrt(1 - self.beta2 ** self.steps)
        return lr * ub2 / ub1

    @torch.no_grad()
    def step_intro(self):
        self.steps += 1
        g = get_gradients(self.param_groups)
        if self.sparse:  # keep only non-pruned weights
            g = g[self.sparsity_mask]
        self.mom1.mul_(self.beta1).add_(g, alpha=1 - self.beta1)

    @torch.no_grad()
    def step_body(self):
        pass

    @torch.no_grad()
    def step_outro(self):
        update = torch.cat(self.block_updates).to(self.main_device)
        if self.sparse:
            self.sparse_update[self.sparsity_mask] = update
            update = self.sparse_update
        update_model(
            params=self.param_groups,
            update=update,
            wd_type='wd',
            weight_decay=self.weight_decay,
            alpha=self._get_unbiased_lr())

    @torch.no_grad()
    def step(self):
        self.step_intro()
        self.step_body()
        self.step_outro()
