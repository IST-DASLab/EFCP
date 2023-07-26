import torch
from torch import matmul as mm

from optimizers.block_adam.blocks import BaseBlock


class SimpleBlock(BaseBlock):
    def __init__(self, start, end, beta2, eps, main_device, compute_device, index):
        super(SimpleBlock, self).__init__(start, end, beta2, eps, main_device, compute_device)
        self.index = index

        self.V = torch.zeros(self.size, self.size, dtype=torch.float, device=self.compute_device)
        self.eye = torch.eye(self.size, dtype=torch.float, device=self.compute_device)

        # """
        #   Initialize the torch.linalg context by performing a dummy operation once before calling torch.linalg.eigh
        # in the GPU threads, otherwise an annoying error related to lazy context will be raised and this dummy call
        # seems to solve the issue
        # """
        # if index == 5:
        #     torch.linalg.eigh(self.eye)

    def _update(self, g):
        self.V.addr_(vec1=g, vec2=g, beta=self.beta2, alpha=1 - self.beta2)

    def _precondition(self, x):
        const = self.eps * self._unbiasing_term()
        L, V = torch.linalg.eig(self.V + const * self.eye)
        L, V = L.type(torch.float), V.type(torch.float)
        invSqrtL = L.unsqueeze(1) ** -0.5
        inv_sqrt_mul_x = mm(V, mm(V.T * invSqrtL, x))
        return inv_sqrt_mul_x
