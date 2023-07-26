import torch
from torch import bmm

from optimizers.block_adam.blocks import BaseBlock
# from optimizers.block_adam.utils import get_batch_identity


class BatchBlock(BaseBlock):
    def __init__(self, start, end, beta2, eps, main_device, compute_device, block_size, batches):
        super(BatchBlock, self).__init__(start, end, beta2, eps, main_device, compute_device)
        self.block_size = block_size
        self.batches = batches

        self.V = torch.zeros(self.batches, self.block_size, self.block_size, dtype=torch.float, device=self.compute_device)
        # self.batch_eye = get_batch_identity(B=batches, size=self.block_size, device=self.compute_device)

    def _update(self, g):
        """
            https://stackoverflow.com/questions/26089893/understanding-numpys-einsum
        """
        self.V.mul_(self.beta2).add_(other=torch.einsum('bi,bj->bij', (g, g)), alpha=1-self.beta2)

    def _precondition(self, x):
        c = self.eps * self._unbiasing_term()
        L, Q = torch.linalg.eigh(self.V)
        L = L.unsqueeze(2)
        torch.nn.functional.relu(L, inplace=True)
        # L, Q = L.type(torch.float), Q.type(torch.float)
        sqrtL = L.sqrt()
        invSqrtL = 1 / (sqrtL + c)
        x = x.unsqueeze(2)
        # print()
        # print(f'V.size() = {V.size()}')
        # print(f'invSqrtL.size() = {invSqrtL.size()}')
        # print(f'V.T.size() = {torch.transpose(V, dim0=1, dim1=2).size()}')
        # print(f'x.size() = {x.size()}')
        # print()
        inv_sqrt_mul_x = bmm(Q, bmm(invSqrtL * torch.transpose(Q, dim0=1, dim1=2), x)).view(-1)
        return inv_sqrt_mul_x.to(self.main_device)
