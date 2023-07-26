from torch.nn.parallel import parallel_apply
import multiprocessing as mp
from helpers.matrix_functions import *

from optimizers.block_adam.adam import BaseBlockAdam
from optimizers.block_adam.utils import worker_gpu, worker_cpu


class SimpleBlockAdamGPU(BaseBlockAdam):
    def __init__(self, params, block_size, lr, beta1, beta2, weight_decay, eps=1e-4):
        super(SimpleBlockAdamGPU, self).__init__(params, block_size, lr, beta1, beta2, weight_decay, eps, compute_on_gpu=True)
        print(f'Using BlockAdamGPU')

    @torch.no_grad()
    def step_body(self):
        inputs = [
            (block, self.mom1[block.start:block.end].contiguous().to(block.compute_device))
            for block in self.mom2
        ]
        self.block_updates = parallel_apply(modules=[worker_gpu] * len(self.mom2), inputs=inputs)


class SimpleBlockAdamCPU(BaseBlockAdam):
    def __init__(self, params, block_size, lr, beta1, beta2, weight_decay, eps=1e-4):
        super(SimpleBlockAdamCPU, self).__init__(params, block_size, lr, beta1, beta2, weight_decay, eps, compute_on_gpu=False)
        print(f'Using BlockAdamCPU')
        mp.set_start_method('spawn')
        self.pool = mp.Pool(processes=int(mp.cpu_count() * 0.50))

    def __del__(self):
        self.pool.close()
        self.pool.join()

    @torch.no_grad()
    def step_body(self):
        inputs = [
            (block, self.mom1[block.start:block.end].contiguous().to(block.compute_device))
            for block in self.mom2
        ]
        """
            From https://stackoverflow.com/questions/41273960/:
            - map runs workers with arbitrary inputs
            - map returns the result in the same order as inputs
        """
        self.block_updates = self.pool.map(func=worker_cpu, iterable=inputs)

"""
Existing torch installation: 1.12.1+cu116
Newer versions:
    - 2.0.1+cu117
    - RuntimeError: CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling `cublasCreate(handle)`
"""

### from Block class:
# def _precondition(self, x):
#     const = self.eps * self._unbiasing_term()
#     A = self.buffer + const * self.eye
#
#     # fct = 'newton'
#     # fct = 'evd'
#     fct = 'evdh'
#
#     if fct == 'newton':
#         inv_p_A = compute_inv_p_power_coupled_newton(A, p=2)
#     else:
#         inv_p_A = compute_matrix_power_evd(A, p=-0.5, t=fct)
#
#     if self.steps == 5:
#         wandb.log(dict(inv_method=fct))
#
#     inv_sqrt_mul_g = inv_p_A @ x
#     return inv_sqrt_mul_g