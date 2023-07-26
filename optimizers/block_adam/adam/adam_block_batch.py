import torch
from torch.nn.parallel import parallel_apply

from optimizers.block_adam.adam import BaseBlockAdam
from optimizers.block_adam.blocks import BatchBlock
from optimizers.block_adam.utils import worker_gpu


class BatchBlockAdam(BaseBlockAdam):
    def __init__(self, params, batch_splits, block_size, lr, beta1, beta2, weight_decay, eps=1e-4, sparse=False):
        self.batch_splits = batch_splits
        super(BatchBlockAdam, self).__init__(params, block_size, lr, beta1, beta2, weight_decay, eps, compute_on_gpu=True, sparse=sparse)
        print(f'Using BatchBlockAdam: batch_splits = {self.batch_splits}')

    def _init_moments(self):
        """
            This method initializes the second order momentum blocks with zeros
            We will always have 2 or more blocks (one will be of smaller size)
            Example:
                - ResNet-20 for CIFAR-10: 272_474 parameters
                - Block size 1024 => 266 blocks of size 1024 x 1024 and one small block of size 90 x 90 (the rest)
                - 1 GPU:
                    * (266, 1024, 1024) - indices 0 to 266 * 1024)
                    * (1, 90, 90) - indices 266 * 1024 to 266 * 1024 + 90
                - 2 GPUs:
                    * (133, 1024, 1024) - indices (             0, 133 * 1024)
                    * (133, 1024, 1024) - indices (    133 * 1024, 2 * 133 * 1024)
                    * (  1,   90,   90) - indices (2 * 133 * 1024, 2 * 133 * 1024 + 90)
                - 3 GPUs:
                    * (89, 1024, 1024) - indices (         0, 89 * 1024)
                    * (89, 1024, 1024) - indices ( 89 * 1024, 178 * 1024)
                    * (88, 1024, 1024) - indices (178 * 1024, 266 * 1024)
                    * ( 1,   90,   90) - indices (266 * 1024, 266 * 1024 + 90)
        """
        self.mom1 = torch.zeros(self.model_size, dtype=torch.float, device=self.main_device)
        self.mom2 = []
        n_gpus = len(self.gpus)
        # n_splits = self.batch_splits

        for index, info in self.block_indices.items():
            block = BatchBlock(
                start=info['start'],
                end=info['end'],
                beta2=self.beta2,
                eps=self.eps,
                main_device=self.main_device if self.compute_on_gpu else 'cpu',
                compute_device=self.gpus[index % n_gpus] if self.compute_on_gpu else 'cpu',
                block_size=info['block_size'],
                batches=info['batch_size'])
            self.mom2.append(block)

    def _get_block_indices(self):
        """This method computes the start and end indices of each block, as well as the batch size"""
        n_blocks = self.model_size // self.block_size
        b = self.model_size - n_blocks * self.block_size  # size of the smallest block
        # n_gpus = 4 # self.n_blocks
        # batch_splits = len(self.gpus)

        B = [n_blocks // self.batch_splits for _ in range(self.batch_splits)]  # batch sizes for each GPU

        rest = n_blocks - sum(B)
        for i in range(rest):
            B[i] += 1

        count = 0
        index = 0
        blocks = dict()
        for bb in B:
            offset = bb * self.block_size # count + offset will never be greater than model size
            blocks[index] = dict(start=count, end=count + offset, batch_size=bb, block_size=self.block_size)
            count += offset
            index += 1
        blocks[index] = dict(start=count, end=count + b, batch_size=1, block_size=b)

        print()
        print(f'block_size = {self.block_size}')
        print(f'n_blocks = {n_blocks}')
        print(f'batch_splits = {self.batch_splits}')
        print(f'model_size = {self.model_size}')
        print(f'B = {B}')
        print(f'b = {b}')
        print(f'blocks = {blocks}')
        print()
        import time
        from tqdm import tqdm
        for _ in tqdm(range(5)):
            time.sleep(1)

        return blocks

    @torch.no_grad()
    def step_body(self):
        inputs = [
            (block, self.mom1[block.start:block.end].reshape(block.batches, block.block_size).to(block.compute_device))
            for block in self.mom2
        ]
        # self.block_updates = parallel_apply(modules=[worker_gpu] * len(self.mom2), inputs=inputs)
        self.block_updates = [worker_gpu(b, m) for b, m in inputs]
