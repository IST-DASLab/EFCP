import torch


def get_batch_identity(B, size, device):
    """Returns a batch of B identity matrices, each having size n x n"""
    eye = torch.eye(size, dtype=torch.float, device=device)
    batch_eye = eye.repeat(B, 1, 1)
    return batch_eye


def worker_gpu(block, g_block):
    block_update = block.update_then_precondition(g_block)
    return block_update


def worker_cpu(params):
    return worker_gpu(*params)
