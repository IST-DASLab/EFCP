import torch

# Taken from PowerSGD implementation.
# https://github.com/epfml/powersgd
@torch.jit.script
def orthogonalize(matrix, eps=torch.tensor(1e-8)):
    _, m = matrix.shape
    for i in range(m):
        # Normalize the i'th column
        col = matrix[:, i : i + 1]
        col /= torch.sqrt(torch.sum(col ** 2)) + eps
        # Project it on the rest and remove it
        if i + 1 < m:
            rest = matrix[:, i + 1 :]
            rest -= torch.sum(col * rest, dim=0) * col

class PowerIterationLayerCompressor():
    def __init__(self, grad, rank=1, enable_error_correction=False):
        self.enable_error_correction = enable_error_correction
        self.grad_shape = grad.shape
        self.rank = rank
        if len(self.grad_shape) < 2:
            return
        remaining_size = 1
        for size in self.grad_shape[1:]:
            remaining_size *= size

        self.grad_shape = (self.grad_shape[0], remaining_size)
        self.p_memory = torch.empty(self.grad_shape[0], rank, device=grad.device)
        self.q_memory = torch.randn(self.grad_shape[1], rank, device=grad.device)

    def compress_op(self, grad):
        if len(self.grad_shape) < 2:
            left = grad.unsqueeze(-1).expand(-1, self.rank)
            right = torch.tensor([[1.0] + [0.0] * (self.rank - 1)], device=grad.device)
            return left, right
        grad_2d = grad.reshape(grad.shape[0], -1)
        torch.matmul(grad_2d, self.q_memory, out=self.p_memory)
        orthogonalize(self.p_memory)
        torch.matmul(grad_2d.t(), self.p_memory, out=self.q_memory)
        return self.p_memory, self.q_memory


    def decompres_op(self, p, q, out):
        torch.matmul(p, q.t(), out=out.reshape(p.shape[0], q.shape[0]))


    def compress(self, grad, state):
        if len(grad.shape) < 2:
            return self.compress_op(grad)
        if self.enable_error_correction:
            if "error_correction" not in state:
                state["error_correction"] = torch.zeros_like(grad)
            e_c = state["error_correction"]
            e_c.add_(grad)
            grad.copy_(e_c)
        p, q = self.compress_op(grad)
        self.decompres_op(p, q, grad)
        if self.enable_error_correction:
            e_c.sub_(grad)
        return p, q


class PowerIterationRankOneLayerCompressor():
    def __init__(self, grad, enable_error_correction=False):
        self.enable_error_correction = enable_error_correction
        self.grad_shape = grad.shape
        if len(self.grad_shape) < 2:
            return
        remaining_size = 1
        for size in self.grad_shape[1:]:
            remaining_size *= size

        self.grad_shape = (self.grad_shape[0], remaining_size)
        self.p_memory = torch.empty(self.grad_shape[0], device=grad.device)
        self.q_memory = torch.randn(self.grad_shape[1], device=grad.device)

    def compress_op(self, grad):
        if len(self.grad_shape) < 2:
            return grad, torch.tensor([1.0], device=grad.device)
        self.p_memory = self.p_memory.to(grad.device)
        self.q_memory = self.q_memory.to(grad.device)
        grad_2d = grad.reshape(grad.shape[0], -1)
        torch.matmul(grad_2d, self.q_memory, out=self.p_memory)
        self.orthogonalize(self.p_memory)
        torch.matmul(grad_2d.t(), self.p_memory, out=self.q_memory)
        return self.p_memory, self.q_memory


    def decompres_op(self, p, q, out):
        torch.outer(p, q, out=out.reshape(p.shape[0], q.shape[0]))

    def orthogonalize(self, vec):
        if len(vec.shape) > 1:
            raise NotImplementedError("Orthogonalization with matrices of rank > 1 is not supported.")
        vec /= torch.norm(vec, p=2)


    def compress(self, grad, state):
        if len(grad.shape) < 2:
            return self.compress_op(grad)
        if self.enable_error_correction:
            if "error_correction" not in state:
                state["error_correction"] = torch.zeros_like(grad)
            e_c = state["error_correction"]
            e_c.add_(grad)
            grad.copy_(e_c)
        p, q = self.compress_op(grad)
        self.decompres_op(p, q, grad)
        if self.enable_error_correction:
            e_c.sub_(grad)
        return p, q

class TopKLayerCompressor():
    def __init__(self, grad, k_ratio=1.0, enable_error_correction=False):
        self.enable_error_correction = enable_error_correction
        self.k_ratio = k_ratio
        self.kth_value = int(grad.numel() * (1 - k_ratio))

    def compress_op(self, grad):
        threshold, _ = grad.ravel().abs().kthvalue(self.kth_value)
        grad[grad.abs() < threshold] = 0.0

    def decompres_op(self, grad):
        pass

    def compress(self, grad, state):
        if self.enable_error_correction:
            if "error_correction" not in state:
                state["error_correction"] = torch.zeros_like(grad)
            e_c = state["error_correction"]
            e_c.add_(grad)
            grad.copy_(e_c)
        self.compress_op(grad)
        self.decompres_op(grad)
        if self.enable_error_correction:
            e_c.sub_(grad)
        return grad 