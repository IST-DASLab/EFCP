import torch
import torch.nn as nn
import torch.nn.functional as F

from optimizers.lowrank.compressor import PowerIterationLayerCompressor, PowerIterationRankOneLayerCompressor

# Disable tensor cores as they can mess with precision
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# Use custom CUDA implementation for computing $B$ / `coef` if installed 
USE_CUDA = True 
try:
    import hinv_cuda
except Exception as e:
    USE_CUDA = False

# Full implementation of the dynamic algorithm with support for splitting gradients across GPUs
class RankOneHInvFastUpMulti:

    # `dev` is the device where all the coefficient calculation happens
    # `left_grads` ... initial $m \times k$ matrix $R$ of left vectors of gradient; assumed to be on CPU (can be deleted afterwards)
    # `right_grads` ... initial $m \times l$ gradient matrix $Q$; assumed to be on CPU (can be deleted afterwards)
    # `dev` ... device where coefficient calculation happens
    # `gpus` ... list of GPUs across which to split stored gradients 
    # `damp` ... dampening constant $\lambda$
    def __init__(self, left_grads, right_grads, dev, gpus, damp=1e-5):
        self.m, self.k = left_grads.shape
        m, self.l = right_grads.shape
        if self.m != m :
            return ValueError("Incompatable left and right gradient matrices with shapes:", 
                left_grads.shape, right_grads.shape)
        self.dev = dev 
        self.gpus = gpus
        self.dtype = left_grads.dtype
        self.lambd = 1. / damp


        if len(gpus) > 1:
            raise NotImplementedError("MultiGPU is not yet supported for low rank.")

        if USE_CUDA and self.m % 32 != 0 or self.m > 1024:
            raise ValueError('CUDA implementation currently on supports $m$ < 1024 and divisible by 32.')

        # Matrix $R$
        self.left_grads = left_grads.to(gpus[0])
        # Matrix $Q$
        self.right_grads = right_grads.to(gpus[0])
        # Compute matrix $GG^T$
        self.dots = (self.left_grads @ self.left_grads.t()) * (self.right_grads @ self.right_grads.t())

        self.last = 0 # ringbuffer index
        self.giHig = self.lambd * self.dots # matrix $D$
        self.denom = torch.zeros(self.m, device=self.dev, dtype=self.dtype) # $D_ii + m$ 
        self.coef = self.lambd * torch.eye(self.m, device=self.dev, dtype=self.dtype) # matrix $B$
        self.setup()

    # Calculate $D$ / `giHig` and $B$ / `coef`
    def setup(self):
        self.giHig = self.lambd * self.dots
        diag = torch.diag(torch.full([self.m], self.m, device=self.dev, dtype=self.dtype))
        self.giHig = torch.lu(self.giHig + diag, pivot=False)[0]
        self.giHig = torch.triu(self.giHig - diag)
        self.denom = self.m + torch.diagonal(self.giHig)
        tmp = -self.giHig.t().contiguous() / self.denom.reshape((1, -1))

        if USE_CUDA:
            diag = torch.diag(torch.full([self.m], self.lambd, device=self.dev, dtype=self.dtype))
            self.coef = hinv_cuda.hinv_setup(tmp, diag)
        else:
            for i in range(max(self.last, 1), self.m):
                self.coef[i, :i] = tmp[i, :i].matmul(self.coef[:i, :i])

    # Not-yet distributed `grads.matmul(x)`
    def grads_matmul(self, x_left, x_right):
        return (self.left_grads @ x_left) * (self.right_grads @ x_right)

    # Not-yet distributed linear combination of stored gradients with `coefs` coefficients.
    def grads_linear_sum(self, coefs):
        return (coefs.reshape(1, -1) * self.left_grads.t()) @ self.right_grads

    # Not-yet distributed `grads[j, :] = g`
    def set_grad(self, j, g_left, g_right):
        self.left_grads[j] = g_left
        self.right_grads[j] = g_right

    # Product with inverse of dampened empirical Fisher
    def mul(self, x_left, x_right, dots=None):
        if dots is None:
            dots = self.grads_matmul(x_left, x_right)
        giHix = self.lambd * dots 
        if USE_CUDA:
            giHix = hinv_cuda.hinv_mul(self.giHig, giHix)
        else:
            for i in range(1, self.m):
                giHix[i:].sub_(self.giHig[i - 1, i:], alpha=giHix[i - 1] / self.denom[i - 1])
        return self.lambd * torch.outer(x_left, x_right) - self.grads_linear_sum((giHix / self.denom).matmul(self.coef))

    # Replace oldest gradient with `g` 
    def update(self, g_left, g_right):
        self.set_grad(self.last, g_left, g_right)
        tmp = self.grads_matmul(g_left, g_right)
        self.dots[self.last, :] = tmp
        self.dots[:, self.last] = tmp
        self.setup()
        self.last = (self.last + 1) % self.m
        return tmp

    # Replace oldest gradient with `g` and then calculate the IHVP with `g`
    def update_mul(self, g_left, g_right):
        dots = self.update(g_left, g_right)
        return self.mul(g_left, g_right, dots)

# Dynamic algorithm with scalar product matrices already computed.
class RankOneHInvFastScalarProds:

    # `dots` a matrix of scalar products to initialize the algorithm
    # `damp` ... dampening constant $\lambda$
    def __init__(self, dots, damp=1e-5):
        self.dev = dots.device
        self.dtype = dots.dtype
        self.m = dots.shape[0]
        # Ring buffer index
        self.last = 0

        self.lambd = 1. / damp
        # Compute matrix $GG^T$
        self.dots = dots
        self.giHig = self.lambd * self.dots # matrix $D$
        self.denom = torch.zeros(self.m, device=self.dev, dtype=self.dtype) # $D_ii + m$ 
        self.coef = self.lambd * torch.eye(self.m, device=self.dev, dtype=self.dtype) # matrix $B$
        self.setup()

    # Calculate $D$ / `giHig` and $B$ / `coef`
    def setup(self):
        self.giHig = self.lambd * self.dots
        diag = torch.diag(torch.full([self.m], self.m, device=self.dev, dtype=self.dtype))
        self.giHig = torch.lu(self.giHig + diag, pivot=False)[0]
        self.giHig = torch.triu(self.giHig - diag)
        self.denom = self.m + torch.diagonal(self.giHig)
        tmp = -self.giHig.t().contiguous() / self.denom.reshape((1, -1))

        diag = torch.diag(torch.full([self.m], self.lambd, device=self.dev, dtype=self.dtype))
        self.coef = hinv_cuda.hinv_setup(tmp, diag)

    # Product with inverse of dampened empirical Fisher
    def get_coefs(self, dots):
        giHix = self.lambd * dots.clone()
        for i in range(1, self.m):
            giHix[i:].sub_(self.giHig[i - 1, i:], alpha=giHix[i - 1] / self.denom[i - 1])
        return (giHix / self.denom).matmul(self.coef)

    # Replace oldest scalar product with `dot` 
    def update(self, dot):
        self.dots[self.last, :] = dot
        self.dots[:, self.last] = dot
        self.setup()
        self.last = (self.last + 1) % self.m
        return dot

# Full implementation of the dynamic algorithm with support for splitting gradients across GPUs
class LowRankHInvFastUpMulti:

    # `dev` is the device where all the coefficient calculation happens
    # `left_grads` ... initial $m \times k \times r$ matrix $R$ of left vectors of gradient; assumed to be on CPU (can be deleted afterwards)
    # `right_grads` ... initial $m \times l \times r$ gradient matrix $Q$; assumed to be on CPU (can be deleted afterwards)
    # `dev` ... device where coefficient calculation happens
    # `gpus` ... list of GPUs across which to split stored gradients 
    # `damp` ... dampening constant $\lambda$
    def __init__(self, left_grads, right_grads, dev, gpus, damp=1e-5):
        self.m, self.k, self.rank = left_grads.shape
        m, self.l, r = right_grads.shape
        if self.m != m  or self.rank != r:
            return ValueError("Incompatable left and right gradient matrices with shapes:", 
                left_grads.shape, right_grads.shape)
        self.dev = dev 
        self.gpus = gpus
        self.dtype = left_grads.dtype
        self.lambd = 1. / damp


        if len(gpus) > 1:
            raise NotImplementedError("MultiGPU is not yet supported for low rank.")

        if USE_CUDA and self.m % 32 != 0 or self.m > 1024:
            raise ValueError('CUDA implementation currently on supports $m$ < 1024 and divisible by 32.')

        # Matrix $R$
        self.left_grads = left_grads.transpose(1, 2).reshape(self.m * self.rank, self.k).to(gpus[0])
        # Matrix $Q$
        self.right_grads = right_grads.transpose(1, 2).reshape(self.m * self.rank, self.l).to(gpus[0])
        # Compute matrix $GG^T$
        scattered_dots = (self.left_grads @ self.left_grads.t()) * (self.right_grads @ self.right_grads.t())
        self.kernel = torch.ones(1, 1, self.rank, self.rank, dtype=scattered_dots.dtype, device=scattered_dots.device)
        self.dots = F.conv2d(scattered_dots.unsqueeze(0).unsqueeze(0), self.kernel, stride=self.rank, padding=0)
        self.dots = self.dots.squeeze()

        self.last = 0 # ringbuffer index
        self.giHig = self.lambd * self.dots # matrix $D$
        self.denom = torch.zeros(self.m, device=self.dev, dtype=self.dtype) # $D_ii + m$ 
        self.coef = self.lambd * torch.eye(self.m, device=self.dev, dtype=self.dtype) # matrix $B$
        self.setup()

    # Calculate $D$ / `giHig` and $B$ / `coef`
    def setup(self):
        self.giHig = self.lambd * self.dots
        diag = torch.diag(torch.full([self.m], self.m, device=self.dev, dtype=self.dtype))
        self.giHig = torch.lu(self.giHig + diag, pivot=False)[0]
        self.giHig = torch.triu(self.giHig - diag)
        self.denom = self.m + torch.diagonal(self.giHig)
        tmp = -self.giHig.t().contiguous() / self.denom.reshape((1, -1))

        if USE_CUDA:
            diag = torch.diag(torch.full([self.m], self.lambd, device=self.dev, dtype=self.dtype))
            self.coef = hinv_cuda.hinv_setup(tmp, diag)
        else:
            for i in range(max(self.last, 1), self.m):
                self.coef[i, :i] = tmp[i, :i].matmul(self.coef[:i, :i])

    # Not-yet distributed `grads.matmul(x)`
    def grads_matmul(self, x_left, x_right):
        scattered_dots = (self.left_grads @ x_left) * (self.right_grads @ x_right)
        dots = F.conv2d(scattered_dots.unsqueeze(0).unsqueeze(0), self.kernel, stride=self.rank, padding=0)
        return dots.squeeze()

    # Not-yet distributed linear combination of stored gradients with `coefs` coefficients.
    def grads_linear_sum(self, coefs):
        left = self.left_grads.reshape(self.m, self.rank, self.k).transpose(1, 2)
        right = self.right_grads.reshape(self.m, self.rank, self.l)
        dummy_input = torch.zeros(1, dtype=left.dtype, device=left.device)
        return torch.addbmm(input=dummy_input, batch1=(coefs.reshape(-1, 1, 1) * left),
            batch2=right, beta=0)

    # Not-yet distributed `grads[j, :] = g`
    def set_grad(self, j, g_left, g_right):
        self.left_grads[j * self.rank : (j + 1) * self.rank] = g_left.t()
        self.right_grads[j * self.rank : (j + 1) * self.rank] = g_right.t()

    # Product with inverse of dampened empirical Fisher
    def mul(self, x_left, x_right, dots=None):
        if dots is None:
            dots = self.grads_matmul(x_left, x_right)
        giHix = self.lambd * dots 
        if USE_CUDA:
            giHix = hinv_cuda.hinv_mul(self.giHig, giHix)
        else:
            for i in range(1, self.m):
                giHix[i:].sub_(self.giHig[i - 1, i:], alpha=giHix[i - 1] / self.denom[i - 1])
        return self.lambd * x_left @ x_right.t() - self.grads_linear_sum((giHix / self.denom).matmul(self.coef))

    # Replace oldest gradient with `g` 
    def update(self, g_left, g_right):
        self.set_grad(self.last, g_left, g_right)
        tmp = self.grads_matmul(g_left, g_right)
        self.dots[self.last, :] = tmp
        self.dots[:, self.last] = tmp
        self.setup()
        self.last = (self.last + 1) % self.m
        return tmp

    # Replace oldest gradient with `g` and then calculate the IHVP with `g`
    def update_mul(self, g_left, g_right):
        dots = self.update(g_left, g_right)
        return self.mul(g_left, g_right, dots)


# PyTorch compatible implementation of the M-FAC optimizer 
class RankOneMFAC(torch.optim.Optimizer):

    # `params` ... model parameters to optimize 
    # `lr` ... learning rate
    # `momentum` ... momentum coefficient
    # `weight_decay` ... weight decay constant
    # `ngrads` ... size of gradient window 
    # `damp` ... dampening constant $\lambda$
    # `moddev` ... device where the model to be optimized resides 
    # `optdev` ... device where coefficient calculation of the dynamic algorithm happens
    def __init__(
        self, params, 
        lr=1e-3, momentum=0, weight_decay=0, 
        ngrads=1024, damp=1e-5,
        moddev=None, optdev=None
    ):
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.moddev = moddev
        self.optdev = optdev

        super().__init__(params, dict(lr=lr))

        self.compressors = []
        self.vs = []
        self.hinvs = []
        self.states = []
        with torch.no_grad():
            for group in self.param_groups:
                self.compressors.append([])
                self.vs.append([])
                self.hinvs.append([])
                self.states.append([])

                for p in group['params']:
                    self.compressors[-1].append(PowerIterationLayerCompressor(
                        p, enable_error_correction=True
                    ))
                    if self.momentum > 0:
                        self.vs[-1].append(torch.zeros(p.shape, device=self.optdev))
                    left_grad, right_grad = self.compressors[-1][-1].compress_op(p)
                    self.hinvs[-1].append(RankOneHInvFastUpMulti(
                        torch.zeros((ngrads, *left_grad.shape), dtype=torch.float), 
                        torch.zeros((ngrads, *right_grad.shape), dtype=torch.float), 
                        dev=self.optdev, gpus=[self.optdev], damp=damp
                    ))
                    self.states[-1].append({})

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError('`closure` not supported')

        for group_id, group in enumerate(self.param_groups):
            for param_id, p in enumerate(group['params']):
                compressor = self.compressors[group_id][param_id]
                hinv = self.hinvs[group_id][param_id]
                state = self.states[group_id][param_id]
                if self.momentum > 0:
                    v = self.vs[group_id][param_id]

                g = p.grad
                if self.weight_decay > 0:
                    g += self.weight_decay * p
                g = g.to(self.optdev)

                ## Decompose gradient
                g_left, g_right = compressor.compress(g, state)

                update = hinv.update_mul(g_left, g_right)
                if self.momentum > 0:
                    self.vs[group_id][param_id] = self.momentum * v + (1 - self.momentum) * update
                    update = self.vs[group_id][param_id] 
                update = update.to(self.moddev)
                p.add_(update.reshape(p.shape), alpha=-group['lr'])

# PyTorch compatible implementation of the M-FAC optimizer 
class RankOneAggregateMFAC(torch.optim.Optimizer):

    # `params` ... model parameters to optimize 
    # `lr` ... learning rate
    # `momentum` ... momentum coefficient
    # `weight_decay` ... weight decay constant
    # `num_grads` ... size of gradient window 
    # `damp` ... dampening constant $\lambda$
    # `moddev` ... device where the model to be optimized resides 
    # `optdev` ... device where coefficient calculation of the dynamic algorithm happens
    def __init__(
        self, params, 
        lr=1e-3, momentum=0, weight_decay=0, 
        num_grads=1024, damp=1e-5,
        moddev=None, optdev=None
    ):
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.moddev = moddev
        self.optdev = optdev
        self.current_step = 0

        super().__init__(params, dict(lr=lr))

        self.compressors = []
        self.vs = []
        self.hinvs = []
        self.states = []
        self.hcoef = RankOneHInvFastScalarProds(
            torch.zeros(num_grads, num_grads, dtype=torch.float, device=self.optdev),
            damp=damp
        )
        with torch.no_grad():
            for group in self.param_groups:
                self.compressors.append([])
                self.vs.append([])
                self.hinvs.append([])
                self.states.append([])

                for p in group['params']:
                    self.compressors[-1].append(PowerIterationRankOneLayerCompressor(
                        p, enable_error_correction=True
                    ))
                    if self.momentum > 0:
                        self.vs[-1].append(torch.zeros(p.shape, device=self.optdev))
                    left_grad, right_grad = self.compressors[-1][-1].compress_op(p)
                    self.hinvs[-1].append(RankOneHInvFastUpMulti(
                        torch.zeros((num_grads, *left_grad.shape), dtype=torch.float), 
                        torch.zeros((num_grads, *right_grad.shape), dtype=torch.float), 
                        dev=self.optdev, gpus=[self.optdev], damp=damp
                    ))
                    self.states[-1].append({})

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError('`closure` not supported')

        # Aggregate low rank updates from different layers.
        dot_update = None
        grad_dot = None
        left_grads = []
        right_grads = []
        for group_id, group in enumerate(self.param_groups):
            left_grads.append([])
            right_grads.append([])
            for param_id, p in enumerate(group['params']):
                compressor = self.compressors[group_id][param_id]
                hinv = self.hinvs[group_id][param_id]
                state = self.states[group_id][param_id]

                g = p.grad
                if self.weight_decay > 0:
                    g += self.weight_decay * p
                g = g.to(self.optdev)

                ## Decompose gradient
                g_left, g_right = compressor.compress(g, state)
                left_grads[-1].append(g_left)
                right_grads[-1].append(g_right)

                dot = hinv.update(g_left, g_right)
                if dot_update is None:
                    dot_update = dot
                else:
                    dot_update += dot

                g_dot = hinv.grads_matmul(g_left, g_right)
                if grad_dot is None:
                    grad_dot = g_dot
                else:
                    grad_dot += g_dot
        self.hcoef.update(dot_update)
        coefs = self.hcoef.get_coefs(grad_dot)


        for group_id, group in enumerate(self.param_groups):
            for param_id, p in enumerate(group['params']):
                g_left = left_grads[group_id][param_id]
                g_right = right_grads[group_id][param_id]
                hinv = self.hinvs[group_id][param_id]
                if self.momentum > 0:
                    v = self.vs[group_id][param_id]

                # Compute update
                update = (hinv.lambd * torch.outer(g_left, g_right) -
                 hinv.grads_linear_sum(coefs))
                if self.momentum > 0:
                    self.vs[group_id][param_id] = self.momentum * v + (1 - self.momentum) * update
                    update = self.vs[group_id][param_id] 
                update = update.to(self.moddev)
                p.add_(update.reshape(p.shape), alpha=-group['lr'])

        self.current_step += 1

# PyTorch compatible implementation of the M-FAC optimizer 
class LowRankAggregateMFAC(torch.optim.Optimizer):

    # `params` ... model parameters to optimize 
    # `lr` ... learning rate
    # `momentum` ... momentum coefficient
    # `weight_decay` ... weight decay constant
    # `num_grads` ... size of gradient window 
    # `damp` ... dampening constant $\lambda$
    # `moddev` ... device where the model to be optimized resides 
    # `optdev` ... device where coefficient calculation of the dynamic algorithm happens
    def __init__(
        self, params, 
        lr=1e-3, momentum=0, weight_decay=0, 
        num_grads=1024, damp=1e-5,
        rank=1,
        moddev=None, optdev=None
    ):
        print(f'Training with rank {rank}')
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.moddev = moddev
        self.optdev = optdev
        self.rank = rank
        self.current_step = 0

        super().__init__(params, dict(lr=lr))

        self.compressors = []
        self.vs = []
        self.hinvs = []
        self.states = []
        self.hcoef = RankOneHInvFastScalarProds(
            torch.zeros(num_grads, num_grads, dtype=torch.float, device=self.optdev),
            damp=damp
        )
        with torch.no_grad():
            for group in self.param_groups:
                self.compressors.append([])
                self.vs.append([])
                self.hinvs.append([])
                self.states.append([])

                for p in group['params']:
                    self.compressors[-1].append(PowerIterationLayerCompressor(
                        p, rank=self.rank, enable_error_correction=True
                    ))
                    if self.momentum > 0:
                        self.vs[-1].append(torch.zeros(p.shape, device=self.optdev))
                    left_grad, right_grad = self.compressors[-1][-1].compress_op(p)
                    self.hinvs[-1].append(LowRankHInvFastUpMulti(
                        torch.zeros((num_grads, *left_grad.shape), dtype=torch.float), 
                        torch.zeros((num_grads, *right_grad.shape), dtype=torch.float), 
                        dev=self.optdev, gpus=[self.optdev], damp=damp
                    ))
                    self.states[-1].append({})

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError('`closure` not supported')

        # Aggregate low rank updates from different layers.
        dot_update = None
        grad_dot = None
        left_grads = []
        right_grads = []
        for group_id, group in enumerate(self.param_groups):
            left_grads.append([])
            right_grads.append([])
            for param_id, p in enumerate(group['params']):
                compressor = self.compressors[group_id][param_id]
                hinv = self.hinvs[group_id][param_id]
                state = self.states[group_id][param_id]

                g = p.grad
                if self.weight_decay > 0:
                    g += self.weight_decay * p
                g = g.to(self.optdev)

                ## Decompose gradient
                g_left, g_right = compressor.compress(g, state)
                left_grads[-1].append(g_left)
                right_grads[-1].append(g_right)

                dot = hinv.update(g_left, g_right)
                if dot_update is None:
                    dot_update = dot
                else:
                    dot_update += dot

                g_dot = hinv.grads_matmul(g_left, g_right)
                if grad_dot is None:
                    grad_dot = g_dot
                else:
                    grad_dot += g_dot
        self.hcoef.update(dot_update)
        coefs = self.hcoef.get_coefs(grad_dot)


        for group_id, group in enumerate(self.param_groups):
            for param_id, p in enumerate(group['params']):
                g_left = left_grads[group_id][param_id]
                g_right = right_grads[group_id][param_id]
                hinv = self.hinvs[group_id][param_id]
                if self.momentum > 0:
                    v = self.vs[group_id][param_id]

                # Compute update
                update = (hinv.lambd * g_left @ g_right.t() -
                 hinv.grads_linear_sum(coefs))
                if self.momentum > 0:
                    self.vs[group_id][param_id] = self.momentum * v + (1 - self.momentum) * update
                    update = self.vs[group_id][param_id] 
                update = update.to(self.moddev)
                p.add_(update.reshape(p.shape), alpha=-group['lr'])

        self.current_step += 1

# PyTorch compatible implementation of the M-FAC optimizer 
class LowRankMFAC(torch.optim.Optimizer):

    # `params` ... model parameters to optimize 
    # `lr` ... learning rate
    # `momentum` ... momentum coefficient
    # `weight_decay` ... weight decay constant
    # `ngrads` ... size of gradient window 
    # `damp` ... dampening constant $\lambda$
    # `rank` ... rank of gradient decompositions
    # `moddev` ... device where the model to be optimized resides 
    # `optdev` ... device where coefficient calculation of the dynamic algorithm happens
    def __init__(
        self, params, 
        lr=1e-3, momentum=0, weight_decay=0, 
        ngrads=1024, damp=1e-5,
        rank=1,
        moddev=None, optdev=None
    ):
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.moddev = moddev
        self.optdev = optdev
        self.rank = rank

        super().__init__(params, dict(lr=lr))

        self.compressors = []
        self.vs = []
        self.hinvs = []
        self.states = []
        with torch.no_grad():
            for group in self.param_groups:
                self.compressors.append([])
                self.vs.append([])
                self.hinvs.append([])
                self.states.append([])

                for p in group['params']:
                    self.compressors[-1].append(PowerIterationLayerCompressor(
                        p, rank=rank, enable_error_correction=True
                    ))
                    if self.momentum > 0:
                        self.vs[-1].append(torch.zeros(p.shape, device=self.optdev))
                    left_grad, right_grad = self.compressors[-1][-1].compress_op(p)
                    self.hinvs[-1].append(LowRankHInvFastUpMulti(
                        torch.zeros((ngrads, *left_grad.shape), dtype=torch.float), 
                        torch.zeros((ngrads, *right_grad.shape), dtype=torch.float), 
                        dev=self.optdev, gpus=[self.optdev], damp=damp
                    ))
                    self.states[-1].append({})

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError('`closure` not supported')

        for group_id, group in enumerate(self.param_groups):
            for param_id, p in enumerate(group['params']):
                compressor = self.compressors[group_id][param_id]
                hinv = self.hinvs[group_id][param_id]
                state = self.states[group_id][param_id]
                if self.momentum > 0:
                    v = self.vs[group_id][param_id]

                g = p.grad
                if self.weight_decay > 0:
                    g += self.weight_decay * p
                g = g.to(self.optdev)

                ## Decompose gradient
                g_left, g_right = compressor.compress(g, state)

                update = hinv.update_mul(g_left, g_right)
                if self.momentum > 0:
                    self.vs[group_id][param_id] = self.momentum * v + (1 - self.momentum) * update
                    update = self.vs[group_id][param_id] 
                update = update.to(self.moddev)
                p.add_(update.reshape(p.shape), alpha=-group['lr'])


# Small test comparing dynamic algorithm results with naive Woodbury implementation
# for rank 1 computations
def _test1():
    import optim

    def dist(x, y):
        return torch.mean(torch.abs(x - y))

    K = 100
    L = 10
    M = 32 
    DEV = torch.device('cuda:0')


    grads_left = torch.randn((M, K), device=DEV, dtype=torch.float64)
    grads_right = torch.randn((M, L), device=DEV, dtype=torch.float64)
    grads_full = torch.empty((M, K * L), device=DEV, dtype=torch.float64)

    for i in range(M):
        grads_full[i, :] = torch.outer(grads_left[i], grads_right[i]).ravel()

    g_left = torch.randn(K, device=DEV, dtype=torch.float64)
    g_right = torch.randn(L, device=DEV, dtype=torch.float64)
    g = torch.outer(g_left, g_right).ravel()
    hinv1 = optim.HInvFastUpMulti(torch.zeros((M, K * L), dtype=torch.float64), DEV, [DEV])
    hinv2 = optim.HInvSlow(grads_full.clone())
    hinv3 = RankOneHInvFastUpMulti(torch.zeros_like(grads_left).to('cpu'),
        torch.zeros_like(grads_right).to('cpu'), DEV, [DEV])
    hinv4 = LowRankHInvFastUpMulti(torch.zeros_like(grads_left).unsqueeze(-1).to('cpu'),
        torch.zeros_like(grads_right).unsqueeze(-1).to('cpu'), DEV, [DEV])

    for i in range(M):
        hinv1.update(grads_full[i, :])
        hinv3.update(grads_left[i, :], grads_right[i, :])
        hinv4.update(grads_left[i, :].unsqueeze(-1), grads_right[i, :].unsqueeze(-1))

    mul1 = hinv1.mul(g)
    mul2 = hinv2.mul(g)
    mul3 = hinv3.mul(g_left, g_right).ravel()
    mul4 = hinv4.mul(g_left.unsqueeze(-1), g_right.unsqueeze(-1)).ravel()
    print('Fast to Slow:', dist(mul1, mul2))
    print('Fast to Rank 1:', dist(mul1, mul3)) 
    print('Slow to Rank 1:', dist(mul2, mul3))
    print('Rank 1 to Low Rank (1):', dist(mul3, mul4))

# Small test comparing low rank computation with full rank computation. 
def _test2():
    import optim

    def dist(x, y):
        return torch.mean(torch.abs(x - y))

    K = 100
    L = 10
    RANK = 5
    M = 32 
    DEV = torch.device('cuda:0')


    grads_left = torch.randn((M, K, RANK), device=DEV, dtype=torch.float64)
    grads_right = torch.randn((M, L, RANK), device=DEV, dtype=torch.float64)
    grads_full = torch.empty((M, K * L), device=DEV, dtype=torch.float64)

    for i in range(M):
        grads_full[i, :] = (grads_left[i] @ grads_right[i].t()).ravel()

    g_left = torch.randn(K, RANK, device=DEV, dtype=torch.float64)
    g_right = torch.randn(L, RANK, device=DEV, dtype=torch.float64)
    g = (g_left @ g_right.t()).ravel()
    hinv1 = optim.HInvFastUpMulti(torch.zeros((M, K * L), dtype=torch.float64), DEV, [DEV])
    hinv2 = LowRankHInvFastUpMulti(torch.zeros_like(grads_left).to('cpu'),
        torch.zeros_like(grads_right).to('cpu'), DEV, [DEV])

    for i in range(M):
        hinv1.update(grads_full[i, :])
        hinv2.update(grads_left[i, :], grads_right[i, :])

    mul1 = hinv1.mul(g)
    mul2 = hinv2.mul(g_left, g_right).ravel()
    print(f'Fast to Low Rank {RANK}:', dist(mul1, mul2))


def _test3():
    import optim

    def dist(x, y):
        return torch.mean(torch.abs(x - y))

    KS = [100, 50]
    LS = [10, 15]
    M = 32 
    DEV = torch.device('cuda:0')
    DAMP = 1.0 * 10**(-5) 


    num_layers = len(KS)
    total_num_params = sum([KS[i] * LS[i] for i in range(num_layers)])


    grads_left = []
    grads_right = []
    grads_full = []

    for i in range(num_layers):
        grad_left = torch.randn((M, KS[i]), device=DEV, dtype=torch.float64)
        grad_right = torch.randn((M, LS[i]), device=DEV, dtype=torch.float64)
        grad_full = torch.empty((M, KS[i] * LS[i]), device=DEV, dtype=torch.float64)
        for i in range(M):
            grad_full[i, :] = torch.outer(grad_left[i], grad_right[i]).ravel()
        grads_left.append(grad_left)
        grads_right.append(grad_right)
        grads_full.append(grad_full)
    
    gs_left = []
    gs_right = []
    gs = []

    for i in range(num_layers):
        gs_left.append(torch.randn(KS[i], device=DEV, dtype=torch.float64))
        gs_right.append(torch.randn(LS[i], device=DEV, dtype=torch.float64))
        gs.append(torch.outer(gs_left[-1], gs_right[-1]).ravel())


    hinv1 = optim.HInvFastUpMulti(torch.zeros((M, total_num_params), dtype=torch.float64), DEV, [DEV], damp=DAMP)
    hinv2 = [
        RankOneHInvFastUpMulti(torch.zeros_like(grads_left[i]).to('cpu'),
        torch.zeros_like(grads_right[i]).to('cpu'), DEV, [DEV], damp=DAMP) 
        for i in range(num_layers)
    ]
    hcoef2 = RankOneHInvFastScalarProds(torch.zeros((M, M), device=DEV, dtype=torch.float64), damp=DAMP)


    for i in range(M):
        # Update MFAC
        updates = []
        for layer_id in range(num_layers):
            updates.append(grads_full[layer_id][i, :])
        full_update = torch.cat(updates)
        hinv1.update(full_update)

        update = torch.zeros(M, device=grads_left[0].device)
        for layer_id in range(num_layers):
            update += hinv2[layer_id].update(grads_left[layer_id][i, :], grads_right[layer_id][i, :])
        hcoef2.update(update)

    mul1 = hinv1.mul(torch.cat(gs))

    dots = torch.zeros(M, device=grads_left[0].device)
    for layer_id in range(num_layers):
        dots += hinv2[layer_id].grads_matmul(gs_left[layer_id], gs_right[layer_id])
    coefs = hcoef2.get_coefs(dots)

    muls2 = [] 
    for layer_id in range(num_layers):
        muls2.append(
            hinv2[layer_id].lambd *
            torch.outer(gs_left[layer_id], gs_right[layer_id]) -
            hinv2[layer_id].grads_linear_sum(coefs)
        )
    mul2 = torch.cat([x.ravel() for x in muls2])
    print('Full rank to Rank 1, multiple layers:')
    print(f'\tdampening coef = {DAMP}')
    print('\tabsolute error:', dist(mul1, mul2))
    print('\trelative error:', dist(mul1, mul2) / torch.norm(mul1))


if __name__ == '__main__':
    print('Running test 1...')
    _test1()
    print('Running test 2...')
    _test2()
    print('Running test 3...')
    _test3()