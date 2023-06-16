import torch
import numpy as np
from torch.optim import Optimizer
from helpers.optim import compute_topk, get_k, get_model_weight_size, apply_topk
from helpers.layer_manipulation import get_batchnorm_mask


def local_sgd(params,
        d_p_list,
        momentum_buffer_list,
        *,
        weight_decay: float,
        momentum: float,
        lr: float,
        dampening: float,
        nesterov: bool):
    update = [] # added by ionut
    for i, param in enumerate(params):
        d_p = d_p_list[i]
        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf

        # param.add_(d_p, alpha=-lr) # commented out by ionut
        update.append(d_p.reshape(-1)) # added by ionut
    update = torch.cat(update) # added by ionut
    return update


class SparseSGD(Optimizer):
    def __init__(self, model, k_init=None, lr=1e-3, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SparseSGD, self).__init__(model.parameters(), defaults)

        self.bn_mask = get_batchnorm_mask(model)
        self.error, self.k = get_k(
            numel=get_model_weight_size(model),
            k_init=k_init,
            device=model.device)
        print(f'k={self.k}')

    def __setstate__(self, state):
        super(SparseSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            lr = group['lr']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            update = local_sgd(params_with_grad,
                  d_p_list,
                  momentum_buffer_list,
                  weight_decay=weight_decay,
                  momentum=momentum,
                  lr=lr,
                  dampening=dampening,
                  nesterov=nesterov)

            ##############################
            ##### APPLY TOP-K COMPRESSION (according to Algorithm 1 in http://proceedings.mlr.press/v97/karimireddy19a/karimireddy19a.pdf)
            ##############################
            update = update * self.bn_mask
            self.error, acc_topk, mask = apply_topk(
                lr=self.param_groups[0]['lr'],
                k=self.k,
                error=self.error,
                vector=update,
                device=update.device)

            # update model
            count = 0
            for p in group['params']:
                p.add_(acc_topk[count:(count + p.numel())].reshape(p.shape), alpha=-1)
                count += p.numel()

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

            return mask.detach().cpu().numpy().astype(np.int64)
