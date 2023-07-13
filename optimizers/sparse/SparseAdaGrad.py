import torch
import wandb
from helpers.optim import update_model, quantify_preconditioning, get_different_params_norm, get_k, apply_topk, get_gradients, get_weights
from helpers.tools import get_sparsity


class SparseAdaGrad(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay, wd_type, k_init, eps=1e-10):
        super(SparseAdaGrad, self).__init__(params, dict(lr=lr, weight_decay=weight_decay))
        self.lr = lr
        self.weight_decay = weight_decay
        self.wd_type = wd_type
        self.eps = eps
        self.k_init = k_init
        self.k = None
        self.steps = 0
        self.named_parameters = None
        self.wandb_data = dict()

        self.G = None # diagonal hessian
        self.error = None # error buffer (for feedback)
        self.dev = None

    def set_named_parameters(self, named_parameters):
        self.named_parameters = named_parameters

    def setup(self, g):
        if self.k is None:
            self.k = get_k(numel=g.numel(), k_init=self.k_init)

        if self.G is None:
            self.G = torch.zeros_like(g)

        if self.error is None:
            self.error = torch.zeros_like(g)

        if self.dev is None:
            self.dev = g.device

    @torch.no_grad()
    def step(self):
        self.steps += 1

        if self.wd_type == 'wd':
            g = get_gradients(self.param_groups)
        elif self.wd_type in ['reg', 'both']:
            w = get_weights(self.param_groups)
            g = get_gradients(self.param_groups)
            # TODO: review this if and update_model method
            if self.weight_decay > 0:
                w.mul_(self.weight_decay)
                g.add_(w)
        else:
            raise RuntimeError(f'[AdaGrad][step] invalid wd_type: {self.wd_type}')

        if self.steps == 1: # call once
            self.setup(g)

        self.error, acc, acc_topk, mask, topk_indices = apply_topk(
            lr=1,
            k=self.k,
            error=self.error,
            vector=g,
            use_ef=True,
            device=self.dev,
            layerwise_topk=False,
            layerwise_index_pairs=None)

        self.G.add_(acc_topk ** 2) # G += acc^2

        update = acc_topk / (self.G.sqrt() + self.eps)

        sp_acc_topk = get_sparsity(acc_topk)
        sp_update = get_sparsity(update)

        shrinking_factor = update_model(
            params=self.param_groups,
            update=update,
            wd_type=self.wd_type,
            alpha=None)

        self.wandb_data.update(quantify_preconditioning(g=g, u=update.to(g.device), return_distribution=False, use_abs=True))
        self.wandb_data.update(get_different_params_norm(self.named_parameters))
        self.wandb_data.update(dict(
            sparsity_acctopk=sp_acc_topk,
            sparsity_update=sp_update,
            norm_upd_w_lr=update.norm(p=2) * self.param_groups[0]['lr'],
            shrinking_factor=shrinking_factor,
            norm_g=g.norm(p=2),
            norm_error=self.error.norm(p=2),))
        wandb.log(self.wandb_data)
