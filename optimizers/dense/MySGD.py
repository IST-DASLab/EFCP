import torch
import wandb
from helpers.optim import get_weights_and_gradients, update_model, quantify_preconditioning, get_different_params_norm


class MySGD(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum):
        super(MySGD, self).__init__(params, dict(lr=lr, momentum=momentum))
        self.momentum = momentum
        self.v = None
        self.wandb_data = dict()
        self.named_parameters = None

    def set_named_parameters(self, named_parameters):
        self.named_parameters = named_parameters

    @torch.no_grad()
    def step(self, closure=None):
        g = get_weights_and_gradients(params=self.param_groups, get_weights=False)

        if self.v is None:
            self.v = torch.zeros_like(g)

        if self.momentum > 0:
            self.v.mul_(self.momentum).add_(g) # .add_(self.param_groups[0]['weight_decay'] * w)
            u = self.v
        else:
            u = g

        lr = self.param_groups[0]['lr']
        shrinking_factor = update_model(params=self.param_groups, update=u, wd_type='wd', alpha=-lr)
        self.wandb_data.update(get_different_params_norm(self.named_parameters))
        self.wandb_data.update(dict(norm_upd_w_lr=lr * u.norm(p=2), shrinking_factor=shrinking_factor))
        self.wandb_data.update(quantify_preconditioning(g=g, u=u, return_distribution=False, use_abs=True))
        wandb.log(self.wandb_data)
