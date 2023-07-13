import torch
import wandb
from helpers.optim import update_model, quantify_preconditioning, get_different_params_norm, get_gradients, get_weights


class AdaGrad(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay, wd_type, eps=1e-10):
        super(AdaGrad, self).__init__(params, dict(lr=lr, weight_decay=weight_decay))
        self.lr = lr
        self.weight_decay = weight_decay
        self.wd_type = wd_type
        self.eps = eps

        self.steps = 0
        self.buffer = None
        self.named_parameters = None
        self.wandb_data = dict()


    def set_named_parameters(self, named_parameters):
        self.named_parameters = named_parameters

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

        if self.buffer is None:
            self.buffer = torch.zeros_like(g)

        g.mul_(g) # g = g^2
        self.buffer.add_(g) # buffer += g^2

        update = g / (self.buffer.sqrt() + self.eps)

        shrinking_factor = update_model(params=self.param_groups, update=update, wd_type=self.wd_type, alpha=None)

        lr = self.param_groups[0]['lr']
        self.wandb_data.update({f'norm_upd_w_lr': lr * update.norm(p=2), f'shrinking_factor': shrinking_factor})
        self.wandb_data.update(quantify_preconditioning(g=g, u=update.to(g.device), return_distribution=False, use_abs=True))
        self.wandb_data.update(get_different_params_norm(self.named_parameters))
        wandb.log(self.wandb_data)
