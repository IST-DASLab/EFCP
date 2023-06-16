import torch
import wandb

from helpers.tools import get_first_device, EMA, SMA
from optimizers.config import Config


class GGT(torch.optim.Optimizer):
    def __init__(self, params, lr, wd, r, beta1, beta2, eps):
        self.lr = lr
        self.wd = wd # weight_decay
        self.r = r
        self.beta1 = beta1 # for the gradient
        self.beta2 = beta2 # for the gradients in the buffer G
        self.eps = eps
        super(GGT, self).__init__(params, dict(lr=self.lr,
                                               wd=self.wd,
                                               r=self.r,
                                               beta1=self.beta1,
                                               beta2=self.beta2,
                                               eps=self.eps))
        self.dev = get_first_device()
        self.d = sum([p.numel() for group in self.param_groups for p in group['params']])
        self.v = torch.zeros(self.d, dtype=torch.float, device=self.dev) # momentum accumulator for the gradient
        self.G = torch.zeros(self.d, self.r, dtype=torch.float, device=self.dev) # gradients are set per column
        self.id_r = torch.eye(self.r, dtype=torch.float, device=self.dev)
        self.decays = torch.zeros(self.r, dtype=torch.float, device=self.dev)
        self.index = 0 # position of next gradient in G
        self.steps = 0 # total number of steps
        print(self)
        self.wandb_data = dict()
        self.cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        self.ema = EMA(beta=0.9)
        self.sma = SMA(size=self.r)

    def __str__(self):
        return f'lr={self.lr}\nwd={self.wd}\nr={self.r}\nbeta1={self.beta1}\nbeta2={self.beta2}\neps={self.eps}\n'

    def __repr__(self):
        return str(self)

    def _get_gradient(self):
        g = []
        for group in self.param_groups:
            for p in group['params']:
                if self.wd > 0:
                    tmp = p.grad + self.wd * p
                else:
                    tmp = p.grad
                g.append(tmp.reshape(-1))
        g = torch.cat(g).to(self.dev)
        return g

    def _update_buffer(self, g):
        self.G[:, self.index].zero_().add_(g)

        self.decays.mul_(self.beta2) # decay current gradients by beta
        self.decays[self.index] = 1. # mark current index as the last gradient (no decay for it)

        self.index = (1 + self.index) % self.r

    def _update_step(self, update, alpha):
        count = 0
        for group in self.param_groups:
            for p in group['params']:
                p.add_(update[count:(count + p.numel())].reshape(p.shape), alpha=alpha)
                count += p.numel()

    @torch.no_grad()
    def step(self, closure=None):
        self.steps += 1

        g = self._get_gradient()
        g_clone = g.clone()
        self._update_buffer(g)

        if self.beta1 > 0:  # momentum for the gradient
            # self.v = self.beta1 * self.v + (1 - self.beta1) * g
            self.v.mul_(self.beta1).add_((1 - self.beta1) * g)
            g = self.v

        decayed_G = self.G * self.decays
        GTG = decayed_G.T @ decayed_G + 1e-6 * self.id_r
        Sr_squared, V = torch.linalg.eigh(GTG.cpu())
        Sr_squared = Sr_squared.to(self.dev)
        V = V.to(self.dev)
        # Sr_squared, V = Sr_squared.real, V.real
        Sr = Sr_squared.sqrt()  # has dimension r (uni-dimensional)
        Ur = self.G @ V @ torch.diag(1. / Sr)

        diag = torch.linalg.inv(torch.diag(Sr + self.eps)) - self.id_r / self.eps
        update = Ur @ diag @ (Ur.T @ g) + (g / self.eps)  # Ur * diag: multiplies column #k from Ur with diag[k]

        self._update_step(update, alpha=-self.param_groups[0]['lr'])
        sim = self.cos(g_clone, update)
        wandb.log({'step/cos_sim_g_u': sim,
                   'step/cos_sim_g_u_ema': self.ema.add_get(sim),
                   'step/cos_sim_g_u_sma': self.sma.add_get(sim)})
        self.wandb_data['step/Sr_evs'] = wandb.Histogram(Sr.detach().cpu().numpy())
