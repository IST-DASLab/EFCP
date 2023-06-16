import torch
import wandb


class MyAdam(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, device='cuda:0', beta1=0.9, beta2=0.999, eps=1e-8):
        super(MyAdam, self).__init__(params, dict(lr=lr, weight_decay=weight_decay, beta1=beta1, beta2=beta2, eps=eps))

        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.device = device

        self.bias_corr1 = 1
        self.bias_corr2 = 1
        self.steps = 0

        self.m = None # torch.zeros_like(g, dtype=torch.float, device=self.device)
        self.v = None # torch.zeros_like(g, dtype=torch.float, device=self.device)

        print(self.__str__())

    def __str__(self):
        return f'\n\tlr={self.lr}' \
               f'\n\twd={self.weight_decay}' \
               f'\n\tbeta1={self.beta1}' \
               f'\n\tbeta2={self.beta2}' \
               f'\n\teps={self.eps}'

    def get_gradient(self):
        g = []
        for group in self.param_groups:
            for p in group['params']:
                if self.weight_decay > 0:
                    tmp = p.grad + self.weight_decay * p
                else:
                    tmp = p.grad
                g.append(tmp.reshape(-1))
        g = torch.cat(g).to(self.device)
        return g

    @torch.no_grad()
    def step(self, closure=None):
        self.steps += 1
        g = self.get_gradient()

        if self.m is None and self.v is None:
            self.m = torch.zeros_like(g, dtype=torch.float, device=self.device)
            self.v = torch.zeros_like(g, dtype=torch.float, device=self.device)

        norm_g = torch.norm(g, p=2)

        self.m = self.beta1 * self.m + (1 - self.beta1) * g
        g.mul_(g) # compute g**2 in place
        self.v = self.beta2 * self.v + (1 - self.beta2) * g

        self.bias_corr1 *= self.beta1
        self.bias_corr2 *= self.beta2

        m_debiased = self.m / (1 - self.bias_corr1)
        v_debiased = self.v / (1 - self.bias_corr2)
        update = m_debiased / (v_debiased.sqrt() + self.eps)

        count = 0
        for group in self.param_groups:
            for p in group['params']:
                p.add_(update[count:(count + p.numel())].reshape(p.shape), alpha=-group['lr'])
                count += p.numel()

        wandb.log(dict(
            norm_g=norm_g,
            norm_m=torch.norm(self.m, p=2),
            norm_v=torch.norm(self.v, p=2),
            norm_m_db=torch.norm(m_debiased, p=2),
            norm_v_db=torch.norm(v_debiased, p=2),
            norm_u=torch.norm(update, p=2)))
