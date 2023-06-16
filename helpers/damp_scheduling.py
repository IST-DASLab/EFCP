import numpy as np


class TikhonovDamping:
    """Implements Tikhonov damping based on Levenberg-Marquardt heuristic"""
    def __init__(self, damp_init):
        self.damp = damp_init
        self.old_loss = None
        self.coef_dec = 2 / 3
        self.coef_inc = 3 / 2
        self.rho = None

    def step(self, new_loss, dot):
        if self.old_loss is None:
            self.old_loss = new_loss
        else:
            rho = 2 * (new_loss - self.old_loss) / dot
            if rho > 0.75:
                self.damp *= self.coef_dec
            if rho < 0.25:
                self.damp *= self.coef_inc
            self.rho = rho
            self.old_loss = new_loss
        return self.damp


class ContinuousDamping:
    def __init__(self, damp_init, total_steps, rule='L', direction='DOWN'):
        direction = direction.upper()
        rule = rule.upper()

        assert len(rule) > 0 and rule[0] in 'LQCES', 'First character in the rule should be L, Q, C, E or S'
        assert direction in ['UP', 'DOWN'], 'Direction should be either UP or DOWN'

        if rule[0] in 'ES':
            assert len(rule) > 1, 'Rules (E)xponential and (S)igmoid require a parameter'

        self.direction = direction
        self.rule = rule

        self.damp_start = damp_init
        self.total_steps = total_steps

        self.k = None
        if len(rule) > 1:
            self.rule = rule[0]
            self.k = abs(float(rule[1:]))

        print(f'Using damp rule {self.rule}({self.k}) in direction {self.direction}')

        self.funcs_up = dict(
            L=lambda x: x, # Linear
            Q=lambda x: x ** 2, # Quadratic
            C=lambda x: 1 - np.sqrt(1 - x ** 2), # Circular
            E=lambda x: np.exp(self.k * (x - 1)), # Exponential
            S=lambda x: 1 / (1 + np.exp(-self.k * (x - 0.5))), # Sigmoid
        )
        self.funcs_down = dict(
            L=lambda x: 1 - x, # Linear
            Q=lambda x: 1 - x ** 2, # Quadratic
            C=lambda x: np.sqrt(1 - x ** 2), # Circular
            E=lambda x: 1 - np.exp(self.k * (x - 1)), # Exponential
            S=lambda x: 1 - 1 / (1 + np.exp(-self.k * (x - 0.5))), # Sigmoid
        )
        self.funcs = self.funcs_up if direction == 'UP' else self.funcs_down
        self.t = 0 # iteration step

    def step(self):
        self.t += 1
        u = self.t / self.total_steps
        r = self.funcs[self.rule](u)
        return self.damp_start * r

    def __str__(self):
        return f'\tdamp_start={self.damp_start}\n\ttotal_steps={self.total_steps}\n\trule={self.rule}\n\tk={self.k}\n'


class KeepRatioDamping:
    """Implement dampening scheduling that maintains lr-damp ratio from the beginning"""
    def __init__(self, damp_init, lr_init):
        self.ratio = lr_init / damp_init
        self.damp = damp_init

    def step(self, new_lr):
        if new_lr == 0:
            print(f'[WARNING] The learning rate is zero and this will cause divide by zero exception!')
        if new_lr > 0:
            self.damp = new_lr / self.ratio
        return self.damp

# class DampScheduler:
#     def __init__(self, damp_start, damp_end, warmup_steps, total_steps):
#         self.damp_init = damp_start
#         self.damp_warmup = damp_end
#         self.warmup_steps = warmup_steps
#         self.total_steps = total_steps
#         self.t = 0 # iteration step
#         self.step = self.step_exp
#
#     def step_exp(self):
#         k = 10
#         self.t += 1
#         if self.t < self.warmup_steps:
#             r = np.exp(-k * self.t / self.warmup_steps)  # goes from 0 to 1 on interval [t, warmup_steps]
#             return r * self.damp_init + (1 - r) * self.damp_warmup  # goes from damp_init to damp_warmup
#         r = (self.t - self.warmup_steps) / (self.total_steps - self.warmup_steps)  # goes from 0 to 1 on interval [warmup_steps, total_steps)
#         return (1 - r) * self.damp_warmup + r * self.damp_init
#
#     def step_lin(self):
#         self.t += 1
#         if self.t <= self.warmup_steps:
#             r = self.t / self.warmup_steps # goes from 0 to 1 on interval [t, warmup_steps]
#             return (1 - r) * self.damp_init + r * self.damp_warmup # goes from damp_init to damp_warmup
#         r = (self.t - self.warmup_steps) / (self.total_steps - self.warmup_steps) # goes from 0 to 1 on interval [warmup_steps, total_steps)
#         return (1 - r) * self.damp_warmup + r * self.damp_init
#
#     def __str__(self):
#         return f'\n\tdamp_init={self.damp_init}\n\tdamp_warmup={self.damp_warmup}\n\twarmup_steps={self.warmup_steps}\n\ttotal_steps={self.total_steps}\n'
#
#     def step_w_warmup(self):
#         self.t += 1
#         if self.t <= self.warmup_steps:
#             r = self.t / self.warmup_steps
#             return self.damp_end + r * (self.damp_start - self.damp_end)
#         u = (self.t - self.warmup_steps) / (self.total_steps - self.warmup_steps)
#         r = self.funcs[self.rule](u)
#         return self.damp_start + r * (self.damp_end - self.damp_start)
