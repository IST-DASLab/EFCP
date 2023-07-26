import math


class BaseBlock:
    def __init__(self, start, end, beta2, eps, main_device, compute_device):
        self.start = start
        self.end = end
        self.beta2 = beta2
        self.eps = eps
        self.main_device = main_device
        self.compute_device = compute_device

        self.steps = 0
        self.size = end - start
        self.V = None

    def _unbiasing_term(self):
        return math.sqrt(1 - self.beta2 ** self.steps)

    def update_then_precondition(self, x):
        self.steps += 1
        self._update(x)
        p = self._precondition(x)
        return p.to(self.main_device)

    def _update(self, x):
        pass

    def _precondition(self, x):
        pass
