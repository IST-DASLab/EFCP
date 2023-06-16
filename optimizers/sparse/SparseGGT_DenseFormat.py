import torch
import wandb

from helpers.optim import get_k, apply_topk
from helpers.tools import get_first_device, EMA, SMA


class SparseGGT_DenseFormat(torch.optim.Optimizer):
    """
        Implements sparse GGT with gradients in dense format (contain zeros)
    """
    def __init__(self, params, k_init, lr, wd, r, beta1, beta2, eps):
        self.lr = lr
        self.wd = wd # weight_decay
        self.r = r
        self.beta1 = beta1 # for the gradient
        self.beta2 = beta2 # for the gradients in the buffer G
        self.eps = eps
        self.k_init = k_init
        super(SparseGGT_DenseFormat, self).__init__(params, dict(lr=self.lr,
                                                                 wd=self.wd,
                                                                 r=self.r,
                                                                 beta1=self.beta1,
                                                                 beta2=self.beta2,
                                                                 eps=self.eps))
        self.d = sum([p.numel() for group in self.param_groups for p in group['params']])
        self.dev = get_first_device()
        self.v = torch.zeros(self.d, dtype=torch.float, device=self.dev) # momentum accumulator for the gradient
        self.id_r = torch.eye(self.r, dtype=torch.float, device=self.dev)
        self.decays = torch.zeros(self.r, dtype=torch.float, device=self.dev)
        self.index = 0 # position of next gradient in G
        self.steps = 0 # total number of steps
        self.wandb_data = dict()
        self.cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        self.ema_sparse, self.ema_dense = EMA(beta=0.9), EMA(beta=0.9)
        self.sma_sparse, self.sma_dense = SMA(size=self.r), SMA(size=self.r)

        # particular for top-k:
        self.k = get_k(numel=self.d, k_init=self.k_init)
        self.error = torch.zeros(self.d, device=self.dev)
        self.G = torch.zeros(self.d, self.r, dtype=torch.float, device=self.dev)
        self.GTG = torch.zeros(self.r, self.r, dtype=torch.float, device=self.dev) # matrix G^T * G
        print(self)

    def __str__(self):
        return f'{self.__class__.__name__}\nk={self.k}\nlr={self.lr}\nwd={self.wd}\nr={self.r}\nbeta1={self.beta1}\nbeta2={self.beta2}\neps={self.eps}\n'

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

    def _update_step(self, update, alpha):
        count = 0
        for group in self.param_groups:
            for p in group['params']:
                p.add_(update[count:(count + p.numel())].reshape(p.shape), alpha=alpha)
                count += p.numel()

    def _update_scalar_products(self, g):
        """
            Computes scalar product between g (new gradient) and all previous gradients stored in G (as G_vals and G_idxs)
            :param g: sparse tensor that contains zeros (dimension d, having d-k zeros)
            :param topk_indices: indices of non-zero elements
        """
        dot = g @ self.G
        print(f'dot.size() = {dot.size()}, dot: {dot}')
        self.GTG[:, self.index].zero_().add_(dot)
        self.GTG[self.index, :].zero_().add_(dot)

    def _save_gradient(self, g):
        """
            Saves a new gradient to the buffer in dense format (has zeros, but store it in dense)
            :param g: sparse tensor that contains zeros (dimension d, having d-k zeros and k non-zeros)
        """
        self.G[:, self.index].zero_().add_(g)

    def _update_buffer(self, g):
        """
            Saves the gradient and then computes+updates the dot products
            :param g: sparse tensor that contains zeros (dimension d, having d-k zeros)
            :param topk_indices: indices of non-zero elements
        """
        self._save_gradient(g)
        self._update_scalar_products(g)
        self.index = (1 + self.index) % self.r

    @torch.no_grad()
    def step(self, closure=None):
        self.steps += 1
        ##################################################
        ########## [1] COLLECT GRADIENT FROM THE MODEL
        ##################################################
        g = self._get_gradient()
        g_dense_clone = g.clone()

        ##################################################
        ########## [2] ERROR FEEDBACK AND SPARSIFICATION
        ##################################################
        self.error, g, mask, topk_indices = apply_topk(
            lr=1,
            k=self.k,
            error=self.error,
            vector=g,
            device=self.dev,
            layerwise_topk=False,
            layerwise_index_pairs=None)

        self._update_buffer(g)
        print(self.GTG + 1e-6 * self.id_r)
        Sr_squared, V = torch.linalg.eigh(self.GTG + 1e-6 * self.id_r)
        Sr = Sr_squared.sqrt()  # has dimension r (uni-dimensional)
        Ur = self.G @ V @ torch.diag(1. / Sr)
        # print(f'Sr sq = {Sr_squared}')
        # print(f'Sr = {Sr}')
        # print('V')
        # print(V)
        diag = torch.linalg.inv(torch.diag(Sr + self.eps)) - self.id_r / self.eps
        update = Ur @ diag @ (Ur.T @ g) + (g / self.eps)  # Ur * diag: multiplies column #k from Ur with diag[k]

        self._update_step(update, alpha=-self.param_groups[0]['lr'])

        sim_dense_update = self.cos(g_dense_clone, update)
        sim_sparse_update = self.cos(g, update)
        self.wandb_data['Sr_evs'] = wandb.Histogram(Sr.detach().cpu().numpy()) # logged once per epoch
        wandb.log(dict(norm_error=self.error.norm(p=2),
                       sparsity_g=(g == 0).sum() / self.d,
                       sparsity_G=(self.G == 0).sum(),
                       cos_sim_gDense_u=sim_dense_update,
                       cos_sim_gDense_u_ema=self.ema_dense.add_get(sim_dense_update),
                       cos_sim_gDense_u_sma=self.sma_dense.add_get(sim_dense_update),
                       cos_sim_gSparse_u=sim_sparse_update,
                       cos_sim_gSparse_u_ema=self.ema_sparse.add_get(sim_sparse_update),
                       cos_sim_gSparse_u_sma=self.sma_sparse.add_get(sim_sparse_update))) # logged at each step
        input('Press any key to continue...')
