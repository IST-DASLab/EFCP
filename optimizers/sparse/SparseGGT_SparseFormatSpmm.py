import torch
import wandb
import warnings
warnings.filterwarnings("ignore")
from helpers.optim import get_k, apply_topk
from helpers.tools import get_first_device, EMA, SMA
from torch_sparse import spmm


class SparseGGT_SparseFormatSpmm(torch.optim.Optimizer):
    """
        Implements sparse GGT with gradients in sparse format and uses spmm to perform matrix multiplication
    """
    def __init__(self, params, k_init, lr, wd, r, beta1, beta2, eps):
        self.lr = lr
        self.wd = wd # weight_decay
        self.r = r
        self.beta1 = beta1 # for the gradient
        self.beta2 = beta2 # for the gradients in the buffer G
        self.eps = eps
        self.k_init = k_init
        super(SparseGGT_SparseFormatSpmm, self).__init__(params, dict(lr=self.lr,
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
        self.full = False # becomes True when the buffer is full
        self.k = get_k(numel=self.d, k_init=self.k_init)
        self.nnz = self.k * self.r
        self.pointers = list(range(0, self.nnz, self.k)) # used as pointers for coumns in CSC format
        self.error = torch.zeros(self.d, device=self.dev)
        self.values = torch.zeros(self.nnz, dtype=torch.float, device=self.dev)
        self.rows = torch.zeros(self.nnz, dtype=torch.long, device=self.dev)
        self.cols = torch.arange(self.r).view(self.r, -1).expand(self.r, self.k).ravel().to(device=self.dev, dtype=torch.long)
        self.GTG = torch.zeros(self.r, self.r, dtype=torch.float, device='cpu') # matrix G^T * G
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
        # idxView = self.indices.view(self.r, self.k)
        # valView = self.values.view(self.r, self.k)
        # gtake = g.take(idxView)
        # dot = (gtake * valView).sum(axis=1) # good!
        dot = spmm(index=[self.cols, self.rows], value=self.values, m=self.r, n=self.d, matrix=g.unsqueeze(1)).squeeze().cpu()
        # print(f'dot.size() = {dot.size()}')
        self.GTG[:, self.index].zero_().add_(dot)
        self.GTG[self.index, :].zero_().add_(dot)

        # print(f'index = {self.index}')
        # print(f'idxView size = {idxView.size()}')
        # print(f'valView size = {valView.size()}')
        # print(f'gtake size = {gtake.size()}')
        # print('idxView')
        # print(idxView[:self.index+1, :])
        # print('valView')
        # print(valView[:self.index+1, :])
        # print('gtake')
        # print(gtake[:self.index+1, :])
        # print(f'dot = {dot}')

    def _save_gradient(self, g, topk_indices):
        """
            Saves the new gradient to the buffer in sparse format (values to G_vals and indices to G_idxs)
            :param g: sparse tensor that contains zeros (dimension d, having d-k zeros)
            :param topk_indices: indices of non-zero elements
        """
        start = self.index * self.k
        end = start + self.k
        self.values[start:end] = g[topk_indices].to(self.dev)
        self.rows[start:end] = topk_indices.to(self.dev)
        # self.cols: already set in constructor

    def _update_buffer(self, g, topk_indices):
        """
            Saves the gradient and then computes+updates the dot products
            :param g: sparse tensor that contains zeros (dimension d, having d-k zeros)
            :param topk_indices: indices of non-zero elements
        """
        self._save_gradient(g, topk_indices)
        self._update_scalar_products(g)
        self.index = (1 + self.index) % self.r
        if self.index == 0:
            self.full = True

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
        self._update_buffer(g, topk_indices)

        # print('GTG')
        # print(self.GTG + 1e-6 * self.id_r)
        Sr_squared, V = torch.linalg.eigh(self.GTG + 1e-6 * torch.eye(self.r, dtype=torch.float, device='cpu'))
        Sr_squared = Sr_squared.to(self.dev)
        V = V.to(self.dev)

        Sr = Sr_squared.sqrt()  # has dimension r (uni-dimensional)
        T = V @ torch.diag(1. / Sr)
        Ur = spmm(index=[self.rows, self.cols], value=self.values, m=self.d, n=self.r, matrix=T)
        diag = torch.linalg.inv(torch.diag(Sr + self.eps)) - self.id_r / self.eps
        # print(f'Sr sq = {Sr_squared}')
        # print(f'Sr = {Sr}')
        # print(f'V.size() = {V.size()}, #NaNs = {torch.isnan(V).sum()}')
        # print('V')
        # print(V)
        # print(f'T.size() = {T.size()}, #NaNs = {torch.isnan(T).sum()}')
        # print('T')
        # print(T)
        # print('diag')
        # print(diag)
        # print(f'Ur.size() = {Ur.size()}, #NaNs = {torch.isnan(Ur).sum()}')
        # print(f'G.size() = {G.size()}, #NaNs = {torch.isnan(G).sum()}')

        update = Ur @ diag @ (Ur.T @ g) + (g / self.eps)  # Ur * diag: multiplies column #k from Ur with diag[k]
        # print(f'#NaNs in update: {torch.isnan(update).sum() / self.d * 100}%')
        # print(f'update_min = {update.min()}, update_max = {update.max()}')
        self._update_step(update, alpha=-self.param_groups[0]['lr'])

        sim_dense_update = self.cos(g_dense_clone, update)
        sim_sparse_update = self.cos(g, update)
        self.wandb_data['Sr_evs'] = wandb.Histogram(Sr.detach().cpu().numpy()) # logged once per epoch
        wandb.log(dict(norm_error=self.error.norm(p=2),
                       sparsity_g=(g == 0).sum() / self.d,
                       # sparsity_G=(self. == 0).sum(),
                       cos_sim_gDense_u=sim_dense_update,
                       cos_sim_gDense_u_ema=self.ema_dense.add_get(sim_dense_update),
                       cos_sim_gDense_u_sma=self.sma_dense.add_get(sim_dense_update),
                       cos_sim_gSparse_u=sim_sparse_update,
                       cos_sim_gSparse_u_ema=self.ema_sparse.add_get(sim_sparse_update),
                       cos_sim_gSparse_u_sma=self.sma_sparse.add_get(sim_sparse_update))) # logged at each step
