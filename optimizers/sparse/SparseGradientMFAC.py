from helpers.mylogger import MyLogger
from helpers.optim import apply_topk, get_k, quantify_preconditioning, get_weights_and_gradients, update_model, get_different_params_norm
from helpers.layer_manipulation import get_batchnorm_mask, get_layer_indices
from helpers.damp_scheduling import ContinuousDamping, TikhonovDamping, KeepRatioDamping
from helpers.tools import *
from optimizers.sparse.SparseHinvSequential import SparseHinvSequential
from optimizers.config import Config


class TrainingState:
    """
    This class specifies the training states. Example:
    (dense training --- warmup for 10 epochs) then (sparse training with top-k --- for 2 epochs) then (masked dense training using the mask from top-k)
    """

    # perform dense training
    DENSE_WARMUP = 0

    # perform top-k training
    SPARSE_TOPK = 1

    # perform masked dense training after freezing the weights which were not updated during top-k
    DENSE_FIXED_MASK_FREEZE = 2

    # perform masked dense training after zerorizing the weights which were not updated during top-k
    DENSE_FIXED_MASK_ZERORIZE = 3


class SparseGradientMFAC(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum, weight_decay, ngrads, k_init, damp, wd_type, sparse=False, dev=None, gpus=None):
        if type(params).__name__ not in ['generator', 'list']:
            params = params.parameters()

        print(f'USING k={k_init*100}%')
        self.wd_type = wd_type
        self.cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

        self.cuda_profile = False
        self.lr = lr
        self.m = ngrads
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.dev = dev if dev is not None else get_first_device()
        self.gpus = gpus if gpus is not None else get_gpus(remove_first=False)
        self.damp = damp
        self.k = None
        self.k_init = k_init
        self.sparse = sparse
        self.sparse_update = None

        self.error = None
        self.sparsity_gradient = None
        self.sparsity_update = None

        self.wandb_data = dict()
        self.named_parameters = None

        self.steps = 0
        self.model_size = 0

        super(SparseGradientMFAC, self).__init__(params, dict(lr=self.lr))

        with torch.no_grad():
            w = []
            for group in self.param_groups:
                for p in group['params']:
                    w.append(p.reshape(-1))

            w = torch.cat(w).to(self.dev)
            print(f'Full Model Size: {w.numel()}')

            if self.sparse:
                self.mask = w != 0
                w = w[self.mask]
                print(f'Pruned Model Size: {w.numel(0)}')

            self.model_size = w.numel()

            if self.sparse:
                self.sparse_update = torch.zeros(self.model_size, dtype=torch.float, device=self.dev)

            if self.momentum > 0:
                self.v = torch.zeros(self.model_size, dtype=torch.float, device=self.dev)

            if len(self.gpus) == 0:
                self.gpus = [self.dev]

            ##############################
            ##### INITIALIZATIONS
            ##############################
            self.error = torch.zeros(self.model_size).to(self.dev)
            self.k = get_k(numel=self.model_size, k_init=k_init)

            print('USING SPARSE TENSORS')
            self.hinv = SparseHinvSequential(
                m=self.m,
                d=self.model_size,
                nnz=self.k,
                dev=self.dev,
                gpus=gpus,
                damp=damp)
            MyLogger.get('optimizer').log(message=self.__class__.__name__)
            MyLogger.get('optimizer').log(message=self.hinv.__class__.__name__)

        MyLogger.get('optimizer').log(message=f'\n{str(self)}').close()

    def set_named_parameters(self, named_parameters):
        self.named_parameters = named_parameters

    def __str__(self):
        return f'\tng={self.m}\n' \
               f'\tlr={self.lr}\n' \
               f'\tdamp={self.damp}\n' \
               f'\tweight_decay={self.weight_decay}\n' \
               f'\tmom={self.momentum}\n' \
               f'\tk={self.k}\n' \
               f'\tgpus={self.gpus}\n' \
               f'\tmodel_size={self.model_size}\n'

    def zerorize_error(self):
        self.error.zero_()

    @torch.no_grad()
    def step(self, closure=None):
        self.steps += 1

        with torch.no_grad():
            ##################################################
            ########## [1] COLLECT GRADIENT FROM THE MODEL
            ##################################################
            if self.wd_type == 'wd':
                g_dense = get_weights_and_gradients(self.param_groups, get_weights=False)
            elif self.wd_type in ['reg', 'both']:
                w, g_dense = get_weights_and_gradients(self.param_groups, get_weights=True)

            if torch.isnan(g_dense).sum() > 0:
                print(f'gradient has NaNs at step {self.steps}')

            ##################################################
            ########## [2] DISCARD PRUNED ENTRIES FROM GRADIENT
            ##################################################
            if self.sparse:
                g_dense = g_dense[self.mask] # keep only non-pruned weights

            ##################################################
            ########## [3] ERROR FEEDBACK AND SPARSIFICATION
            ##################################################
            self.error, acc, acc_topk, mask, topk_indices = apply_topk(
                lr=1 if Config.kgmfac.topk_lr_on_update else self.param_groups[0]['lr'],
                k=self.k,
                error=self.error,
                vector=g_dense,
                use_ef=True,
                device=self.dev,
                layerwise_index_pairs=None)

            ##################################################
            ########## [4] GET MFAC UPDATE
            ##################################################
            if not isinstance(self.hinv, SparseHinvSequential):
                raise RuntimeError('Invalid type for hinv!')

            x = acc_topk if Config.kgmfac.precondition_sparse_grad else g_dense
            if self.wd_type == 'wd':
                update = self.hinv.integrate_gradient_and_precondition(g=acc_topk, indices=topk_indices, x=x)
            elif self.wd_type in ['reg', 'both']:
                update = self.hinv.integrate_gradient_and_precondition(g=acc_topk, indices=topk_indices, x=x + self.weight_decay * w)

            ##################################################
            ########## [5] APPLY MOMENTUM TO THE UPDATE
            ##################################################
            if self.momentum > 0:
                self.v = self.momentum * self.v + update
                update = self.v
            update = update.to(self.dev)

            self.sparsity_gradient = (acc_topk == 0).sum() / self.model_size * 100.
            self.sparsity_update = (update == 0).sum() / self.model_size * 100.

            ##################################################
            ########## [6] IF SPARSE TRAINING, THEN ASSIGN THE UPDATE AT THE RIGHT INDICES IN A D-DIMENSIONAL ARRAY
            ########## If model has overall sparsity 98%, then g_dense will have only 2% entries and sparse_update
            ########## will have full size, but only the 2% values will be updated
            ##################################################
            if self.sparse:
                self.sparse_update[mask] = update
                update = self.sparse_update

            ##################################################
            ########## [7] UPDATE THE MODEL
            ##################################################
            lr = self.param_groups[0]['lr']
            shrinking_factor = update_model(
                params=self.param_groups,
                update=update,
                wd_type=self.wd_type,
                alpha=None)

            ##################################################
            ########## LOGS
            ##################################################
            self.wandb_data.update(dict(
                # new_cos_g_a=self.cos(g_dense, acc),        # rotation induced by error feedback over g
                # new_cos_g_TKa=self.cos(g_dense, acc_topk), # rotation induced by EF and TopK over g
                # new_cos_g_u=self.cos(g_dense, update),     # rotation induced by EF, TopK and preconditioning over g
                # new_cos_a_TKa=self.cos(acc, acc_topk),     # rotation induced by TopK over the accumulator
                # new_cos_a_u=self.cos(acc, update),         # rotation induced by TopK and preconditioning over accumulator
                # new_cos_TKa_u=self.cos(acc_topk, update),  # rotation induced by preconditioning over the sparse accumulator

                new_norm_g=g_dense.norm(p=2),
                new_norm_error=self.error.norm(p=2),
                new_norm_acc=acc.norm(p=2),
                new_norm_TKa=acc_topk.norm(p=2),
                new_norm_u=update.norm(p=2),
            )) # computed new_* metrics on March 9th
            self.wandb_data.update(dict(norm_upd_w_lr=lr * update.norm(p=2), shrinking_factor=shrinking_factor))
            # self.wandb_data.update(quantify_preconditioning(g=g_dense, u=update, return_distribution=False, use_abs=True))
            self.wandb_data.update(self.hinv.wandb_data)
            # self.wandb_data.update(get_different_params_norm(self.named_parameters))
            wandb.log(self.wandb_data)

            ##################################################
            ########## END STEP METHOD
            ##################################################
