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
    def __init__(self, params, lr, momentum, weight_decay, ngrads, k_init, damp,
                 fix_scaling,
                 grad_norm_recovery,
                 grad_momentum,
                 use_ef,
                 adaptive_damp,
                 wd_type,
                 damp_rule=None,
                 damp_type=None,
                 use_bias_correction=False,
                 use_grad_for_gnr=False,
                 dev='cuda:0',
                 gpus=['cuda:0'],
                 sparse=False,
                 use_sparse_tensors=False,
                 use_sparse_cuda=False,
                 layerwise_topk=False,
                 model=None):
        if type(params).__name__ not in ['generator', 'list']:
            params = params.parameters()

        self.layerwise_index_pairs = None
        if model is not None:
            self.layerwise_index_pairs = get_layer_indices(model)

        self.use_ef = use_ef

        print(f'USING k={k_init*100}%')
        self.wd_type = wd_type
        self.layerwise_topk = layerwise_topk
        # self._step_supports_amp_closure = True
        self.cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

        self.damp_type = damp_type
        self.adaptive_damp = adaptive_damp
        self.grad_norm_recovery = grad_norm_recovery
        self.fix_scaling = fix_scaling
        self.cuda_profile = False
        self.use_sparse_tensors = use_sparse_tensors
        self.use_sparse_cuda = use_sparse_cuda
        self.lr = lr
        self.m = ngrads
        self.momentum = momentum
        self.grad_momentum = grad_momentum
        self.use_bias_correction = use_bias_correction
        self.use_grad_for_gnr = use_grad_for_gnr
        self.bias_correction = 1
        self.weight_decay = weight_decay
        self.dev = dev
        self.gpus = gpus
        self.model_size = 0
        self.damp = damp
        self.k = None
        self.k_init = k_init
        self.sparse = sparse
        self.sparsity_mask = None
        self.printed = False
        self.error = None
        self.training_state = None
        self.sparsity_gradient = None
        self.sparsity_update = None
        self.steps = 0
        self.ratio_lr_damp = self.lr / self.damp
        self.wandb_data = dict()
        self.damp_scheduler = None
        self.named_parameters = None
        if self.adaptive_damp:
            if damp_type == 'c':
                self.damp_scheduler = ContinuousDamping(damp_init=damp, total_steps=None, rule=damp_rule, direction='DOWN')
            elif damp_type == 't':
                self.damp_scheduler = TikhonovDamping(self.damp)
            elif damp_type == 'kr':
                self.damp_scheduler = KeepRatioDamping(self.damp, self.lr)
            else:
                raise RuntimeError(f'Invalid value for damp_type: {damp_type}')

        super(SparseGradientMFAC, self).__init__(params, dict(lr=self.lr))

        with torch.no_grad():
            for group in self.param_groups:
                for p in group['params']:
                    self.model_size += p.numel()

            # logic to save EMA for gradient and update for the switch
            if Config.general.save_ema_artifacts:
                self.switch_param = 0.9
                self.switch_ema_g = torch.zeros(self.model_size, dtype=torch.float, device=self.dev)
                self.switch_ema_u = torch.zeros(self.model_size, dtype=torch.float, device=self.dev)

            print(f'Model size: {self.model_size}')

            if self.momentum > 0:
                self.v = torch.zeros(self.model_size, device=self.dev)

            if self.grad_momentum > 0:
                self.grad_v = torch.zeros(self.model_size, device=self.dev)

            if len(self.gpus) == 0:
                self.gpus = [self.dev]

            ##############################
            ##### INITIALIZATIONS
            ##############################

            # self.bn_mask = get_batchnorm_mask(model).to(self.dev) # 1 on BN params, 0 otherwise
            # self.sparsity_mask = torch.ones_like(w).to(self.dev)
            # self.sparsity_mask = sparsity_mask.to(self.dev)

            self.error = torch.zeros(self.model_size).to(self.dev)
            self.k = get_k(numel=self.model_size, k_init=k_init)

            if use_sparse_tensors:
                print('USING SPARSE TENSORS')
                # method = SparseHinvCUDA if self.use_sparse_cuda else SparseHinvSequential
                nnz = self.k
                if self.layerwise_topk:
                    nnz = 0
                    for start, end in self.layerwise_index_pairs:
                        nnz += int(self.k_init * (end-start))
                self.hinv = SparseHinvSequential(
                    m=self.m,
                    d=self.model_size,
                    nnz=nnz,
                    fix_scaling=self.fix_scaling,
                    dev=self.dev,
                    gpus=gpus,
                    damp=damp)
                MyLogger.get('optimizer').log(message=self.__class__.__name__)
                MyLogger.get('optimizer').log(message=self.hinv.__class__.__name__)
            else:
                raise RuntimeError('USING DENSE TENSORS IS NOT POSSIBLE IN SPARSE-MFAC')
                # print('USING DENSE TENSORS')
                # self.hinv = HInvFastUpMulti(
                #     grads=torch.zeros((ngrads, self.model_size), dtype=torch.float),
                #     dev=self.dev,
                #     gpus=gpus,
                #     damp=damp)

        MyLogger.get('optimizer').log(message=f'\n{str(self)}{str(self.damp_scheduler)}').close()

    def set_named_parameters(self, named_parameters):
        self.named_parameters = named_parameters

    def __str__(self):
        return f'\tng={self.m}\n' \
               f'\tlr={self.lr}\n' \
               f'\tdamp={self.damp}\n' \
               f'\tweight_decay={self.weight_decay}\n' \
               f'\tmom={self.momentum}\n' \
               f'\tgnr={self.grad_norm_recovery}\n' \
               f'\tgrad_mom={self.grad_momentum}\n' \
               f'\tuse_bias_correction={self.use_bias_correction}\n' \
               f'\tuse_grad_for_gnr={self.use_grad_for_gnr}\n' \
               f'\tfs={self.fix_scaling}\n' \
               f'\tk={self.k}\n' \
               f'\tgpus={self.gpus}\n' \
               f'\tuse_sparse_tensors={self.use_sparse_tensors}\n' \
               f'\tuse_sparse_cuda={self.use_sparse_cuda}\n' \
               f'\tmodel_size={self.model_size}\n'

    def zerorize_error(self):
        self.error.zero_()

    @torch.no_grad()
    def step(self, closure=None):
        # self.wandb_data = dict()
        self.steps += 1

        # new_loss = None
        # if closure is not None:
        #     new_loss = closure()

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

            # norm_grad_dense = torch.norm(g_dense, p=2)

            ########## [2] APPLY MOMENTUM WITH BIAS CORRECTION TO THE GRADIENT
            #     if self.grad_momentum > 0:
            #         self.grad_v = self.grad_momentum * self.grad_v + (1 - self.grad_momentum) * g_dense
            #         g_dense = self.grad_v
            #         if self.use_bias_correction:
            #             self.bias_correction *= self.grad_momentum
            #             g_dense.div_(1 - self.bias_correction)
            #     norm_grad_dense_mom = torch.norm(g_dense, p=2)

            ##################################################
            ########## [3] ERROR FEEDBACK AND SPARSIFICATION
            ##################################################
            self.error, acc, acc_topk, mask, topk_indices = apply_topk(
                lr=1 if Config.kgmfac.topk_lr_on_update else self.param_groups[0]['lr'],
                k=self.k_init if self.layerwise_topk else self.k,
                error=self.error,
                vector=g_dense,
                use_ef=self.use_ef,
                device=self.dev,
                layerwise_topk=self.layerwise_topk,
                layerwise_index_pairs=self.layerwise_index_pairs)

            # norm_grad_sparse = torch.norm(acc_topk, p=2)
            # if self.steps % 100 == 0:
            #     wandb.log(dict(sparse_grad_hist=wandb.Histogram(g.detach().cpu().numpy())))

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

            if Config.general.save_ema_artifacts:
                self.switch_ema_g = self.switch_param * self.switch_ema_g + g_dense
                self.switch_ema_u = self.switch_param * self.switch_ema_u + update

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
            ########## [6] APPLY GRADIENT NORM RECOVERY (GNR)
            ##################################################
            #     norm_upd_before_gnr = torch.norm(update, p=2)
            #     if self.grad_norm_recovery:
            #         # perform GNR with the smoothed gradient norm (from step 2)
            #         chosen_norm = norm_grad_dense if self.use_grad_for_gnr else norm_grad_dense_mom
            #         update.mul_(chosen_norm / norm_upd_before_gnr)
            #     norm_upd_after_gnr = torch.norm(update, p=2)

            ##################################################
            ########## [7] UPDATE THE MODEL
            ##################################################
            lr = self.param_groups[0]['lr']
            shrinking_factor = update_model(
                params=self.param_groups,
                update=update,
                wd_type=self.wd_type,
                alpha=None)

            # if self.adaptive_damp: # and self.steps % S == 0:
            #     if isinstance(self.damp_scheduler, TikhonovDamping):
            #         if new_loss is None:
            #             raise RuntimeError(f'Please provide a valid, callable closure that returns a loss value!')
            #         new_damp = self.damp_scheduler.step(new_loss=new_loss, dot=torch.dot(acc_topk, update))
            #         self.wandb_data.update(dict(rho=self.damp_scheduler.rho))
            #     elif isinstance(self.damp_scheduler, KeepRatioDamping):
            #         new_damp = self.damp_scheduler.step(new_lr=self.param_groups[0]['lr'])
            #     elif isinstance(self.damp_scheduler, ContinuousDamping):
            #         new_damp = self.damp_scheduler.step()
            #     else:
            #         print(f'Invalid type for damp scheduler: {type(self.damp_scheduler)}')
            #
            #     self.hinv.set_damp(new_damp=new_damp)
            #     self.wandb_data.update(dict(damp=new_damp))

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

# if isinstance(self.hinv, SparseCooHinv):
#     g = g.view(-1, 1)
#     update = self.hinv.update_mul(g)
# elif isinstance(self.hinv, SparseSpmmHinv):
#     update = self.hinv.update_mul(g, topk_indices)
# return np.zeros((self.weight_count_full,), dtype=np.int64)
# if self.cuda_profile and self.steps > 2000:
#     torch.cuda.cudart().cudaProfilerStop()
#     print('ENDED PROGRAM AFTER 2000 STEPS')
#     sys.exit(666)
# if self.cuda_profile and self.steps == self.m:
#     torch.cuda.cudart().cudaProfilerStart()
#     print('********** STARTED CUDA PROFILER **********')
# if self.cuda_profile and self.steps >= self.m:
#     torch.cuda.nvtx.range_push(f'[step]it@{self.steps}')
#     torch.cuda.nvtx.range_push(f'[step]update_mul@{self.steps}')
# if self.cuda_profile and self.steps >= self.m:
#     torch.cuda.nvtx.range_pop() # pop step
#     torch.cuda.nvtx.range_pop() # pop update

# cos_sim_gSparse_u=self.cos(acc_topk, update),
# cos_sim_gDense_u=self.cos(g_dense, update),
# norm_error=self.error.norm(p=2)))
# norm_grad_dense=norm_grad_dense,
# norm_grad_dense_mom=norm_grad_dense_mom,
# norm_grad_sparse=norm_grad_sparse,
# norm_upd_before_gnr=norm_upd_before_gnr,
# norm_upd_after_gnr=norm_upd_after_gnr,
# norm_upd_w_lr=self.param_groups[0]['lr'] * norm_upd_after_gnr))