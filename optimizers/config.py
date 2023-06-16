# this file contains global settings for optimizers to avoid modifying the constructor
class Config:
    class general:
        """Set this variable to True to log information about preconditioning"""
        quantify_preconditioning = True

        """Set this variable to True to save the model and EMA for g and u"""
        save_ema_artifacts = False

        """Set this variable to True if you want the preconditioned gradient to have the SGD gradient norm (which includes weight decay)"""
        scale_preconditioned_grad_to_grad_norm = False


    class kgmfac:
        """This sub-class sets some settings for SparseGradDenseUpdateMFAC"""

        """
            Currently, the data can be logged directly to wandb for HuggingFace and vanilla CV experiments
            For MosaicML, it requires a callback that takes the data from wandb_data dict and logs it to wandb
            Setting this variable to True allows an explicit call to wandb.log for HuggingFace and vanilla CV experiments
            This variable is set before calling SparseGradDenseUpdateMFAC constructor: True for HF & CV, False for MML
        """
        force_wandb_logging = None

        """
            There are two versions of top-k compression scheme:
            1) g = get_gradient(...)
               acc = err + g (NO lr!)
               c = topk(acc)
               err = acc - c
               x_new = x_old - lr * c
            2) g = get_gradient(...)
               acc = err + lr * g (lr HERE!)
               c = topk(acc)
               err = acc - c
               x_new = x_old - c
            Setting this variable to True leads to version 1, other wise to version 2
        """
        topk_lr_on_update = True

        """Set this variable to True if you want to precondition the sparse gradient, otherwise precondition the dense gradient"""
        precondition_sparse_grad = True
