from dataclasses import dataclass, field


@dataclass
class CustomArgs:
    lr: float = field(default=1e-4, metadata={"help": "The learning rate for MFAC/SGD"})
    damp: float = field(default=1e-6, metadata={"help": "The dampening for MFAC"})
    ngrads: int = field(default=1024, metadata={"help": "Number of gradients for MFAC"})
    momentum: float = field(default=0, metadata={"help": "Momentum parameter for MFAC/SGD"})
    k: float = field(default=1, metadata={"help": "The value to be used in Top-K"})

    shmp_eps: float = field(default=1e-4, metadata={"help": "Epsilon value for Shampoo"})
    shmp_upd_freq: int = field(default=1, metadata={"help": "Update frequency for Shampoo"})

    use_sparse_tensors: int = field(default=1, metadata={"help": "Set to 1 to use sparse tensor implementation"})
    use_sparse_cuda: int = field(default=0, metadata={"help": "Set to 1 to use CUDA kernels for dot products and linear combination"})
    skip_embeddings: bool = field(default=False, metadata={"help": "Set to True if you want to exclude embeddings from the optimization set"})
    fix_scaling: int = field(default=0, metadata={"help": "Set to 1 to divide by min(#grads, m) instead of m"})
    adaptive_damp: int = field(default=0, metadata={"help": "Set to 1 to  change the dampening adaptively based on the damp_rule"})
    damp_rule: str = field(default=None, metadata={"help": "Define the rule for dampening c-circle, q-quadratic, e-exponential"})
    empty_buffer_on_decay: int = field(default=0, metadata={"help": "Set to S to empty MFAC buffers once at S steps (when learning rate decay is performed)"})
    cancel_grad_clip: int = field(default=0, metadata={"help": "Set to 1 to cancel gradient clipping"})
    grad_momentum: float = field(default=0, metadata={"help": "Momentum to use for the gradient before feeding it to the MFAC optimizer"})
    grad_norm_recovery: int = field(default=0, metadata={"help": "Boolean indicating whether gtradient norm recovery should applied"})
    # optim_args: '{\"lr\":1e-4,\"ngrads\":1024,\"damp\":1e-6,\"momentum\":0,\"weight_decay\":0}'
    # skip_layers: list = field(default=None, metadata={"help": "Specify the prefixes of layers that you want to exclude from the optimization set"})

    optim: str = field(default="", metadata={"help": "Use --optim MFAC for MFAC optimizer"})

    wandb_project: str = field(default="", metadata={"help": "the name for wandb project"})
    wandb_group: str = field(default="", metadata={"help": "the name for wandb group"})
    wandb_job_type: str = field(default="", metadata={"help": "the name for wandb job type"})
    wandb_name: str = field(default="", metadata={"help": "the name for wandb run name"})

    kd_teacher: str = field(default=None, metadata={"help": "The teacher model for KD"})
    kd_student: str = field(default=None, metadata={"help": "The student model for KD"})
    kd_alpha: float = field(default=0.5,  metadata={"help": "The alpha for KD loss"})
    kd_temp: float = field(default=4.0, metadata={"help": "The temperature for KD loss"})