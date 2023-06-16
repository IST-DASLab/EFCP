import argparse
from helpers.tools import *


def get_arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_random_model', type=int, default=1, required=False, choices=[0, 1], help='Set this to 1 if you want to load an initial model saved on the disk.')
    parser.add_argument('--step_checkpoint', type=int, default=0, required=False, help='Steps interval to save the model parameters to.')
    parser.add_argument('--wandb_project', type=str, required=True, help='The wandb project inside "ist" owner.')
    parser.add_argument('--wandb_group', type=str, required=True,help='The wandb group in the project.')
    parser.add_argument('--wandb_name', type=str, required=True, default=None, help='The name for the experiment in wandb runs')
    parser.add_argument('--wandb_job_type', type=str, default=None, required=True, help='The wandb job type')
    parser.add_argument('--rank', type=int, default=None, required=False, help='Rank to be used in Low-Rank MFAC')
    parser.add_argument('--fix_mask_after_epoch', type=int, default=0, required=False,
                        help='Fix the mask after this many epochs of top-k (including warmup). E.g.: warmup=10 (dense training) and '
                             'fix_mask_after_epoch=12 (top-k for epochs 10, 11 and then dense training with top-k mask starting with epoch 12')
    parser.add_argument('--warmup', type=int, default=0, required=False, help='Perform full dense training for `warmup` steps, then perform top-k training.')
    parser.add_argument('--model', type=str, required=True, choices=['logreg', 'rn18', 'rn20', 'rn32', 'rn50', 'mn', 'wrn-22-2', 'wrn-40-2', 'wrn-22-4'], help='Type of model to train.')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to dataset to use for training.')
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset name',
                        choices=['rn50x16openai', 'vitb16laion400m', 'vitb16openai', 'cifar10', 'cifar100', 'imagenet', 'imagenette', 'imagewoof'])
    parser.add_argument('--optim', required=True, help='Type of optimizer to use for training.',
                        choices=['gd', 'sgd', 'ksgd',
                                 'ag', 'kag', 'skag', 'ptag',
                                 'adam', 'kadam',
                                 'mfac', 'kgmfac', 'kumfac', 'kgumfac', 'edgmfac', 'lrmfac', 'r1mfac',
                                 'ggt', 'kdggt', 'ksggt',
                                 'shmp',
                                 'lbfgs', 'flbfgs'])
    parser.add_argument('--k', type=float, default=0, required=False, help='The value of K for the top-k strategy')
    parser.add_argument('--epochs', type=int, required=True, help='The number of epochs to train the model for')
    parser.add_argument('--ngrads', type=int, required=False, help='Size of the gradient buffer to use for the M-FAC optimizer.')
    parser.add_argument('--batch_size', type=int, required=True, help='Batchsize to use for training.')
    parser.add_argument('--momentum', type=float, required=False, help='Momentum to use for the optimizer.')
    parser.add_argument('--grad_momentum', type=float, required=False, default=0, help='Momentum to use for the gradient before feeding it to the MFAC optimizer.')
    parser.add_argument('--grad_norm_recovery', type=float, required=False, default=0, choices=[0, 1], help='Boolean indicating whether gtradient norm recovery should applied')
    # parser.add_argument('--momentum_dampening', type=float, required=True, help='Momentum dampening to use for the optimizer when momentum > 0, according to SGD algorithm here: https://pytorch.org/docs/stable/generated/torch.optim.SGD.html.')
    parser.add_argument('--weight_decay', type=float, default=0, required=False, help='Weight decay to use for the optimizer.')
    parser.add_argument('--lr', type=float, required=True, help='Learning rate')
    parser.add_argument('--damp', type=float, required=False, help='Dampening value')
    parser.add_argument('--shmp_upd_freq', type=float, required=False, help='The update_freq parameter for Shampoo')
    parser.add_argument('--shmp_eps', type=float, required=False, default=1e-4, help='The epsilon parameter for Shampoo')
    parser.add_argument('--ggt_beta1', type=float, required=False, default=0.9, help='Beta1 parameter for GGT')
    parser.add_argument('--ggt_beta2', type=float, required=False, default=0.999, help='Beta2 parameter for GGT')
    parser.add_argument('--ggt_eps', type=float, required=False, default=1e-8, help='Epsilon parameter for GGT')
    parser.add_argument('--seed', type=int, required=True, help='The seed used to initialize the random number generator')
    parser.add_argument('--root_folder', type=str, required=True, help='Name of the file where the checkpoint of the most recent epoch is persisted.')
    parser.add_argument('--lr_sched', type=str, required=True, choices=['cos', 'step', 'const'], help='The learning rate scheduler used for the optimizer')
    # if args.maintain_lr_damp_ratio: optim.set_damp(lr / args.lr_damp_ratio)
    parser.add_argument('--maintain_lr_damp_ratio', type=int, required=False, default=0, choices=[0, 1], help='Maintain the ratio between learning rate and dampenng that has to be maintained during learning rate scheduling')
    parser.add_argument('--use_sparse_tensors', type=int, required=False, default=0, choices=[0, 1], help='Specify whether you want to use sparse tensors in MFAC or not')
    parser.add_argument('--use_sparse_cuda', type=int, required=False, default=0, choices=[0, 1], help='Specify whether you want to use CUDA kernels for sparse MFAC or not')
    parser.add_argument('--profile', action='store_true', default=False, required=False, help='Use this parameter if you want to profile the step method of the optimizer')
    parser.add_argument('--use_sq_newton', type=int, choices=[0, 1], required=False, default=0, help='Whether to use Squared Newton (precondition twice)')
    parser.add_argument('--wd_type', type=str, choices=['wd', 'reg', 'both'], required=True, default='wd', help='Regularization type')
    parser.add_argument('--ignore_checks', type=int, choices=[0, 1], required=False, default=1, help='Whether to ignore loss/accuracy checks or not')
    parser.add_argument('--clip_grad_val', type=float, required=False, default=0, help='gradient clipping value')
    return preprocess_args(parser.parse_args())


def preprocess_args(args):
    args.dev = get_first_device()
    args.gpus = get_gpus(remove_first=False)
    if args.wandb_name is None:
        args.wandb_name = f'seed={args.seed}'

    #################### LR - DAMP RATIO ####################
    if 'mfac' in args.model:
        args.lr_damp_ratio = args.lr / args.damp
    #########################################################

    args.monitor_gpu_memory = False
    args.base_lr = args.lr

    # if args.lr_sched == 'cos':
    #     print(f'LR before change: {args.lr}')
    #     args.lr = lr_sched_cos_w_warmup(base_lr=args.lr, epoch=1, epochs=args.epochs, warmup_length=5)
    #     print(f'LR after change: {args.lr}')

    return args
