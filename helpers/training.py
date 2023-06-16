from os.path import isfile

from torchvision.models import resnet18, resnet50

from lib.models.resnet_cifar10 import *
from lib.models.mobilenet import mobilenet as custom_mobilenet
from lib.models.logistic_regression import LogisticRegression
from lib.models.wide_resnet import *

from helpers.tools import *
from optimizers.GGT import GGT
from optimizers.MySGD import MySGD
from optimizers.ScaledSparseAdaGrad import ScaledSparseAdaGrad
from optimizers.Shampoo import Shampoo
from optimizers.AdaGrad import AdaGrad
from optimizers.SparseAdaGrad import SparseAdaGrad
from torch.optim import Adagrad as AdaGradPyTorch
from optimizers.SparseAdam import SparseAdam
from optimizers.SparseGGT_DenseFormat import SparseGGT_DenseFormat
from optimizers.SparseGGT_SparseFormatSpmm import SparseGGT_SparseFormatSpmm
from optimizers.sparse.SparseGradientMFAC import SparseGradientMFAC

from optimizers.sparse.SparseSGD import SparseSGD

from optimizers.lowrank.low_rank_optim import RankOneAggregateMFAC, LowRankAggregateMFAC
from optimizers.dense.MFAC import DenseMFAC

FOLDER_NUMPY = 'numpy_simple'
FOLDER_HTMLS = 'html_simple'
FILE_LAYERS = 'layer_names_simple.txt'


def get_optimizer(args, param_groups):
    if args.optim in ['gd', 'sgd']:
        # return lambda m: torch.optim.SGD(m.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        # return lambda m: PyTorchSGD(m.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        return MySGD(param_groups, lr=1, momentum=args.momentum)
    if args.optim == 'ag':
        return AdaGrad(param_groups, lr=args.lr, weight_decay=args.weight_decay, wd_type=args.wd_type)
    if args.optim == 'kag':
        return SparseAdaGrad(param_groups, lr=args.lr, weight_decay=args.weight_decay, wd_type=args.wd_type, k_init=args.k)
    if args.optim == 'skag':
        return ScaledSparseAdaGrad(param_groups, lr=args.lr, weight_decay=args.weight_decay, wd_type=args.wd_type, k_init=args.k)
    if args.optim == 'agpt':
        return AdaGradPyTorch(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    if args.optim == 'ksgd':
        return lambda m: SparseSGD(m, k_init=args.k, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if args.optim == 'adam':
        return lambda m: torch.optim.Adam(m.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.optim == 'kadam':
        return lambda m: SparseAdam(m, k_init=args.k, lr=args.lr, weight_decay=args.weight_decay)
    if args.optim == 'shmp':
        return Shampoo(param_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, epsilon=args.shmp_eps, update_freq=args.shmp_upd_freq)
    if args.optim == 'mfac':
        return DenseMFAC(
            param_groups,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            ngrads=args.ngrads,
            damp=args.damp,
            fix_scaling=False,
            adaptive_damp=False,
            empty_buffer_on_decay=False,
            grad_norm_recovery=False,
            rescaling_kfac64=False,
            wd_type=args.wd_type,
            use_sq_newton=bool(args.use_sq_newton),
            grad_momentum=0,
            moddev=get_first_device(),
            dev=get_first_device(),
            gpus=get_gpus(remove_first=False),
            sparse=False)
    if args.optim == 'kgmfac':
        return SparseGradientMFAC(
            param_groups,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            ngrads=args.ngrads,
            k_init=args.k,
            damp=args.damp,
            wd_type=args.wd_type,
            use_ef=True,
            fix_scaling=False,
            grad_norm_recovery=False,
            grad_momentum=0,
            adaptive_damp=False,
            damp_type='kr',
            # damp_rule='L', # doesn't matter when adaptive_damp=False
            # use_bias_correction=False, # default
            # use_grad_for_gnr=False, # default
            # sparse=False, # default
            # model=None, # default
            dev=get_first_device(),
            gpus=get_gpus(remove_first=False),
            use_sparse_tensors=True,
            use_sparse_cuda=True)
    if args.optim == 'ggt':
        return lambda m: GGT(
            m.parameters(),
            lr=args.lr,
            wd=args.weight_decay,
            r=args.ngrads,
            beta1=args.ggt_beta1,
            beta2=args.ggt_beta2,
            eps=args.ggt_eps)
    if args.optim == 'kdggt':
        return lambda m: SparseGGT_DenseFormat(
            m.parameters(),
            k_init=args.k,
            lr=args.lr,
            wd=args.weight_decay,
            r=args.ngrads,
            beta1=args.ggt_beta1,
            beta2=args.ggt_beta2,
            eps=args.ggt_eps)
    if args.optim == 'ksggt':
        return lambda m: SparseGGT_SparseFormatSpmm(
            m.parameters(),
            k_init=args.k,
            lr=args.lr,
            wd=args.weight_decay,
            r=args.ngrads,
            beta1=args.ggt_beta1,
            beta2=args.ggt_beta2,
            eps=args.ggt_eps)
    if args.optim in ['lbfgs', 'flbfgs']:
        return lambda m: torch.optim.LBFGS(
            m.parameters(),
            lr=args.lr,
            max_iter=5)
    if args.optim == 'r1mfac':
        return lambda m: RankOneAggregateMFAC(
            m.parameters(),
            lr=args.lr,
            num_grads=args.ngrads,
            damp=args.damp,
            moddev=get_first_device(),
            optdev=get_first_device(),
            momentum=args.momentum,
            weight_decay=args.weight_decay)
    if args.optim == 'lrmfac':
        return lambda m: LowRankAggregateMFAC(
            m.parameters(),
            rank=args.rank,
            lr=args.lr,
            num_grads=args.ngrads,
            damp=args.damp,
            moddev=get_first_device(),
            optdev=get_first_device(),
            momentum=args.momentum,
            weight_decay=args.weight_decay)
    raise RuntimeError(f'Optimizer {args.optim} is not implemented!')


def get_model(model_name, dataset_name):
    num_classes = dict(imagenette=10,
                       imagewoof=10,
                       cifar10=10,
                       cifar100=100,
                       imagenet=1000,
                       rn50x16openai=1000,
                       vitb16laion400m=1000,
                       vitb16openai=1000)[dataset_name]
    if model_name == 'rn18':
        return resnet18(pretrained=False, num_classes=num_classes)
    if model_name == 'rn20':
        return resnet20(num_classes=num_classes)
    if model_name == 'rn32':
        return resnet32(num_classes=num_classes)
    if model_name == 'mn':
        return custom_mobilenet()
    if model_name == 'rn50':
        return resnet50(pretrained=False, num_classes=num_classes)
    if model_name == 'wrn-22-2':
        return Wide_ResNet(22, 2, 0, num_classes=num_classes)
    if model_name == 'wrn-22-4':
        return Wide_ResNet(22, 4, 0, num_classes=num_classes)
    if model_name == 'wrn-40-2':
        return Wide_ResNet(40, 2, 0, num_classes=num_classes)
    if model_name == 'logreg':
        if dataset_name == 'rn50x16openai':
            return LogisticRegression(input_size=768, output_size=num_classes)
        if dataset_name == 'vitb16laion400m':
            return LogisticRegression(input_size=512, output_size=num_classes)
        if dataset_name == 'vitb16openai':
            return LogisticRegression(input_size=512, output_size=num_classes)
        raise RuntimeError(f'Dataset {dataset_name} is invalid for logistic regression!')

    raise RuntimeError(f'Model {model_name} is not implemented!')


def save_model_params_and_update_at_checkpoint(update, model, root_folder, step):
    folder = join(root_folder, 'models')
    makedirs(folder, exist_ok=True)
    params = torch.cat([p.reshape(-1) for p in model.parameters()]).detach().cpu().numpy()
    np.save(file=join(folder, f'model-{step:05d}.pt'), arr=params)
    np.save(file=join(folder, f'update-{step:05d}.pt'), arr=update)


def save_to_file(file_name, content):
    if not isfile(file_name):
        with open(file_name, 'w') as handle:
            for name in content:
                handle.write(f'{name}\n')


def get_shape(param):
    params_count = param.numel()
    dim1 = param.size()[0]
    dim2 = int(params_count / dim1)  # compress all other dimensions into a single one
    return params_count, dim1, dim2
