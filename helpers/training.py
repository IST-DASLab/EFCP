from os.path import join
from torchvision.models import resnet18, resnet50
from torch.optim import Adagrad as AdaGradPyTorch

from lib.models.resnet_cifar10 import *
from lib.models.mobilenet import mobilenet as custom_mobilenet
from lib.models.logistic_regression import LogisticRegression
from lib.models.wide_resnet import *

from helpers.tools import get_gpus, get_first_device, makedirs

from optimizers.lowrank.low_rank_optim import RankOneAggregateMFAC, LowRankAggregateMFAC

from optimizers.dense.GGT import GGT
from optimizers.dense.MySGD import MySGD
from optimizers.dense.Shampoo import Shampoo
from optimizers.dense.AdaGrad import AdaGrad
from optimizers.dense.MFAC import DenseMFAC

from optimizers.sparse.SparseAdaGrad import SparseAdaGrad
from optimizers.sparse.SparseAdam import SparseAdam
from optimizers.sparse.SparseGGT_DenseFormat import SparseGGT_DenseFormat
from optimizers.sparse.SparseGGT_SparseFormatSpmm import SparseGGT_SparseFormatSpmm
from optimizers.sparse.SparseGradientMFAC import SparseGradientMFAC
from optimizers.sparse.SparseSGD import SparseSGD
from optimizers.sparse.ScaledSparseAdaGrad import ScaledSparseAdaGrad


def get_optimizer(args, param_groups):
    if args.optim in ['gd', 'sgd']: # ok
        # return lambda m: torch.optim.SGD(m.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        # return lambda m: PyTorchSGD(m.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        return MySGD(param_groups,
                     lr=1,
                     momentum=args.momentum)
    if args.optim == 'ag': # ok
        return AdaGrad(param_groups,
                       lr=args.lr,
                       weight_decay=args.weight_decay,
                       wd_type=args.wd_type)
    if args.optim == 'kag': # ok
        return SparseAdaGrad(param_groups,
                             lr=args.lr,
                             weight_decay=args.weight_decay,
                             wd_type=args.wd_type,
                             k_init=args.k)
    if args.optim == 'skag': # ok
        return ScaledSparseAdaGrad(param_groups,
                                   lr=args.lr,
                                   weight_decay=args.weight_decay,
                                   wd_type=args.wd_type,
                                   k_init=args.k)
    if args.optim == 'agpt': # ok
        return AdaGradPyTorch(param_groups,
                              lr=args.lr,
                              weight_decay=args.weight_decay)
    if args.optim == 'ksgd': # ok
        return SparseSGD(param_groups,
                         k_init=args.k,
                         lr=args.lr,
                         momentum=args.momentum,
                         weight_decay=args.weight_decay)
    if args.optim == 'adam': # ok
        return torch.optim.Adam(param_groups,
                                lr=args.lr,
                                weight_decay=args.weight_decay)
    if args.optim == 'kadam': # ok
        return SparseAdam(param_groups,
                          k_init=args.k,
                          lr=args.lr,
                          weight_decay=args.weight_decay)
    if args.optim == 'shmp': # ok
        return Shampoo(param_groups,
                       lr=args.lr,
                       momentum=args.momentum,
                       weight_decay=args.weight_decay,
                       epsilon=args.shmp_eps,
                       update_freq=args.shmp_upd_freq)
    if args.optim == 'mfac': # ok
        return DenseMFAC(
            param_groups,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            ngrads=args.ngrads,
            damp=args.damp,
            wd_type=args.wd_type,
            dev=get_first_device(),
            gpus=get_gpus(remove_first=False))
    if args.optim == 'kgmfac': # ok
        return SparseGradientMFAC(
            param_groups,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            ngrads=args.ngrads,
            k_init=args.k,
            damp=args.damp,
            wd_type=args.wd_type,
            dev=get_first_device(),
            gpus=get_gpus(remove_first=False))
    if args.optim == 'ggt': # ok
        return GGT(
            param_groups,
            lr=args.lr,
            wd=args.weight_decay,
            r=args.ngrads,
            beta1=args.ggt_beta1,
            beta2=args.ggt_beta2,
            eps=args.ggt_eps)
    if args.optim == 'kdggt': # ok
        return SparseGGT_DenseFormat(
            param_groups,
            k_init=args.k,
            lr=args.lr,
            wd=args.weight_decay,
            r=args.ngrads,
            beta1=args.ggt_beta1,
            beta2=args.ggt_beta2,
            eps=args.ggt_eps)
    if args.optim == 'ksggt': # ok
        return SparseGGT_SparseFormatSpmm(
            param_groups,
            k_init=args.k,
            lr=args.lr,
            wd=args.weight_decay,
            r=args.ngrads,
            beta1=args.ggt_beta1,
            beta2=args.ggt_beta2,
            eps=args.ggt_eps)
    if args.optim in ['lbfgs', 'flbfgs']: # ok
        return torch.optim.LBFGS(
            param_groups,
            lr=args.lr,
            max_iter=5)
    if args.optim == 'r1mfac':
        return RankOneAggregateMFAC(
            param_groups,
            lr=args.lr,
            num_grads=args.ngrads,
            damp=args.damp,
            moddev=get_first_device(),
            optdev=get_first_device(),
            momentum=args.momentum,
            weight_decay=args.weight_decay)
    if args.optim == 'lrmfac':
        return LowRankAggregateMFAC(
            param_groups,
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
