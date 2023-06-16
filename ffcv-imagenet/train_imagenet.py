import sys
import os
sys.path.append(os.environ['EFCP_ROOT'])
import warnings; warnings.filterwarnings("ignore")
from optimizers.SWA import SWA
from optimizers.MySGD import MySGD
from optimizers.SparseGradDenseUpdateMFAC import SparseGradDenseUpdateMFAC
from optimizers.LayerWiseMFAC import LayerWiseMFAC
from helpers.tools import setup_wandb, get_first_device, get_gpus, set_all_seeds, mkdirs, get_gpu_mem_usage
from helpers.optim import get_weights_and_gradients
from helpers.layer_manipulation import get_resnet_layer_indices_and_params
from helpers.mylogger import MyLogger
from optimizers.mfac import OriginalMFAC
from optimizers.config import Config
from optimizers.PyTorchSGD import PyTorchSGD
import models_no_bn
import inspect

import torch
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
import torch.nn.functional as F
import torch.distributed as dist

torch.backends.cudnn.benchmark = True
torch.autograd.profiler.emit_nvtx(False)
torch.autograd.profiler.profile(False)

from torchvision import models
import torchmetrics
import numpy as np
from tqdm import tqdm
import wandb
import gpustat
import os
import time
import json
import socket
import psutil
import traceback
from typing import List
from types import FunctionType
from pathlib import Path
from argparse import ArgumentParser

from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import And, OneOf

from ffcv.pipeline.operation import Operation
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, RandomHorizontalFlip, ToTorchImage
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, RandomResizedCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder

OPTIM_LIST = ['sgd', 'mfac', 'kgmfac', 'lwmfac', 'lwkgmfac']
STEP_DECAY_AT_PERCENT = [0.5, 0.75] # decay at 50% and 75% of training for STEP learning rate
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224/256

Section('model', 'model details').params(
    arch=Param(And(str, OneOf(models.__dir__() + models_no_bn.__dir__())), default='resnet18'),
    pretrained=Param(int, 'is pretrained? (1/0)', default=0)
)

Section('resolution', 'resolution scheduling').params(
    min_res=Param(int, 'the minimum (starting) resolution', default=160),
    max_res=Param(int, 'the maximum (starting) resolution', default=160),
    end_ramp=Param(int, 'when to stop interpolating resolution', default=0),
    start_ramp=Param(int, 'when to start interpolating resolution', default=0)
)

Section('data', 'data related stuff').params(
    train_dataset=Param(str, '.dat file to use for training', required=True),
    val_dataset=Param(str, '.dat file to use for validation', required=True),
    num_workers=Param(int, 'The number of workers', required=True),
    in_memory=Param(int, 'does the dataset fit in memory? (1/0)', required=True)
)

Section('lr', 'lr scheduling').params(
    step_ratio=Param(float, 'learning rate step ratio', default=0.1),
    step_length=Param(int, 'learning rate step length', default=30),
    lr_schedule_type=Param(OneOf(['step', 'cyclic', 'linear', 'cos', 'const']), default='cyclic'),
    lr=Param(float, 'learning rate', default=0.5),
    lr_peak_epoch=Param(int, 'Epoch at which LR peaks', default=2),
)

Section('logging', 'how to log stuff').params(
    folder=Param(str, 'log location', required=True),
    log_level=Param(int, '0 if only at end 1 otherwise', default=100)
)

Section('validation', 'Validation parameters stuff').params(
    batch_size=Param(int, 'The batch size for validation', default=512),
    resolution=Param(int, 'final resized validation image size', default=224),
    lr_tta=Param(int, 'should do lr flipping/avging at test time', default=1)
)

Section('training', 'training hyper param stuff').params(
    eval_only=Param(int, 'eval only?', default=0),
    batch_size=Param(int, 'The batch size', default=512),
    optimizer=Param(And(str, OneOf(OPTIM_LIST)), 'The optimizer', default='sgd'),
    momentum=Param(float, 'SGD momentum', default=0.9),
    weight_decay=Param(float, 'weight decay', default=4e-5),
    epochs=Param(int, 'number of epochs', default=30),
    label_smoothing=Param(float, 'label smoothing parameter', default=0.1),
    distributed=Param(int, 'is distributed?', default=0),
    use_blurpool=Param(int, 'use blurpool?', default=0)
)

Section('dist', 'distributed training options').params(
    world_size=Param(int, 'number gpus', default=1),
    address=Param(str, 'address', default='localhost'),
    port=Param(str, 'port', default='12355')
)

Section('wandb', 'wandb settings').params(
    project=Param(str, 'the name for wandb project'),
    group=Param(str, 'the name for wandb group'),
    job_type=Param(str, 'the name for wandb job type'),
    name=Param(str, 'the name for wandb run name')
)

Section('custom', 'mfac params').params(
    seed=Param(int, 'seed for reproducibility', default=42),
    damp=Param(float, 'dampening for MFAC', default=1),
    ngrads=Param(int, 'number of gradients for MFAC', default=1024),
    k=Param(float, 'top-k parameter', default=0.01),
    adaptive_damp=Param(int, 'whether to use adaptive damping or not', default=0),
    topk_lr_on_update=Param(int, 'whether to use lr on model update', default=1),
    rescaling_kfac64=Param(int, 'whether to apply rescaling from K-FAC paper - section 6.4', default=0),
    zerorize_error=Param(int, 'whether to zerorize error feedback accumulator', default=0),
    switch_to_sgd=Param(float, 'float indicating the percentage of training when to switch from (KG)MFAC to SGD', default=0),
    rt=Param(int, 'whether to resume training or not. If set to 1, then the next rt params will be used', default=0),
    rt_path=Param(str, 'path to a switch_artefacts/MFAC-or-KGMFAC folder', default='', required=False),
    rt_epoch=Param(int, 'zero based index for the training epoch used (will look into artefacts-epoch)', default=0, required=False),
    rt_type=Param(str, 'u to use EMA-u or g to use EMA-g', default='', required=False),
    scale_prec_grad=Param(int, 'whether to scale the preconditioned gradient to have the scale of gradient', default=0),
    use_bn_model=Param(int, 'whether to use a model with bn or not', default=1),
    wd_type=Param(OneOf(['wd', 'reg', 'both']), 'wd schema: wd, reg or both', default='wd'),
    ignore_checks=Param(int, 'whether to ignore the accuracy and loss checks', default=1),
    swa_start_epoch=Param(int, 'training percentage when to start SWA', default=0),
    precondition_sparse_grad=Param(int, 'whether to precondition the sparse gradient or not', default=1),
    use_sq_newton=Param(int, 'whether to use squared newton or not (double preconditioning)', default=0),
    use_ef=Param(int, 'whether to use error feedback or not', default=1),
)


def extract_wrapped(decorated):
    closure = (c.cell_contents for c in decorated.__closure__)
    return next((c for c in closure if isinstance(c, FunctionType)), None)


@param('lr.lr')
@param('lr.step_ratio')
@param('training.epochs')
def get_step_lr(epoch, lr, step_ratio, epochs):
    if epoch >= epochs:
        return 0
    decay_epochs = [int(epochs * percent) for percent in STEP_DECAY_AT_PERCENT]
    count = sum([int(epoch >= decay_epoch) for decay_epoch in decay_epochs]) # count how many decay_at_epochs the current epoch is larger than
    return (step_ratio ** count) * lr


@param('lr.lr')
@param('training.epochs')
@param('lr.lr_peak_epoch')
def get_cyclic_lr(epoch, lr, epochs, lr_peak_epoch):
    xs = [0, lr_peak_epoch, epochs]
    ys = [1e-4 * lr, lr, 0]
    return np.interp([epoch], xs, ys)[0]


@param('lr.lr')
@param('training.epochs')
def get_cos_lr(lr, epochs, epoch, steps_per_epoch, step):
    global_total_steps = epochs * steps_per_epoch
    global_current_step = epoch * steps_per_epoch + step
    progress = global_current_step / global_total_steps
    cos_lr = 0.5 * (1 + np.cos(np.pi * progress)) * lr
    return cos_lr


@param('lr.lr')
def get_const_lr(epoch, lr):
    return lr


@param('lr.lr')
@param('training.epochs')
def get_linear_lr(epoch, lr, epochs):
    return (1 - epoch / epochs) * lr


class BlurPoolConv2d(torch.nn.Module):
    def __init__(self, conv):
        super().__init__()
        default_filter = torch.tensor([[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]]) / 16.0
        filt = default_filter.repeat(conv.in_channels, 1, 1, 1)
        self.conv = conv
        self.register_buffer('blur_filter', filt)

    def forward(self, x):
        blurred = F.conv2d(x, self.blur_filter, stride=1, padding=(1, 1),
                           groups=self.conv.in_channels, bias=None)
        return self.conv.forward(blurred)


class ImageNetTrainer:
    @param('training.optimizer')
    @param('training.distributed')
    @param('logging.folder')
    @param('custom.rt')
    @param('custom.rt_path')
    @param('custom.rt_epoch')
    @param('custom.rt_type')
    @param('custom.ignore_checks')
    def __init__(self, gpu, optimizer, distributed, folder, rt, rt_path, rt_epoch, rt_type, ignore_checks):
        self.all_params = get_current_config()
        self.gpu = gpu

        wandb.log(dict(hostname=socket.gethostname()))

        mkdirs(folder)

        items = os.listdir(folder)
        if not bool(ignore_checks) and os.path.isdir(folder) and ('log' in items) and ('params.json' in items):
            print(f'Experiment already exists at: {folder}')
            wandb.log(dict(end_reason='ExperimentAlreadyExists'))
            sys.exit(666)

        self.optimizer_path = os.path.join(folder, 'optimizer.txt')
        MyLogger.setup('optimizer', self.optimizer_path)

        ## write get_param_groups code
        # with open(os.path.join(folder, 'methods_code.txt'), 'w') as w:
        #     w.write(f'{inspect.getsource(extract_wrapped(self.get_param_groups))}\n')

        if distributed:
            self.setup_distributed()

        self.optimizer = None
        self.train_loader = self.create_train_loader()
        self.val_loader = self.create_val_loader()
        self.n_samples = 1281024 # sum([x.size(0) for x, _ in self.train_loader])
        self.model, self.scaler = self.create_model_and_scaler()
        self.create_optimizer()
        self.initialize_logger()

        if bool(rt):
            if ('sgd' == optimizer) and ('mfac' in rt_path): # use MFAC checkpoint and continue with SGD
                print('Continue with SGD from MFAC checkpoint at the end of training')
                artefacts_folder = os.path.join(rt_path, f'artefacts-{rt_epoch}')
                if not os.path.isdir(artefacts_folder):
                    print('>>>>> Artefacts folder does not exist, exiting')
                    wandb.log(dict(end_reason='InexistentArtefactsFolder'))
                    sys.exit(666)
                print(f'Reading & setting model state from epoch {rt_epoch}')
                model_state = torch.load(os.path.join(artefacts_folder, f'model-epoch-{rt_epoch}.pt'))
                self.model.load_state_dict(model_state)
                print('Set model state')
                print(f'Reading & setting optimizer state from epoch {rt_epoch} of type {rt_type}')
                optim_state = torch.load(os.path.join(artefacts_folder, f'ema-{rt_type}-{rt_epoch}.pt'))
                c = 0
                for group in self.optimizer.param_groups:
                    group['momentum_buffer'] = []
                    for p in group['params']:
                        group['momentum_buffer'].append(optim_state[c:c + p.numel()])
                        c += p.numel()
                print('Set optimizer state')
            elif ('mfac' in optimizer) and ('sgd' in rt_path): # use SGD checkpoint and continue with MFAC
                if 'cyclic' in rt_path: # continue MFAC from warmup checkpoint
                    print('Continue with MFAC from SGD checkpoint after LR warmup')
                    artefacts_folder = os.path.join(rt_path, f'warmup-artefacts-{rt_epoch}')
                    if not os.path.isdir(artefacts_folder):
                        print('>>>>> Artefacts folder does not exist, exiting')
                        wandb.log(dict(end_reason='InexistentArtefactsFolder'))
                        sys.exit(666)
                    model_state = torch.load(os.path.join(artefacts_folder, f'warmup-model-epoch-{rt_epoch}.pt'))
                    self.model.load_state_dict(model_state)
                    print(f'>>>>> READ SGD CHECKPOINT FROM EPOCH {rt_epoch}')
                else: # use SGD checkpoint to continue with MFAC at the end of training (6 april 2023)
                    print('Continue with MFAC from SGD checkpoint at the end of training')
                    artefacts_folder = os.path.join(rt_path, f'artefacts-{rt_epoch}')
                    if not os.path.isdir(artefacts_folder):
                        print('>>>>> Artefacts folder does not exist, exiting')
                        wandb.log(dict(end_reason='InexistentArtefactsFolder'))
                        sys.exit(666)
                    print(f'Reading & setting model state from epoch {rt_epoch}')
                    model_state = torch.load(os.path.join(artefacts_folder, f'model-epoch-{rt_epoch}.pt'))
                    self.model.load_state_dict(model_state)
                    self._fill_mfac_gradient_buffer()

    @param('dist.address')
    @param('dist.port')
    @param('dist.world_size')
    def setup_distributed(self, address, port, world_size):
        os.environ['MASTER_ADDR'] = address
        os.environ['MASTER_PORT'] = port

        dist.init_process_group("nccl", rank=self.gpu, world_size=world_size)
        torch.cuda.set_device(self.gpu)

    def cleanup_distributed(self):
        dist.destroy_process_group()

    @param('lr.lr_schedule_type')
    def get_lr(self, epoch, lr_schedule_type):
        lr_schedules = {
            'cyclic': get_cyclic_lr,
            'step': get_step_lr,
            'linear': get_linear_lr,
            'cos': get_cos_lr,
            'const': get_const_lr,
        }

        return lr_schedules[lr_schedule_type](epoch)

    # resolution tools
    @param('resolution.min_res')
    @param('resolution.max_res')
    @param('resolution.end_ramp')
    @param('resolution.start_ramp')
    def get_resolution(self, epoch, min_res, max_res, end_ramp, start_ramp):
        assert min_res <= max_res, 'Min resolution must be lower than max resolution'

        if epoch <= start_ramp:
            return min_res

        if epoch >= end_ramp:
            return max_res

        # otherwise, linearly interpolate to the nearest multiple of 32
        interp = np.interp([epoch], [start_ramp, end_ramp], [min_res, max_res])
        final_res = int(np.round(interp[0] / 32)) * 32
        return final_res

    @param('training.weight_decay')
    @param('training.optimizer')
    def get_param_groups(self, weight_decay, optimizer):
        # if 'lw' in optimizer:
        #     return [
        #         dict(params=layer, weight_decay=weight_decay)
        #         for layer in get_resnet_layer_indices_and_params(self.model)['layers']
        #     ]
        names_bn, layers_bn = [], []
        names_non_bn, layers_non_bn = [], []

        for name, p in self.model.named_parameters():
            if 'bn' in name:
                layers_bn.append(p)
                names_bn.append(name)
            elif 'conv' in name:
                layers_non_bn.append(p)
                names_non_bn.append(name)
            elif 'fc' in name:
                layers_non_bn.append(p)
                names_non_bn.append(name)
            elif 'downsample' in name:
                if len(p.size()) == 1:
                    layers_bn.append(p)
                    names_bn.append(name)
                else:
                    layers_non_bn.append(p)
                    names_non_bn.append(name)

        MyLogger.get('optimizer').log('BN layers:', log_console=False)
        for name, p in zip(names_bn, layers_bn):
            MyLogger.get('optimizer').log(f'\t{name} {p.size()}', log_console=False)
        MyLogger.get('optimizer').log('\nNON-BN layers:')
        for name, p in zip(names_non_bn, layers_non_bn):
            MyLogger.get('optimizer').log(f'\t{name} {p.size()}', log_console=False)
        print(f'Wrote information about BN/NON-BN layers to {self.optimizer_path}')

        if len(layers_bn) == 0: # the case when the model does not BN layers at all
            return [dict(params=layers_non_bn, weight_decay=weight_decay)]
        return [dict(params=layers_bn, weight_decay=0), dict(params=layers_non_bn, weight_decay=weight_decay)]
        # return [dict(params=list(self.model.parameters()), weight_decay=weight_decay)]


    # @param('training.weight_decay')
    # @param('custom.use_bn_model')
    # def get_param_groups_old(self, weight_decay, use_bn_model):
    #     # Only do weight decay on non-batchnorm parameters
    #     all_params = list(self.model.named_parameters())
    #
    #     if not bool(use_bn_model): # return all params if the model does not have BN
    #         return all_params
    #
    #     bn_params = [v for k, v in all_params if ('bn' in k)]
    #     other_params = [v for k, v in all_params if not ('bn' in k)]
    #     param_groups = [{
    #         'params': bn_params,
    #         'weight_decay': 0.
    #     }, {
    #         'params': other_params,
    #         'weight_decay': weight_decay
    #     }]
    #     return param_groups

    @param('training.momentum')
    @param('training.optimizer')
    @param('training.weight_decay')
    @param('training.label_smoothing')
    @param('custom.damp')
    @param('custom.ngrads')
    @param('custom.k')
    @param('custom.adaptive_damp')
    @param('lr.lr')
    @param('custom.rescaling_kfac64')
    @param('custom.rt')
    @param('logging.folder')
    @param('custom.wd_type')
    @param('custom.swa_start_epoch')
    @param('custom.use_sq_newton')
    @param('custom.use_ef')
    def create_optimizer(self, momentum, optimizer, weight_decay, label_smoothing, damp, ngrads, k, adaptive_damp, lr, rescaling_kfac64, rt, folder, wd_type, swa_start_epoch, use_sq_newton, use_ef):
        assert optimizer in OPTIM_LIST, f'Optimizer {optimizer} not in {OPTIM_LIST}'

        param_groups = self.get_param_groups()

        if 'lw' in optimizer:
            for index in range(len(param_groups)):
                name = f'optimizer-{index}'
                MyLogger.setup(name, os.path.join(folder, f'{name}.txt'))

        if optimizer == 'sgd':
            # self.optimizer = MySGD(param_groups, lr=1, momentum=momentum)
            self.optimizer = torch.optim.SGD(param_groups, lr=1, momentum=momentum)
            # if bool(rt):
            #     print('***** USING LOCAL SGD BECAUSE TRAINING IS RESUMED')
            #     self.optimizer = PyTorchSGD(param_groups, lr=1, momentum=momentum)
            # else:
            #     print('***** USING TORCH SGD BECAUSE TRAINING IS FROM SCRATCH')
            #     self.optimizer = torch.optim.SGD(param_groups, lr=1, momentum=momentum)
        elif optimizer == 'mfac':
            self.optimizer = OriginalMFAC(
                param_groups,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                ngrads=ngrads,
                damp=damp,
                fix_scaling=False,
                adaptive_damp=bool(adaptive_damp),
                empty_buffer_on_decay=False,
                grad_norm_recovery=False,
                rescaling_kfac64=bool(rescaling_kfac64),
                wd_type=wd_type,
                use_sq_newton=bool(use_sq_newton),
                grad_momentum=0,
                moddev=get_first_device(),
                optdev=get_first_device(),
                gpus=get_gpus(remove_first=False),
                sparse=False)
        elif optimizer == 'lwmfac':
            self.optimizer = LayerWiseMFAC(
                param_groups,
                model=self.model,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                ngrads=ngrads,
                damp=damp,
                fix_scaling=False,
                adaptive_damp=bool(adaptive_damp),
                empty_buffer_on_decay=False,
                grad_norm_recovery=False,
                rescaling_kfac64=bool(rescaling_kfac64),
                grad_momentum=0,
                moddev=get_first_device(),
                optdev=get_first_device(),
                gpus=get_gpus(remove_first=False),
                sparse=False)
        elif optimizer == 'kgmfac':
            self.optimizer = SparseGradDenseUpdateMFAC(
                param_groups,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                ngrads=ngrads,
                k_init=k,
                damp=damp,
                wd_type=wd_type,
                fix_scaling=False,
                grad_norm_recovery=False,
                grad_momentum=0,
                use_ef=bool(use_ef),
                adaptive_damp=bool(adaptive_damp),
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
        if hasattr(self.optimizer, 'set_named_parameters'):
            self.optimizer.set_named_parameters(list(self.model.named_parameters()))
        self.loss = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        if swa_start_epoch > 0:
            MyLogger.get('optimizer').log(f'Wrapping {optimizer.upper()} into SWA')
            self.optimizer = SWA(self.optimizer)
            self.optimizer.defaults = dict()

    @param('data.train_dataset')
    @param('data.num_workers')
    @param('training.batch_size')
    @param('training.distributed')
    @param('data.in_memory')
    def create_train_loader(self, train_dataset, num_workers, batch_size,
                            distributed, in_memory):
        this_device = f'cuda:{self.gpu}'
        self.device = torch.device(this_device)
        train_path = Path(train_dataset)
        assert train_path.is_file(), f'FFCV train dataset does not exist at {train_dataset}'

        res = self.get_resolution(epoch=0)
        self.decoder = RandomResizedCropRGBImageDecoder((res, res))
        image_pipeline: List[Operation] = [
            self.decoder,
            RandomHorizontalFlip(),
            ToTensor(),
            ToDevice(torch.device(this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
        ]

        label_pipeline: List[Operation] = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(torch.device(this_device), non_blocking=True)
        ]

        order = OrderOption.RANDOM if distributed else OrderOption.QUASI_RANDOM
        loader = Loader(train_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        order=order,
                        os_cache=in_memory,
                        drop_last=True,
                        pipelines={
                            'image': image_pipeline,
                            'label': label_pipeline
                        },
                        distributed=distributed)

        return loader

    @param('data.val_dataset')
    @param('data.num_workers')
    @param('validation.batch_size')
    @param('validation.resolution')
    @param('training.distributed')
    def create_val_loader(self, val_dataset, num_workers, batch_size,
                          resolution, distributed):
        this_device = f'cuda:{self.gpu}'
        val_path = Path(val_dataset)
        assert val_path.is_file(), 'FFCV val dataset does not exist'
        res_tuple = (resolution, resolution)
        cropper = CenterCropRGBImageDecoder(res_tuple, ratio=DEFAULT_CROP_RATIO)
        image_pipeline = [
            cropper,
            ToTensor(),
            ToDevice(torch.device(this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
        ]

        label_pipeline = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(torch.device(this_device),
            non_blocking=True)
        ]

        loader = Loader(val_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        order=OrderOption.SEQUENTIAL,
                        drop_last=False,
                        pipelines={
                            'image': image_pipeline,
                            'label': label_pipeline
                        },
                        distributed=distributed)
        return loader

    @param('training.epochs')
    @param('logging.log_level')
    @param('training.optimizer')
    @param('custom.zerorize_error')
    @param('custom.switch_to_sgd')
    @param('custom.rt')
    @param('custom.rt_epoch')
    @param('custom.rt_path')
    @param('custom.rt_type')
    @param('lr.lr_peak_epoch')
    @param('lr.lr_schedule_type')
    @param('custom.swa_start_epoch')
    @param('custom.use_bn_model')
    def train(self, epochs, log_level, optimizer, zerorize_error, switch_to_sgd, rt, rt_epoch, rt_path, rt_type, lr_peak_epoch, lr_schedule_type, swa_start_epoch, use_bn_model):
        start_train = time.time()
        checkpoint_epochs = [0, lr_peak_epoch, int(0.25 * epochs), int(0.50 * epochs), int(0.66 * epochs), int(0.75 * epochs)]
        print(f'TORCH CUDA: {torch.cuda.is_available()}', file=sys.stderr)
        for epoch in range(rt * (rt_epoch + 1), epochs):
            res = self.get_resolution(epoch)
            self.decoder.output_size = (res, res)

            start_epoch = time.time()
            train_loss = self.train_loop(epoch)
            epoch_time = time.time() - start_epoch

            if (zerorize_error == 1) and ('kgmfac' == optimizer):
                self.optimizer.zerorize_error()

            # switch from (KG)MFAC to SGD at a specific percentage of training (SGD starts from scratch then)
            if switch_to_sgd > 0 and ('mfac' in optimizer) and (epoch+1 == int(epochs * switch_to_sgd)):
                del self.optimizer
                self.optimizer = torch.optim.SGD(self.get_param_groups(), lr=1, momentum=0.9)
                print(f'-----SWITCHED FROM ***{optimizer.upper()}*** TO ***SGD*** AT EPOCH {epoch} ({switch_to_sgd * 100:.2f}% OF TRAINING)')
                torch.save(self.model.state_dict(), self.log_folder / f'model-epoch-{epoch}-switch-{switch_to_sgd}.pt')

            # automatically save some checkpoints at 25%, 50% and 75% of training with (KG)MFAC, including the internal EMAs
            if Config.general.save_ema_artifacts and ('mfac' in optimizer) and (epoch + 1 in checkpoint_epochs):
                artefacts_folder = os.path.join(str(self.log_folder), f'artefacts-{epoch}')
                mkdirs(artefacts_folder)
                torch.save(self.model.state_dict(), os.path.join(artefacts_folder, f'model-epoch-{epoch}.pt'))
                torch.save(self.optimizer.switch_ema_g.detach().cpu(), os.path.join(artefacts_folder, f'ema-g-{epoch}.pt'))
                torch.save(self.optimizer.switch_ema_u.detach().cpu(), os.path.join(artefacts_folder, f'ema-u-{epoch}.pt'))

            # logic to save checkpoint after warming up
            # if ('sgd' in optimizer) and (lr_schedule_type == 'cyclic') and (epoch + 1 == lr_peak_epoch):
            #     artefacts_folder = os.path.join(str(self.log_folder), f'warmup-artefacts-{epoch}')
            #     mkdirs(artefacts_folder)
            #     torch.save(self.model.state_dict(), os.path.join(artefacts_folder, f'warmup-model-epoch-{epoch}.pt'))

            if ('sgd' == optimizer) and (epoch+1 in checkpoint_epochs):
                artefacts_folder = os.path.join(str(self.log_folder), f'artefacts-{epoch}')
                mkdirs(artefacts_folder)
                torch.save(self.model.state_dict(), os.path.join(artefacts_folder, f'model-epoch-{epoch}.pt'))

            if log_level > 0:
                self.eval_and_log(dict(
                    train_loss=train_loss,
                    epoch=epoch,
                    elapsed_epoch=epoch_time,
                    ram_mem_usage=round(psutil.Process().memory_info().rss / (2 ** 30), 2),
                    gpu_mem_usage=get_gpu_mem_usage()
                ))

        if self.gpu == 0:
            file_name = 'final_weights.pt'
            # if bool(rt):
            #     rt_from = 'kgmfac1024' if 'kgmfac' in rt_path else ('mfac' if 'mfac1024' in rt_path else None)
            #     if rt_from is None:
            #         print('>>>>> rt_path does not contain MFAC artifacts => EXIT')
            #         wandb.log(dict(end_reason='RtPathNotFromMFAC'))
            #         sys.exit(666)
            #     file_name = f'final_weights_from-{rt_from}-{rt_epoch}-{rt_type}.pt'
            torch.save(self.model.state_dict(), self.log_folder / file_name)
        train_time = time.time() - start_train
        wandb.log(dict(elapsed_train=train_time, end_reason='success'))
        MyLogger.get('optimizer').log(f'Training finished')

        if swa_start_epoch > 0:
            MyLogger.get('optimizer').log(f'Switching weights from {optimizer.upper()} to SWA running averages')
            self.optimizer.swap_swa_sgd()
            if bool(use_bn_model):
                MyLogger.get('optimizer').log(f'Updating BN running averages')
                self.optimizer.bn_update(self.train_loader, self.model, device=self.device, use_autocast=True)
            MyLogger.get('optimizer').log(f'')
            with autocast():
                stats = self.val_loop()
            MyLogger.get('optimizer').log(f'stats after SWA: {str(stats)}')
            torch.save(self.model.state_dict(), self.log_folder / "final_weights_swa.pt")
            wandb.log(dict(swa_top_1=stats['top_1'], swa_top_5=stats['top_5'], swa_val_loss=stats['loss']))


    @param('custom.ignore_checks')
    def eval_and_log(self, extra_dict, ignore_checks):
        if extra_dict is None:
            extra_dict = {}
        start_val = time.time()
        stats = self.val_loop()
        val_time = time.time() - start_val
        if self.gpu == 0:
            log_dict = {
                'current_lr': self.optimizer.param_groups[0]['lr'],
                'top_1': stats['top_1'],
                'top_5': stats['top_5'],
                'val_time': val_time,
                'val_loss': stats['loss']
            }
            log_dict.update(extra_dict)
            print(f'log_dict = {log_dict}')
            self.log(log_dict)
            wandb.log(log_dict)

            if not bool(ignore_checks):
                if stats['top_1'] < 0.1 and extra_dict['epoch'] >= 2:
                    wandb.log(dict(end_reason=f'TooLowAccuracy@Epoch{extra_dict["epoch"]}<0.1'))
                    sys.exit(666)
                thresh = np.log(1000)
                if extra_dict['train_loss'] > thresh and extra_dict['epoch'] == 0:
                    wandb.log(dict(end_reason=f'TooLargeLoss@Epoch={extra_dict["epoch"]}:{stats["loss"]:.2f}>{thresh:.2f}'))
                    sys.exit(666)

        return stats

    @param('model.arch')
    @param('model.pretrained')
    @param('training.distributed')
    @param('training.use_blurpool')
    @param('custom.use_bn_model')
    def create_model_and_scaler(self, arch, pretrained, distributed, use_blurpool, use_bn_model):
        scaler = GradScaler()
        module, arch = (models, arch) if bool(use_bn_model) else (models_no_bn, f'{arch}nobn')
        model = getattr(module, arch)(pretrained=pretrained)

        def apply_blurpool(mod: torch.nn.Module):
            for (name, child) in mod.named_children():
                if isinstance(child, torch.nn.Conv2d) and (np.max(child.stride) > 1 and child.in_channels >= 16):
                    setattr(mod, name, BlurPoolConv2d(child))
                else: apply_blurpool(child)
        if use_blurpool: apply_blurpool(model)

        model = model.to(memory_format=torch.channels_last)
        model = model.to(self.gpu)

        gpus = get_gpus(remove_first=False)
        if len(gpus) > 1:
            model = torch.nn.DataParallel(model, device_ids=gpus)

        if distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.gpu])

        return model, scaler

    @param('training.optimizer')
    def _fill_mfac_gradient_buffer(self, optimizer):
        """Fill in the gradient buffer for MFAC"""
        model = self.model
        model.train()
        if optimizer != 'mfac':
            raise RuntimeError('This method must be used with MFAC only!')

        iterator = iter(self.train_loader)
        print('Computing gradients for MFAC buffer...')
        for _ in tqdm(range(self.optimizer.m)):
            image, target = next(iterator)

            self.optimizer.zero_grad(set_to_none=True)
            with autocast():
                output = model(image)
                loss_train = self.loss(output, target)
            loss_train.backward()
            # self.scaler.scale(loss_train).backward()
            g = get_weights_and_gradients(self.optimizer.param_groups, get_weights=False)
            self.optimizer.hinv.update(g)
            # self.scaler.update()

    @param('training.epochs')
    @param('lr.lr_schedule_type')
    @param('logging.log_level')
    @param('wandb.group')
    @param('wandb.job_type')
    @param('custom.swa_start_epoch')
    def train_loop(self, epoch, epochs, log_level, lr_schedule_type, group, job_type, swa_start_epoch):
        model = self.model
        model.train()
        losses = []

        iters = len(self.train_loader)

        if lr_schedule_type != 'cos':
            lr_start, lr_end = self.get_lr(epoch), self.get_lr(epoch + 1)
            lrs = np.interp(np.arange(iters), [0, iters], [lr_start, lr_end])

        crt_loss, step_loss, total_loss = 0, 0, 0
        iterator = tqdm(self.train_loader)
        for ix, (images, target) in enumerate(iterator):
            ### Training start
            if lr_schedule_type == 'cos':
                current_lr = get_cos_lr(epoch=epoch, steps_per_epoch=iters, step=ix)
            else:
                current_lr = lrs[ix]
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr
            wandb.log(dict(lr=current_lr, step=epoch * iters + ix))

            self.optimizer.zero_grad(set_to_none=True)
            with autocast():
                output = self.model(images)
                loss_train = self.loss(output, target)
                crt_loss = loss_train.item()
                step_loss = crt_loss * images.size(0) / self.n_samples
                total_loss += step_loss

            self.scaler.scale(loss_train).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            ### Training end

            ### Logging start
            if log_level > 0:
                losses.append(loss_train.detach())

                # group_lrs = []
                # for _, group in enumerate(self.optimizer.param_groups):
                #     group_lrs.append(f'{group["lr"]:.6f}')

                names = ['ep', 'lr']
                values = [f'{epoch}/{epochs}', f'{current_lr:.8f}']
                if log_level > 1:
                    names += ['loss']
                    values += [f'{crt_loss:.6f}']

                msg = ', '.join(f'{n}={v}' for n, v in zip(names, values))
                iterator.set_description(f'{os.getpid()} {group}_{job_type}: {msg}')

            if np.isnan(crt_loss):
                print('ENCOUNTERED NaN LOSS ===> EXITING')
                wandb.log(dict(end_reason=f'FoundNanLoss@Epoch{epoch}-Step{ix}'))
                sys.exit(666)

            if 0 < swa_start_epoch <= epoch + 1:
                self.optimizer.update_swa()
                # MyLogger.get('optimizer').log(f'Updated SWA at the end of epoch {epoch} / {epochs} (0 indexed)')
            ### Logging end
        ### end for

        return total_loss

    @param('validation.lr_tta')
    def val_loop(self, lr_tta):
        model = self.model
        model.eval()

        with torch.no_grad():
            with autocast():
                for images, target in tqdm(self.val_loader):
                    output = self.model(images)
                    if lr_tta:
                        output += self.model(torch.flip(images, dims=[3]))

                    for k in ['top_1', 'top_5']:
                        self.val_meters[k](output, target)

                    loss_val = self.loss(output, target)
                    self.val_meters['loss'](loss_val)

        stats = {k: m.compute().item() for k, m in self.val_meters.items()}
        [meter.reset() for meter in self.val_meters.values()]
        return stats

    @param('logging.folder')
    def initialize_logger(self, folder):
        self.val_meters = {
            'top_1': torchmetrics.Accuracy(task='multiclass', num_classes=1000, compute_on_step=False).to(self.gpu),
            'top_5': torchmetrics.Accuracy(task='multiclass', num_classes=1000, compute_on_step=False, top_k=5).to(self.gpu),
            'loss': MeanScalarMetric(compute_on_step=False).to(self.gpu)
        }

        if self.gpu == 0:
            folder = Path(folder).absolute()
            folder.mkdir(parents=True, exist_ok=True)

            self.log_folder = folder
            self.start_time = time.time()

            print(f'=> Logging in {self.log_folder}')
            params = {
                '.'.join(k): self.all_params[k] for k in self.all_params.entries.keys()
            }

            with open(folder / 'params.json', 'w+') as handle:
                json.dump(params, handle)

    def log(self, content):
        print(f'=> Log: {content}')
        if self.gpu != 0: return
        cur_time = time.time()
        with open(self.log_folder / 'log', 'a+') as fd:
            fd.write(json.dumps({
                'timestamp': cur_time,
                'relative_time': cur_time - self.start_time,
                **content
            }) + '\n')
            fd.flush()

    @classmethod
    @param('training.distributed')
    @param('dist.world_size')
    def launch_from_args(cls, distributed, world_size):
        if distributed:
            torch.multiprocessing.spawn(cls._exec_wrapper, nprocs=world_size, join=True)
        else:
            cls.exec(0)

    @classmethod
    def _exec_wrapper(cls, *args, **kwargs):
        make_config(quiet=True)
        cls.exec(*args, **kwargs)

    @classmethod
    @param('training.distributed')
    @param('training.eval_only')
    def exec(cls, gpu, distributed, eval_only):
        trainer = cls(gpu=gpu)
        if eval_only:
            trainer.eval_and_log(None)
        else:
            trainer.train()

        if distributed:
            trainer.cleanup_distributed()


# Utils
class MeanScalarMetric(torchmetrics.Metric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_state('sum', default=torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        self.optimizer = None

    def update(self, sample: torch.Tensor):
        self.sum += sample.sum()
        self.count += sample.numel()

    def compute(self):
        return self.sum.float() / self.count


# Running
def make_config(quiet=False):
    config = get_current_config()
    parser = ArgumentParser(description='Fast imagenet training')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    if not quiet:
        config.summary()
    return config.get()


@param('wandb.project')
@param('wandb.group')
@param('wandb.job_type')
@param('wandb.name')
@param('custom.seed')
@param('custom.topk_lr_on_update')
@param('custom.switch_to_sgd')
@param('custom.ngrads')
@param('custom.rt')
@param('custom.rt_path')
@param('training.optimizer')
@param('logging.folder')
@param('custom.scale_prec_grad')
@param('custom.precondition_sparse_grad')
def main(args, project, group, job_type, name, seed, topk_lr_on_update, switch_to_sgd, ngrads, rt, rt_path, optimizer, folder, scale_prec_grad, precondition_sparse_grad):
    try:
        Config.kgmfac.precondition_sparse_grad = precondition_sparse_grad
        Config.general.scale_preconditioned_grad_to_grad_norm = bool(scale_prec_grad)
        print(f'******************** Config.general.scale_preconditioned_grad_to_grad_norm: {Config.general.scale_preconditioned_grad_to_grad_norm}')

        if 'kgmfac' in group and int(topk_lr_on_update) == 1:
            print(f'******************** Config.kgmfac.topk_lr_on_update = {Config.kgmfac.topk_lr_on_update}')
            Config.kgmfac.topk_lr_on_update = bool(topk_lr_on_update)
            group = group.replace('rn18', 'RAW_rn18')

        if switch_to_sgd > 0 and ('mfac' in group):
            group = group.replace(f'mfac{ngrads}', f'mfac{ngrads}-sgd@{switch_to_sgd}')
            args.logging.folder = args.logging.folder.replace(args.wandb.group, group)
            args.wandb.group = group
            mkdirs(args.wandb.group)

        if bool(rt):
            if 'sgd' in optimizer:
                rt_from = 'kgmfac1024' if 'kgmfac' in rt_path else ('mfac' if 'mfac1024' in rt_path else None)
                if rt_from is None:
                    print('>>>>> rt_path does not contain MFAC artifacts => EXIT')
                    wandb.log(dict(end_reason='RtPathNotFromMFAC'))
                    sys.exit(666)
                group = group.replace('sgd', f'{rt_from}-sgd')
            if 'mfac' in optimizer:
                pass
        print(args)

        setup_wandb(project=project, job_type=job_type, group=group, name=name, config=args)
        wandb.log(dict(pid=os.getpid()))
        set_all_seeds(seed)
        ImageNetTrainer.launch_from_args()
    except Exception as e:
        MyLogger.setup('error', os.path.join(folder, 'error.txt'))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        MyLogger.get('error').log(f'exc_type: {exc_type}\n', log_error=True)
        MyLogger.get('error').log(f'fname: {fname}\n', log_error=True)
        MyLogger.get('error').log(f'line: {exc_tb.tb_lineno}\n', log_error=True)
        MyLogger.get('error').log(f'e: {str(e)}\n', log_error=True)
        MyLogger.get('error').log(f'traceback: {traceback.format_exc()}\n', log_error=True)
        MyLogger.get('error').close()
        wandb.log(dict(end_reason=e.__class__.__name__))


if __name__ == "__main__":
    cfg = make_config()
    main(cfg)
