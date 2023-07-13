import torch
import wandb
import math
import numpy as np

from helpers.tools import map_interval


class LoadedGPU:
    """This class represents a GPU on which multiple layers run on"""
    def __init__(self):
        self.layers = [] # holds layer indexes
        self.count = 0 # holds the total number of parameters on all layers

    def add_layer(self, index, count):
        self.layers.append(index)
        self.count += count


def balance_layers_on_gpus(index_count_pairs, gpus_count):
    index_count_dict = sorted(index_count_pairs, key=lambda d: d[1], reverse=True)
    gpus = [LoadedGPU() for _ in range(gpus_count)]
    for index, count in index_count_dict:
        gpus[0].add_layer(index, count)
        gpus = sorted(gpus, key=lambda gpu: gpu.count)
    layer_index_gpu = {}
    for index_gpu, gpu in enumerate(gpus):
        for layer in gpu.layers:
            layer_index_gpu[layer] = index_gpu
    return layer_index_gpu


@torch.no_grad()
def get_different_params_norm(named_params):
    if named_params is None:
        return dict()
    norm_bn, norm_fc, norm_conv = 0, 0, 0
    for name, param in named_params:
        norm_p = (param.norm(p=2) ** 2).item()
        if 'bn' in name:
            norm_bn += norm_p
        elif 'conv' in name:
            norm_conv += norm_p
        elif 'fc' in name:
            norm_fc += norm_p
        elif 'downsample' in name:
            if len(param.size()) == 1:
                norm_bn += norm_p
            else:
                norm_conv += norm_p

    return dict(
        norm_weights=math.sqrt(norm_bn + norm_fc + norm_conv),
        norm_bn=math.sqrt(norm_bn),
        norm_fc=math.sqrt(norm_fc),
        norm_conv=math.sqrt(norm_conv))


def get_weights_and_gradients(params, get_weights):
    """
        This method returns:
        - w: the raw weights collected from the model if get_weights=True
        - g: the gradients (without WD added)
    """
    w, g = [], []
    for group in params:
        for p in group['params']:
            if get_weights:
                w.append(p.reshape(-1))
            g.append(p.grad.reshape(-1))
    if get_weights:
        return torch.cat(w), torch.cat(g)
    return torch.cat(g)


def update_model(params, update, wd_type, weight_decay=0, alpha=None):
    """
        Applies the `update` to the model
        When alpha=None, alpha is set to lr in the group
        Returns the shrinking factor for the weights
    """
    count = 0
    for group in params:
        lr = group['lr']
        wd = group.get('weight_decay', weight_decay) # if the param groups do not have weight decay, then use the external one
        # lr_wd = lr * wd
        for p in group['params']:
            u = update[count:(count + p.numel())].reshape(p.shape).to(p.device) # move the update from its custom GPU to the device of parameter
            if (wd_type in ['wd', 'both']) and (wd > 0):
                p.mul_(1 - wd)
            p.add_(u, alpha=-lr if alpha is None else alpha)
            count += p.numel()
    return 1 - wd


def get_cos_and_angle(x, y):
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)(x, y).item()
    angle = map_interval(x=cos, a=-1, b=1, A=180, B=0)
    return cos, angle


@torch.no_grad()
def quantify_preconditioning(g, u, return_distribution=True, use_abs=True):
    """
        This method computes some metrics that describe the preconditioning
        In the case of SGD with momentum, the update u should be momentum accumulator
        :param g: the raw gradient
        :param u: the update returned by an optimization algorithm
        :param return_distribution: set to True if you want to return the entire empirical distribution
        :param use_abs: set to True if you want to use the abs for the ratio
    """
    norm_g = g.norm(p=2).item()
    norm_u = u.norm(p=2).item()
    ratio = (u / (g + 1e-8)).detach().cpu().numpy()
    if use_abs:
        ratio = np.abs(ratio)
    q = np.array([0.25, 0.5, 0.75])
    q25, q50, q75 = np.quantile(ratio, q, interpolation='midpoint')
    cos, angle = get_cos_and_angle(g, u)
    return {
        f'qp_cos': cos,
        f'qp_angle': angle,
        f'qp_q25': q25,
        f'qp_q50': q50,
        f'qp_q75': q75,
        f'qp_min': ratio.min(),
        f'qp_max': ratio.max(),
        f'qp_mean': ratio.mean(),
        f'qp_distribution': wandb.Histogram(ratio) if return_distribution else None,
        f'qp_norm_g': norm_g,
        f'qp_norm_u': norm_u,
        f'qp_norm_ratio_u_g': norm_u / norm_g
    }


@torch.no_grad()
def compute_topk(vector, k, device, layerwise_index_pairs):
    if layerwise_index_pairs is None:
        topk_indices = torch.topk(torch.abs(vector), k=k, sorted=False).indices
    else:
        topk_indices = []
        for start, end in layerwise_index_pairs:
            idx = torch.topk(torch.abs(vector[start:end]), k=int(k * (end-start)), sorted=False).indices + start
            topk_indices.append(idx)
        topk_indices = torch.cat(topk_indices)
    mask = torch.zeros_like(vector).to(device)
    mask[topk_indices] = 1.
    vector_topk = vector * mask
    return vector_topk, topk_indices, mask


@torch.no_grad()
def apply_topk(lr, k, error, vector, use_ef, device, layerwise_index_pairs):
    acc = error + lr * vector
    acc_topk, topk_indices, mask = compute_topk(vector=acc, k=k, device=device, layerwise_index_pairs=layerwise_index_pairs)
    if use_ef:
        error = acc - acc_topk
    return error, acc, acc_topk, mask, topk_indices


@torch.no_grad()
def get_model_weight_size(model):
    size = 0
    for p in model.parameters():
        size += p.numel()
    return size


@torch.no_grad()
def get_k(numel, k_init):
    k = numel
    if k_init is not None:
        if k_init <= 1:
            # select top-k% parameters (PERCENT)
            k = int(k_init * numel)
        else:
            # select top-k parameters (CONSTANT)
            k = int(k_init)
    print(f'k={k}')
    return k


@torch.no_grad()
def count_params_from_generator(gen):
    count = 0
    for p in gen:
        count += p.numel()
    return count


def aggregate(vector, size):
    dev0 = vector[0].device
    for idx in range(1, size):
        vector[0].add_(vector[idx].to(dev0))


def zerorize(vector, size):
    for idx in range(size):
        vector[idx].zero_()
