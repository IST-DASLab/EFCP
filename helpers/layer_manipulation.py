import torch


def get_resnet_layer_indices_and_params(model):
    """
        This method is used with ResNet-18/50 models to generate pairs (start, end) of indices
    where a layer starts and ends. For example, for ResNet-18, we have that first three layers
    called conv1.weight (9408 params), bn1.weight (64 params), bn1.bias (64 params) have the
    pair of indices (0, 9536). Note that the conv and BN layers are considered a single layer
        This method returns such pairs of indices to be used in top-k strategy per layer.
        Valid layers:
    - conv weight + bn weight + bn bias
    - downsample 0 weight + downsample 1 weight + downsample 1 bias (all downsamples)
    - fc weight + fc bias
    """
    i, n, count = 0, 0, 0
    params, layer_names, names, sizes, indices, limits, layers = [], [], [], [], [], [], []
    named_params = list(model.named_parameters())

    for name, p in named_params:
        n += 1
        pair = (count, count + p.numel())
        params.append(p)
        names.append(name)
        limits.append(pair)
        count += p.numel()

    while i < n-3:
        layer_names.append([names[i], names[i+1], names[i+2]])
        layers.append([params[i], params[i+1], params[i+2]])
        first = limits[i]
        third = limits[i+2]
        start = first[0]
        end = third[1]
        indices.append((start, end))
        sizes.append(end - start)
        i += 3
    layer_names.append([names[i], names[i + 1]])
    layers.append([params[i], params[i + 1]])
    first = limits[i]
    second = limits[i+1]
    start = first[0]
    end = second[1]
    indices.append((start, end))
    sizes.append(end - start)
    return dict(indices=indices, layers=layers, sizes=sizes, names=layer_names)


@torch.no_grad()
def get_sparsity_mask(module_states, model):
    """
    The module_states dict should contain the filtered modules (present in yaml field called `load_modules`)
    The model is used to get the masks for fully-connected layers (they have a size related to down-stream dataset)
    :param module_states: a filtered dictionary with keys containing module names loaded and values containing associated tensors
    :param model: the model used to create the masks for fully-connected layers
    """
    mask = []
    # add the masks for the convolutional backbone
    # s = 'pruning mask:'
    for i, (name, param) in enumerate(module_states.items()):
        # has_fc = 'fc' in name
        has_weight_mask = name.endswith('_weight_mask')
        has_bn = 'bn' in name
        has_layer_weight = '_layer.weight' in name
        has_bias = 'bias' in name
        has_conv = 'conv' in name
        has_dot_weight = '.weight' in name
        has_downsample = 'downsample' in name

        text = f'#{i:4d}\t{param.numel():10d}\t{name}'
        if has_conv and has_weight_mask:
            # layer1.0.conv1._weight_mask
            mask.append(param.reshape(-1))
            # s = f'{s}\n{text}'

        if has_bn and (has_dot_weight or has_bias):
            # layer1.0.bn1.weight OR layer1.0.bn1.bias
            mask.append(torch.ones_like(param).reshape(-1))
            # s = f'{s}\n{text}'

        if has_downsample and has_weight_mask:
            # layer1.0.downsample.0._weight_mask
            mask.append(param.reshape(-1))
            # s = f'{s}\n{text}'

        if has_downsample and ((has_dot_weight and not has_layer_weight) or has_bias):
            # layer1.0.downsample.1.weight OR layer1.0.downsample.1.bias
            mask.append(torch.ones_like(param).reshape(-1))
            # s = f'{s}\n{text}'

    # s = f'{s}\n\nfc mask:'
    # add the masks for the fully connected layer in the model
    for i, (name, param) in enumerate(model.named_parameters()):
        if 'fc' in name:
            mask.append(torch.ones_like(param).reshape(-1))
            # s = f'{s}\n#{i:4d}\t{param.numel():10d}\t{name}'

    mask = torch.cat(mask)
    return mask


@torch.no_grad()
def get_batchnorm_mask(model):
    """
    This method returns a mask containing 1 at location of batch-norm weights and 0 otherwise.
    This mask is used to remove the batch normalization parameters from top-k strategy
    """
    mask = []
    for name, param in model.named_parameters():
        f = torch.ones_like if 'bn' in name else torch.zeros_like
        mask.append(f(param).reshape(-1))
    mask = torch.cat(mask)
    return mask


@torch.no_grad()
def filter_embeddings(model):
    count = 0
    for name, param in model.named_parameters():
        if name.startswith('bert.embeddings'):
            param.requires_grad = False
            param.grad = None
            continue
        count += param.numel()
        yield param
    print(f'#PARAMETERS WITHOUT EMBEDDINGS: {count}')


def get_layer_indices(model):
    name = model._get_name().lower()
    if 'resnet' in name:
        return get_resnet_layer_indices_and_params(model)['indices']
    if 'bert' in name:
        print('Layer indices not implemented yet for BERT models!')
        return None
    return None


def get_param_groups(model, weight_decay):
    names_bn, layers_bn = [], []
    names_non_bn, layers_non_bn = [], []

    for name, p in model.named_parameters():
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

    if len(layers_bn) == 0:  # the case when the model does not BN layers at all
        return [dict(params=layers_non_bn, weight_decay=weight_decay)]
    return [dict(params=layers_bn, weight_decay=0), dict(params=layers_non_bn, weight_decay=weight_decay)]
