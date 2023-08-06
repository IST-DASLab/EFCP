from os import makedirs
from os.path import join, isfile

import numpy
import numpy as np
import random
import socket
import pickle
import torch
import time
import sys
import os
import yaml
import wandb
import platform
from tqdm import tqdm


class MA:
    def __init__(self): pass

    def add(self, x): pass

    def get(self): pass

    def add_get(self, x):
        self.add(x)
        return self.get()


class SMA(MA):
    def __init__(self, size):
        super(SMA, self).__init__()
        self.size = size
        self.buffer = []
        self.index = 0
        self.full = False

    def add(self, x):
        if self.full:
            self.buffer[self.index] = x
            self.index = (1 + self.index) % self.size
        else:
            self.buffer.append(x)
            self.index += 1
            if self.index == self.size:
                self.full = True
                self.index = 0

    def get(self):
        return sum(self.buffer) / (self.size if self.full else self.index)


class EMA(MA):
    def __init__(self, beta):
        assert 0 <= beta <= 1, f'Parameter beta should be in [0, 1], got beta={beta}'
        super(MA, self).__init__()
        self.beta = beta
        self.ema = 0

    def add(self, x):
        self.ema = self.beta * self.ema + (1 - self.beta) * x

    def get(self):
        return self.ema


def on_windows():
    return platform.system().lower() == 'windows'


if not on_windows():
    import gpustat


def cv_experiment_exists(folder):
    if os.path.isdir(folder):
        log = os.path.join(folder, 'log.txt')
        if os.path.isfile(log):
            with open(log, 'r') as handle:
                lines = handle.readlines()
                for line in reversed(lines):
                    if '[main] training ended' in line.lower():
                        return True
    return False


def nlp_experiment_exists(folder):
    if os.path.isdir(folder):
        files = os.listdir(folder)
        if 'all_results.json' in files:
            ex_folder = '/tmp/existence'
            os.makedirs(ex_folder, exist_ok=True)
            f = folder.replace('/tmp/results/test/', '').replace('/', '-') + '.txt'
            with open(os.path.join(ex_folder, f), 'w') as _:
                pass
            return True
    return False


def time_profiler(method, out_file, **kwargs):
    profiler = LineProfiler()
    wrapper = profiler(method)
    output = wrapper(**kwargs)
    with open(out_file, 'w') as w:
        w.write(f'@@@@@ step = {str(method)}')
        profiler.print_stats(stream=w)
    return output


def mkdirs(folder):
    os.makedirs(folder, exist_ok=True)


def get_gpu_mem_usage():
    """
        This method returns the GPU memory usage for the current process.
        It uses gpustat to query the GPU used by the current process (using CUDA_VISIBLE_DEVICES)

        GPUSTAT usage:
        stat = gpustat.new_query().gpus # this is a list containing information about each GPU indexed from 0 to 7
        stat[i] (GPU #i) has the following keys:
            - 'index'
            - 'uuid'
            - 'name'
            - 'temperature.gpu'
            - 'fan.speed'
            - 'utilization.gpu'
            - 'utilization.enc'
            - 'utilization.dec'
            - 'power.draw'
            - 'enforced.power.limit'
            - 'memory.used'
            - 'memory.total'
            - 'processes'
        Among these keys, only the key 'processes' is used here.
        stat[i].processes is a list of dicts, where each dict contains information about each process currently running on the GPU #i
            - 'username'
            - 'command'
            - 'full_command'
            - 'gpu_memory_usage'
            - 'cpu_percent'
            - 'cpu_memory_usage'
            - 'pid'
    """
    if on_windows():
        return 0
    gpus = gpustat.new_query().gpus
    gids = list(map(int, os.environ['CUDA_VISIBLE_DEVICES'].split(',')))
    gpu_mem = sum([int(proc['gpu_memory_usage']) for gid in gids for proc in gpus[gid]['processes'] if int(proc['pid']) == os.getpid()])
    return gpu_mem



def allocate_gpus():
    if on_windows():
        return

    run_distributed = bool(os.environ['RUN_DISTRIBUTED'])

    if not run_distributed:
        user = os.getlogin()
        max_jobs = int(os.environ['MAX_JOBS'])
        while True:
            gpu_stat = gpustat.new_query().gpus
            gpu_int_ids = [g for g in range(len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')))] # ints
            d = {
                gpu_id: len([p for p in gpu_stat[gpu_id].processes if p['username'] == user])
                for gpu_id in gpu_int_ids
            }

            for gpu, count in d.items():
                if count < max_jobs:
                    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
                    dummy_tensor = torch.tensor(data=[0], device=get_first_device())
                    pause_process(10)
                    break
            pause_process(1)


def pause_process(seconds):
    # print(f'Process is paused for {seconds} seconds')
    for _ in tqdm(range(seconds)):
        time.sleep(1)


def setup_wandb(project, job_type, group, name, config):
    return wandb.init(
        project=project,
        job_type=job_type,
        entity='ist',
        group=group,
        name=name,
        config=config,
        settings=None if on_windows() else wandb.Settings(start_method='fork'))


def write_yaml(file, data):
    with open(file, 'w') as w:
        yaml.dump(data, w, default_flow_style=False)


def read_yaml(file):
    with open(file) as f:
        data = yaml.load(f, Loader=yaml.loader.SafeLoader)
        return data


def run_yaml_code(file, key):
    with open(file) as f:
        data = yaml.load(f, Loader=yaml.loader.FullLoader)
        eval(data[key])


def set_all_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def on_colab():
    # return 'google.colab' in str(get_ipython())
    try:
        import google.colab
        return True
    except:
        return False


def link_experiment_library(lib_path):
    if on_colab():
        sys.path.append('/content/ExperimentBuilder-main')
    else:
        # home = os.path.expanduser('~')
        # exp_lib_path = os.path.join(home, 'workplace', 'ExperimentBuilder')
        sys.path.append(lib_path)


def sleep(t):
    print(f'sleeping {t} seconds...')
    for s in range(t):
        print(t-s, end=' ')
        sys.stdout.flush()
        time.sleep(1)
    print()


def torch2numpy(v):
    return v.cpu().detach().numpy()


def norm(v):
    return sum(v ** 2) / v.shape[0]


def get_first_device():
    if not torch.cuda.is_available():
        return torch.device('cpu')
    return torch.device('cuda:0')


def get_gpus(remove_first):
    if not torch.cuda.is_available():
        return ['cpu']

    if torch.distributed.is_available():
        gpus = [f'cuda:{torch.distributed.get_rank()}']
    else:
        if torch.cuda.device_count() == 1:
            gpus = [get_first_device()]
        else:
            gpus = [torch.device(f'cuda:{i}') for i in range(len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')))]
            if remove_first:
                print(f'Removing first device {get_first_device()}')
                gpus.remove(get_first_device())
    print(f'GPUs: {gpus}')
    print(f'CUDA_VISIBLE_DEVICES={os.environ["CUDA_VISIBLE_DEVICES"]}')
    return gpus


def write_pickle(ob, file, mode='wb'):
    with open(file, mode) as handle:
        pickle.dump(ob, handle, protocol=pickle.HIGHEST_PROTOCOL)


def write_to_file(file, text):
    with open(file, 'w') as w:
        w.write(text)
    print(f'wrote text to file {file}')


def log_parameters_from_shell_script(logger, args):
    """
        This method logs the parameters used in the shell script to run the experiment
        to be able to quickly reproduce the experiment again by simply copying these
        parameters back to shell script.
    """
    logger.log(f'mom={args.momentum}')
    logger.log(f'wd={args.weightdecay}')
    logger.log(f'lr_decay={args.lr_decay}')
    logger.log(f'lr={args.lr}')
    logger.log(f'damp={args.damp}')
    logger.log(f'grads={args.ngrads}')
    logger.log(f'seed={args.seed}')
    logger.log('-----------------------------------------------')


def get_sparsity(v):
    return (v == 0).sum() / v.numel()


def map_interval(x, a, b, A, B):
    """This method maps x in [a, b] to y in [A, B]"""
    return A + (B - A) * (x - a) / (b - a)


def save_masks_and_plot_heatmaps(model, mask, step, root_folder):
    step_str = f'{step:06d}'

    numpy_folder = os.path.join(root_folder, 'histograms', FOLDER_NUMPY)
    # htmls_folder = os.path.join(root_folder, 'histograms', FOLDER_HTMLS)

    makedirs(numpy_folder, exist_ok=True)
    # makedirs(htmls_folder, exist_ok=True)

    makedirs(join(numpy_folder, step_str), exist_ok=True)

    layer_names_dict = {}  # key = layer name, value = dict(weight=numpy, bias=numpy/None)
    count = 0

    # create dictionary: 'fc': {'weight'=array(640, 10), 'bias': (10,1)}
    for name, param in model.named_parameters():
        params_count, dim1, dim2 = get_shape(param)
        mask_slice = mask[count:count + params_count].reshape(dim1, dim2)
        count += params_count

        clean_name = name.replace('.weight', '').replace('.bias', '')
        if clean_name not in layer_names_dict:
            layer_names_dict[clean_name] = dict(weight=None, bias=None)

        if name.endswith('weight'):
            layer_names_dict[clean_name]['weight'] = mask_slice
        elif name.endswith('bias'):
            layer_names_dict[clean_name]['bias'] = mask_slice

    save_to_file(
        file_name=join(root_folder, 'histograms', FILE_LAYERS),
        content=[key for key in layer_names_dict])

    for clean_name in layer_names_dict:
        mask = layer_names_dict[clean_name]['weight']
        b = layer_names_dict[clean_name]['bias']

        # MERGE weight and bias matrices (bias will be on the last column)
        if b is not None:
            mask = np.hstack((mask, b))

        # WRITE numpy mask on disk
        np.save(
            file=join(root_folder, 'histograms', FOLDER_NUMPY, step_str, f'{step_str}_{clean_name}'),
            arr=mask)


def save_random_model(model, dataset_name, model_name, seed):
    torch.save(model.state_dict(), join(f'initial_models/{dataset_name}_{model_name}_{seed}.pt'))
    print(f'Saved model {model_name} with seed {seed} for task {dataset_name}')


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
