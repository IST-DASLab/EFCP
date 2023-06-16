import os
import sys
sys.path.append(os.environ['EFCP_ROOT'])

import math
from datetime import datetime

import psutil
from torch.nn.functional import cross_entropy as CE
from torch.utils.data import DataLoader

from helpers.layer_manipulation import get_param_groups
from helpers.lr_scheduling import update_lr
from helpers.mylogger import MyLogger
from helpers.training import get_optimizer, get_model

from lib.data.datasets import *
from args.args_mfac import *


@torch.no_grad()
def test(model, data, batch_size, num_workers, pin_memory):
    if not isinstance(data, DataLoader):
        data = DataLoader(data, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    device = torch.device(get_first_device())
    loss, correct = 0, 0
    dataset_size = len(data.dataset)
    for x, y in tqdm(data):
        x, y = x.to(device), y.to(device)
        y_hat = model(x)

        loss += CE(y_hat, y, reduction='sum').item()
        pred = y_hat.argmax(dim=1, keepdim=True)
        correct += pred.eq(y.view_as(pred)).sum().item()
    accuracy = correct / dataset_size
    loss /= dataset_size
    return accuracy, loss


def train(logger, args, model, train_data, test_data, optim, epochs, decay_at, batch_size):
    start_time = datetime.now()
    step = 0
    n_params = sum([p.numel() for p in model.parameters()])
    logger.log(f'Model has {n_params} parameters')

    MyLogger.setup(name='debug', path=os.path.join(args.root_folder, 'loop-debugging.txt'))
    n_samples = len(train_data)
    steps_per_epoch = math.ceil(n_samples / batch_size)
    total_steps = steps_per_epoch * epochs
    decay_at = [dec * steps_per_epoch for dec in decay_at]
    for epoch in range(1, epochs + 1):
        logger.log(f'Epoch {epoch}/{epochs}: {args.wandb_group}_{args.wandb_job_type}')

        train_loss, test_loss, test_accuracy, train_accuracy = 0., 0, 0, 0
        predictions, targets = [], []

        train_start = datetime.now()
        for x, y in tqdm(DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=6, pin_memory=True)):
            lr = update_lr(sched=args.lr_sched,
                           optimizer=optim,
                           base_lr=args.base_lr,
                           step=step,
                           steps=100_100,
                           decay_at=decay_at,
                           step_rate=0.1)
            x, y = x.to(model.device), y.to(model.device)

            if args.optim in ['lbfgs', 'flbfgs']:
                def closure():
                    optim.zero_grad()
                    y_hat = model(x)
                    loss = CE(y_hat, y)
                    loss.backward()
                    return loss
                optim.step(closure)
                y_hat = model(x)
                loss = closure()
            else:
                optim.zero_grad()
                y_hat = model(x)
                loss = CE(y_hat, y)
                with torch.no_grad():
                    if torch.isnan(loss):
                        MyLogger.get('debug').log(f'FOUND NaN LOSS AT STEP {step}')
                        wandb.log(dict(end_reason=f'FoundNanLoss@Epoch{epoch}Step{step}'))
                        sys.exit(666)
                loss.backward()

                if args.clip_grad_val > 0:
                    torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=args.clip_grad_val)

                optim.step()

            step_loss = loss.item() * x.size(0) / n_samples
            train_loss += step_loss
            step += 1

            predictions.append(torch.argmax(y_hat, 1))
            targets.append(y)

            wandb.log({'step': step, 'step/step': step, 'step/lr': lr})
        # end for

        train_accuracy = torch.mean((torch.cat(predictions) == torch.cat(targets)).float()).item()

        test_start = datetime.now()
        model.eval()
        test_accuracy, test_loss = test(model, test_data, batch_size=batch_size, num_workers=6, pin_memory=True)
        model.train()

        now = datetime.now()
        total_elapsed = now - start_time
        train_elapsed = now - train_start
        test_elapsed = now - test_start

        d = {
            'epoch/step': step,
            'epoch/epoch': epoch,
            'epoch/train_acc': train_accuracy,
            'epoch/train_loss': train_loss,
            'epoch/test_acc': test_accuracy,
            'epoch/test_loss': test_loss,
            'epoch/epoch_time': train_elapsed.total_seconds(),
            'epoch/total_elapsed': total_elapsed.total_seconds(),
            'epoch/lr': lr,
            'epoch/ram_mem_usage': round(psutil.Process().memory_info().rss / (2 ** 30), 2),
            'epoch/gpu_mem_usage': get_gpu_mem_usage()
        }
        wandb.log(d)

        logger.log(f'Loss Train/Test:    \t{train_loss:.4f} / {test_loss:.4f}')
        logger.log(f'Accuracy Train/Test:\t{train_accuracy*100:.2f}% / {test_accuracy*100:.2f}%')
        logger.log(f'Elapsed Train/Test/Total:\t{train_elapsed} / {test_elapsed} / {total_elapsed}')
        logger.log(f'Current/Base Learning Rate: {lr} / {args.base_lr}')

        if hasattr(optim, 'sparsity_gradient') and hasattr(optim, 'sparsity_update'):
            logger.log(f'sparsity_gradient: {optim.sparsity_gradient:.2f}%')
            logger.log(f'sparsity_update: {optim.sparsity_update:.2f}%')

        logger.log(f'Steps so far: {step}\n')

        if not bool(args.ignore_checks):
            # define some rules to stop the process earlier
            if loss.item() > 10:
                MyLogger.get('debug').log(f'ENDED @step{step} BECAUSE LOSS = {loss.item()} > 10')
                wandb.log(dict(end_reason=f'TooLargeLoss@Epoch={epoch}:{train_loss:.2f}>10'))
                break
            if epoch == 3 and test_accuracy < 0.15:
                MyLogger.get('debug').log(f'ENDED @step{step} BECAUSE test_acc < 0.15')
                wandb.log(dict(end_reason=f'TooLowAcc@Epoch={epoch}:{test_accuracy:.2f}<0.15'))
                break

    logger.log(f'Training ended at {datetime.now()}')


def main():
    args = get_arg_parse()

    if cv_experiment_exists(args.root_folder):
        print(f'Experiment {args.root_folder} already exists!')
        return

    set_all_seeds(args.seed)


    logger = MyLogger(name='main', path=os.path.join(args.root_folder, 'log.txt'))

    MyLogger.setup(name='sparse-hinv', path=os.path.join(args.root_folder, 'sparse-hinv.txt'))
    MyLogger.setup(name='optimizer', path=os.path.join(args.root_folder, 'optimizer-args.txt'))

    model = get_model(args.model, args.dataset_name)
    param_groups = get_param_groups(model, args.weight_decay)
    optim = get_optimizer(args, param_groups)

    if hasattr(optim, 'set_named_parameters'):
        optim.set_named_parameters(list(model.named_parameters()))

    os.makedirs('initial_models', exist_ok=True)
    model_name = f'initial_models/{args.dataset_name}_{args.model}_{args.seed}.pt'
    if args.load_random_model == 1:
        if not os.path.isfile(model_name):
            save_random_model(model, args.dataset_name, args.model, args.seed)
        model.load_state_dict(torch.load(model_name))
        print(f'Loaded model {args.model} with seed {args.seed}')

    train_data, test_data = get_datasets(args.dataset_name, args.dataset_path)

    if not on_windows():
        torch.cuda.set_device(args.dev)

    model = model.to(args.dev)
    if len(args.gpus) > 1:
        model = torch.nn.DataParallel(model, device_ids=args.gpus)
    model.device = args.dev
    model.dtype = torch.float32
    model.train()

    if args.optim.lower() in ['gd', 'flbfgs']: # change this right before logging args to wandb
        train_size = len(train_data)
        print(f'Using {args.optim.upper()}: set batch_size from {args.batch_size} to {train_size}')
        args.batch_size = train_size

    setup_wandb(args.wandb_project, args.wandb_job_type, args.wandb_group, args.wandb_name, args)

    train(
        logger=logger,
        args=args,
        model=model,
        train_data=train_data,
        test_data=test_data,
        optim=optim,
        epochs=args.epochs,
        decay_at=[int(args.epochs * 0.5), int(args.epochs * 0.75)],
        batch_size=args.batch_size)
    print('main ended')


if __name__ == '__main__':
    main()
    max_memory = torch.cuda.max_memory_allocated() / float(1 << 30)
    wandb.log(dict(cuda_max_memory=max_memory))
    print(f'cuda_max_memory: {max_memory:.2f}GB')
