import numpy as np


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def update_lr(sched, optimizer, base_lr, step, steps, decay_at, step_rate):
    if sched == 'cos':
        lr = schedule_cosine(base_lr=base_lr, step=step, steps=steps)
    elif sched == 'step':
        lr = schedule_multistep(base_lr=base_lr, step=step, decay_at_steps=decay_at, rate=step_rate)
    elif sched == 'const':
        lr = base_lr
    else:
        raise RuntimeError(f'Learning rate schedule "{sched}" is not implemented!')
    set_lr(optimizer, lr)
    return lr


def schedule_cosine(base_lr, step, steps):
    progress = step / steps
    lr = 0.5 * (1 + np.cos(np.pi * progress)) * base_lr
    return lr


def schedule_multistep(base_lr, step, decay_at_steps, rate):
    count = sum([int(step >= decay_step) for decay_step in decay_at_steps])
    return base_lr * (rate ** count)


# ##### SCHEDULES CALLABLE PER EPOCH
# def lr_sched_warmup_per_epoch(base_lr, warmup_length, epoch):
#     return base_lr * (epoch + 1) / warmup_length
#
#
# def lr_sched_cos_w_warmup_per_epoch(base_lr, epoch, epochs, warmup_length):
#     if epoch < warmup_length:
#         lr = lr_sched_warmup_per_epoch(base_lr, warmup_length, epoch)
#     else:
#         e = epoch - warmup_length
#         es = epochs - warmup_length
#         lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
#     return lr
