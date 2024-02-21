import torch
import torch.nn as nn
import torch.optim as optim
import gin
import functools
from internal import math
from internal import models
from internal import configs




def get_lr_fn(lr_init, lr_final, **lr_kwargs):
    return functools.partial(
        math.create_lr_func,
        lr_init=lr_init,
        lr_final=lr_final,
        **lr_kwargs
    )


class Schedule(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, lr_fn):
        self.lr_fn = lr_fn
        self._step = 0
        super().__init__(optimizer)

    def get_lr(self):

        return [self.lr_fn(self.step)]

    def step(self, step=None):
        if step is None:
            step = self._step
        self._step = step
        super().step()



def create_optimizer(
    model: models.Model,
    configs: configs.Config
):
    adam_kwargs = {
        'betas': (configs.beta1, configs.beta2),
        'eps': configs.eps
    }
    lr_kwargs = {
        'lr_delay_steps': configs.lr_delay_steps,
        'lr_delay_mult': configs.lr_delay_mult,
        'max_steps': configs.max_steps
    }
    lr_fn = get_lr_fn(configs.lr_init, configs.lr_final, **lr_kwargs)
    optimizer = optim.Adam( # TODO: think if grids need different lr from MLP
        model.parameters(),
        lr=configs.lr_init,
        **adam_kwargs
    )
    scheduler = Schedule(optimizer, lr_fn)
    return optimizer, lr_fn
