import sys
import pathlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from scratch.utils.wizard import WandbWizard
from tqdm.auto import tqdm
from internal import models
from internal import configs
from internal import datasets


def launch(config, train=True):
    if train:
        trainer = Trainer(config)
        trainer.run()
    else:
        trainer = Trainer(config).load_checkpoint()
        trainer.inference()


class Trainer(nn.Module):
    def __init__(self, config: configs.Config):
        super().__init__()
        self.wizard = WandbWizard(config.name, config.save_dir)
        self.dataloader = datasets.dataset_dict[config.dataset](config.data_dir)

        self.model = models.Model(config)
        self.optimizer, self.scheduler = create_optimizer(self.model, config)

        self.config = config

    def run(self):
        pbar = tqdm(range(self.config.max_steps + 1), miniters=self.config.update_pbar, file=sys.stdout)
        for step in pbar:
            ...

            rgb = self.model.train_step()


            if step % self.config.checkpoint_every == 0 and step > 0:
                self.checkpoint(step)

    def inference(self):
        pass

    def checkpoint(self, step):
        torch.save(self.model.state_dict(), pathlib.Path(self.config.checkpoint_dir, f"model_{step}.pt"))

    def load_checkpoint(self):
        try:
            path = sorted(pathlib.Path(self.config.checkpoint_dir).glob("model_*.pt"))[-1]
            self.model.load_state_dict(torch.load(path))
            print(f"Loaded model from {path}")
        except IndexError:
            raise FileNotFoundError(f"No checkpoint found in {self.config.checkpoint_dir}")
        return self


class DelayLRScheduler(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1, verbose=False,
                 lr_init=5e-2, lr_final=5e-4, lr_delay_steps=1000,
                 lr_delay_mult=1, max_steps=25000):
        super().__init__(optimizer, last_epoch, verbose)
        self.lr_init = lr_init
        self.lr_final = lr_final
        self.lr_delay_steps = lr_delay_steps
        self.lr_delay_mult = lr_delay_mult
        self.max_steps = max_steps

    def get_lr(self):
        # taken from Plenoxels and JaxNeRF
        if self.lr_delay_steps > 0: # apply a reverse cosine delay
            delay_rate = self.lr_delay_mult + (1 - self.lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(self.last_epoch / self.lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(self.last_epoch / self.max_steps, 0, 1)
        log_lerp = np.exp(np.log(self.lr_init) * (1 - t) + np.log(self.lr_final) * t)
        return [delay_rate * log_lerp for _ in self.base_lrs]


def create_optimizer(model: models.Model, configs: configs.Config):
    adam_kwargs = {
        'betas': (configs.beta1, configs.beta2),
        'eps': configs.eps
    }
    lr_kwargs = {
        'lr_init': configs.lr_init,
        'lr_final': configs.lr_final,
        'lr_delay_steps': configs.lr_delay_steps,
        'lr_delay_mult': configs.lr_delay_mult,
        'max_steps': configs.max_steps
    }
    optimizer = optim.Adam(
        model.parameters(),
        lr=configs.lr_init,
        **adam_kwargs
    )
    scheduler = DelayLRScheduler(optimizer, **lr_kwargs)
    return optimizer, scheduler
