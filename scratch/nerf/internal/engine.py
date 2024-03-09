import sys
import pathlib
import torch
import torch.nn as nn
import torch.optim as optim
from scratch.utils.wizard import WandbWizard
from scratch.nn.scheduler import DecayLR
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
    scheduler = DecayLR(optimizer, **lr_kwargs)
    return optimizer, scheduler
