import wandb
import os
import numpy as np
from pathlib import Path


class WandbWizard():
    def __init__(self,
        project: str,
        log_dir="./wandb",
        reinit=False    # enable multiple runs from the same script - not necessary for our purposes
    ) -> None:
        os.makedirs(log_dir, exist_ok=True)
        wandb.init(
            project=project,
            dir=log_dir,
            name=Path(os.getcwd()).name,    # use name of current directory; assumes circe is already called to plant experiments
            reinit=reinit
        )

        if os.path.exists('config.json'):
            import json
            with open('config.json') as f:
                config_dict = json.load(f)
            self.update_config(**config_dict)

    def update_config(self, **config_dict) -> None:
        wandb.config.update(config_dict, allow_val_change=True)

    def log(self, step=None, **kwargs):
        wandb.log(kwargs, step=step)

    def save_images(self, step=None, **kwargs):
        for name, value in kwargs.items():
            wandb.log({name: wandb.Image(value)}, step=step)


if __name__ == "__main__":
    wizard = WandbWizard(project="test")
    for i in range(0, 100, 5):
        wizard.log(step=i, a=i, b=i**2)
        wizard.save_images(step=i, img=np.random.rand(64, 64, 3))
