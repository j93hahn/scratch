import wandb
import os
import numpy as np
from typing import Optional
from pathlib import Path
from einops import rearrange


class WandbWizard():
    def __init__(self,
        project: str,
        name: Optional[str] = None,
        log_dir="./wandb",
        reinit=True    # enable multiple runs from the same script
    ) -> None:
        os.makedirs(log_dir, exist_ok=True)
        name = name or Path(os.getcwd()).name
        wandb.init(
            project=project,
            dir=log_dir,
            name=name,
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

    def save_movies(self, step=None, **kwargs):
        for name, value in kwargs.items():
            wandb.log({name: wandb.Video(value)}, step=step)


if __name__ == "__main__":
    wizard = WandbWizard(project="test")
    for i in range(0, 100, 5):
        wizard.log(step=i, a=i, b=i**2)
        wizard.save_images(step=i, img=np.random.rand(64, 64, 3))
        if i == 50:
            movie = (np.clip(np.random.randn(200, 800, 800, 3), 0, 1) * 255).astype(np.uint8)
            movie = rearrange(movie, 't h w c -> t c h w')
            wizard.save_movies(step=i, movie=movie)
