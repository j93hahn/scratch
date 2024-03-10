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
        config: Optional[dict] = None,
        log_dir="./wandb",
        reinit=True    # enable multiple runs from the same script
    ) -> None:
        os.makedirs(log_dir, exist_ok=True)
        name = name or Path(os.getcwd()).name
        wandb.init(
            project=project,
            dir=log_dir,
            name=name,
            reinit=reinit,
            config=config,
        )

    def update_config(self, **config_dict) -> None:
        wandb.config.update(config_dict, allow_val_change=True)

    @staticmethod
    def log_scalars(step=None, **kwargs):
        wandb.log(kwargs, step=step)

    @staticmethod
    def log_distributions(step=None, **kwargs):
        """
        You can only see the histograms for a given run, not across the whole project.
        https://github.com/wandb/wandb/issues/1211#issuecomment-993504902

        For more detailed plots, just use matplotlib and seaborn locally.
        """
        for name, value in kwargs.items():
            assert isinstance(value, np.ndarray), "Statistics must be numpy arrays"
            wandb.log({name: wandb.Histogram(value)}, step=step)

    @staticmethod
    def log_images(step=None, **kwargs):
        for name, value in kwargs.items():
            wandb.log({name: wandb.Image(value)}, step=step)

    @staticmethod
    def log_movies(step=None, **kwargs):
        for name, value in kwargs.items():
            wandb.log({name: wandb.Video(value)}, step=step)

    def finish(self):
        wandb.finish()


if __name__ == "__main__":
    config = {"lr": 1e-3, "step_size": 5, "optimizer": "adam"}
    wizard = WandbWizard(project="test", name="test_run", config=config)
    for i in range(0, 100, 5):
        wizard.log_scalars(step=i, a=i, b=i**2)
        wizard.log_images(step=i, img=np.random.rand(64, 64, 3))
        if i == 50:
            movie = (np.clip(np.random.randn(40, 60, 60, 3), 0, 1) * 255).astype(np.uint8)
            movie = rearrange(movie, 't h w c -> t c h w')
            wizard.log_movies(step=i, movie=movie)
