import wandb
import os
import torch
from torchtyping import TensorType
from pathlib import Path
from jaxtyping import Float
from .writer import Metric


_WANDB_STORAGE_STACK = []


def get_wandb_storage():
    """Returns:
        The :class:`WandbStorage` object that's currently being used.
        Throws an error if no :class:`WandbStorage` is currently enabled.
    """
    assert len(
        _WANDB_STORAGE_STACK
    ), "get_wandb_storage() has to be called inside a 'with WandbStorage(...)' context!"
    storage: WandbWizard = _WANDB_STORAGE_STACK[-1]
    return storage


class WandbWizard():
    def __init__(self,
        log_dir: Path,              # local directory where wandb logs are stored
        project_name: str,          # name of the wandb project
        experiment_name: str=None,  # name of the current experiment
        mode: str="online"
    ) -> None:
        assert mode in [
            "online",   # online logging
            "offline",  # offline logging
            "disabled"  # no logging
        ], f"Invalid mode '{mode}' specified for WandbStorage."
        os.environ["WANDB_MODE"] = mode
        os.environ["WANDB_API_KEY"] = Path(__file__).parent.parent.joinpath("wandb.key").read_text()

        log_dir = os.environ.get("WANDB_DIR", log_dir.__str__())
        os.makedirs(log_dir, exist_ok=True)
        wandb.init(     # calls wandb.login() if necessary
            project=os.environ.get("WANDB_PROJECT", project_name),
            dir=log_dir,
            name=os.environ.get("WANDB_NAME", experiment_name),
            reinit=True # enable multiple runs from the same script
        )

    def set_config(self, config_dict: dict) -> None:
        """Sets the config for the current run."""
        wandb.config.update(config_dict, allow_val_change=True)

    def write_scalar(self, name: str, value: float, step: int):
        wandb.log({name: value}, step=step)

    def write_image(self, name: str, value: Float[TensorType, "C H W"], step: int):
        wandb.log({name: wandb.Image(value)}, step=step)

    def _close(self):
        wandb.finish()

    def __enter__(self):
        _WANDB_STORAGE_STACK.append(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        _WANDB_STORAGE_STACK.pop()
        self._close()


if __name__ == "__main__":
    with WandbWizard(Path("wandb"), "tensoRF") as w:
        for i in range(100):
            w.write_scalar("psnr", i, i)
            w.write_image("test", torch.randn((3, 32, 32)), i)
