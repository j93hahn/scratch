import torch
from internal import fields
from internal import configs
from lightning import seed_everything
import gin
from scratch.utils.wizard import WandbWizard


def main():
    args = configs.parse_args()
    config = configs.parse_yaml(args.config)
    print(config)


if __name__ == '__main__':
    with gin.config_scope('train'):
        main()
