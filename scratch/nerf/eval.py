from internal import configs
from internal import engine
from lightning import seed_everything
from absl import app
import gin


def main():
    configs.define_flags()
    with gin.config_scope('eval'):
        config = configs.load_config()
        seed_everything(config.seed)
        engine.launch(config, train=False)


if __name__ == '__main__':
    app.run(main)
