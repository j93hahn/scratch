from internal import configs
from internal import engine
from absl import app
import gin


def main():
    configs.define_flags()
    with gin.config_scope('eval'):
        config = configs.load_config()
        engine.launch(config, train=False)


if __name__ == '__main__':
    app.run(main)
