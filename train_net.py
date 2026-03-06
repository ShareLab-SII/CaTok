import os.path as osp
import argparse
from omegaconf import OmegaConf
from catok.engine.trainer_utils import instantiate_from_config
from catok.utils.device_utils import configure_compute_backend

def train():
    configure_compute_backend()
    parser = argparse.ArgumentParser("Train CaTok meanflow tokenizer")
    parser.add_argument(
        '--cfg',
        type=str,
        default='configs/catok_b_256.yaml',
        help='Path to trainer config file',
    )
    args = parser.parse_args()

    cfg_file = args.cfg
    assert osp.exists(cfg_file)
    config = OmegaConf.load(cfg_file)
    trainer = instantiate_from_config(config.trainer)
    trainer.train(args.cfg)

if __name__ == '__main__':
    train()
