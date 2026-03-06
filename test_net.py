import argparse
import itertools
import os.path as osp

from omegaconf import OmegaConf

from catok.engine.trainer_utils import instantiate_from_config
from catok.utils.device_utils import configure_compute_backend


def parse_args():
    parser = argparse.ArgumentParser("Evaluate CaTok meanflow tokenizer")
    parser.add_argument('--model', type=str, nargs='+', default=[None], help="Path to checkpoint root")
    parser.add_argument('--step', type=int, nargs='+', default=[250000], help="Checkpoint step")
    parser.add_argument('--cfg', type=str, default=None, help="Path to config file")
    parser.add_argument('--dataset', type=str, default='imagenet', help="Dataset name (imagenet|coco)")
    parser.add_argument('--data_root', type=str, default=None, help="Dataset root override")
    parser.add_argument('--test_split', type=str, default='val', help="Test split name")
    parser.add_argument('--fid_stats', type=str, default=None, help="Path to precomputed FID stats")
    parser.add_argument('--fid_real_dir', type=str, default=None, help="Real image folder for FID")
    parser.add_argument('--cfg_value', type=float, nargs='+', default=[None], help="Classifier-free guidance scale")
    parser.add_argument('--test_num_slots', type=int, nargs='+', default=[None], help="Slot count for inference")
    parser.add_argument('--test_num_steps', type=int, nargs='+', default=[None], help="Sampling step count")
    parser.add_argument('--num_test_images', type=int, nargs='+', default=[50000], help="Number of images for FID")
    return parser.parse_args()


def load_config(model_path, cfg_path=None):
    if cfg_path is not None and osp.exists(cfg_path):
        config_path = cfg_path
    elif model_path and osp.exists(osp.join(model_path, 'config.yaml')):
        config_path = osp.join(model_path, 'config.yaml')
    else:
        raise ValueError(f"No config file found at {model_path} or {cfg_path}")
    return OmegaConf.load(config_path)


def setup_checkpoint_path(model_path, step, config):
    if model_path:
        ckpt_path = osp.join(model_path, 'models', f'step{step}')
    else:
        result_folder = config.trainer.params.result_folder
        ckpt_path = osp.join(result_folder, 'models', f'step{step}')

    if not osp.exists(ckpt_path):
        print(f"Skipping non-existent checkpoint: {ckpt_path}")
        return None

    config.trainer.params.model.params.ckpt_path = ckpt_path
    return ckpt_path


def setup_test_config(config, args):
    dataset_cfg = OmegaConf.create(OmegaConf.to_container(config.trainer.params.dataset, resolve=True))
    dataset_name = (args.dataset or "imagenet").lower()
    if dataset_name in ("coco", "coco2017"):
        dataset_cfg.target = "catok.utils.datasets.COCO2017"
    elif dataset_name in ("imagenet", "imagenet1k"):
        dataset_cfg.target = "catok.utils.datasets.ImageNet"
    if args.data_root is not None:
        dataset_cfg.params.root = args.data_root
    config.trainer.params.dataset = dataset_cfg

    config.trainer.params.test_dataset = OmegaConf.create(OmegaConf.to_container(dataset_cfg, resolve=True))
    config.trainer.params.test_dataset.params.split = args.test_split
    config.trainer.params.test_only = True
    config.trainer.params.compile = False
    config.trainer.params.eval_fid = True

    if args.fid_stats is not None:
        if args.fid_stats.strip().lower() in ("none", "null"):
            config.trainer.params.fid_stats = None
        else:
            config.trainer.params.fid_stats = args.fid_stats
    else:
        if dataset_name in ("coco", "coco2017"):
            config.trainer.params.fid_stats = "fid_stats/coco_val2017_stats.npz"
        else:
            config.trainer.params.fid_stats = "fid_stats/adm_in256_stats.npz"

    if args.fid_real_dir is not None:
        config.trainer.params.fid_real_dir = args.fid_real_dir

    config.trainer.params.model.params.num_sampling_steps = '250'


def apply_cfg_params(config, param_dict):
    if param_dict.get('cfg_value') is not None:
        config.trainer.params.cfg = param_dict['cfg_value']
        print(f"Setting cfg to {param_dict['cfg_value']}")
    if param_dict.get('test_num_slots') is not None:
        config.trainer.params.test_num_slots = param_dict['test_num_slots']
        print(f"Setting test_num_slots to {param_dict['test_num_slots']}")
    if param_dict.get('test_num_steps') is not None:
        config.trainer.params.test_num_steps = param_dict['test_num_steps']
        print(f"Setting test_num_steps to {param_dict['test_num_steps']}")
    if param_dict.get('num_test_images') is not None:
        config.trainer.params.num_test_images = param_dict['num_test_images']
        print(f"Setting num_test_images to {param_dict['num_test_images']}")


def generate_param_combinations(args):
    param_grid = {
        'cfg_value': [None] if args.cfg_value == [None] else args.cfg_value,
        'test_num_slots': [None] if args.test_num_slots == [None] else args.test_num_slots,
        'test_num_steps': [None] if args.test_num_steps == [None] else args.test_num_steps,
        'num_test_images': [None] if args.num_test_images == [None] else args.num_test_images,
    }
    active_params = [k for k, v in param_grid.items() if v != [None]]
    if not active_params:
        yield {k: None for k in param_grid}
        return
    active_values = [param_grid[k] for k in active_params]
    for combination in itertools.product(*active_values):
        param_dict = {k: None for k in param_grid}
        for i, param_name in enumerate(active_params):
            param_dict[param_name] = combination[i]
        yield param_dict


def run_test(config):
    trainer = instantiate_from_config(config.trainer)
    trainer.train()


def test(args):
    for model in args.model:
        for step in args.step:
            print(f"Testing model: {model} at step: {step}")
            config = load_config(model, args.cfg)
            ckpt_path = setup_checkpoint_path(model, step, config)
            if ckpt_path is None:
                continue
            setup_test_config(config, args)
            for param_dict in generate_param_combinations(args):
                current_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
                param_str = ", ".join([f"{k}={v}" for k, v in param_dict.items() if v is not None])
                print(f"Testing with parameters: {param_str}")
                apply_cfg_params(current_config, param_dict)
                run_test(current_config)


def main():
    args = parse_args()
    configure_compute_backend()
    test(args)


if __name__ == "__main__":
    main()
