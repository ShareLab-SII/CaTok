#!/usr/bin/env python3
"""
Example:
python scripts/infer_recon.py \
  --checkpoint /path/to/models/step50040/custom_checkpoint_1.pkl \
  --image /path/to/your/input_image.webp \
  --cfg 2.0 \
  --num-tokens 256 \
  --start-token 0 \
  --sample-steps 25 \
  --output-dir ./infer_outputs
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple

if "graalpy" in sys.version.lower() or "graalvm" in sys.version.lower():
    raise SystemExit(
        "Detected GraalPy runtime. This script requires CPython due to torch/numpy binary extensions.\n"
        "Please run with a CPython interpreter, e.g.:\n"
        "  PYTHON_BIN=/usr/bin/python bash scripts/run_infer.sh ...\n"
        "or use a CPython conda env and run scripts/infer_recon.py there."
    )

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from PIL import Image

try:
    from omegaconf import OmegaConf
except ModuleNotFoundError as exc:
    raise SystemExit(
        "ModuleNotFoundError: omegaconf is not available in the current interpreter.\n"
        f"Current python: {sys.executable}\n"
        f"Install with: {sys.executable} -m pip install omegaconf"
    ) from exc

from catok.engine.trainer_utils import load_safetensors, load_state_dict
from catok.tokenizer.meanflow_slot import MeanFlowSlot
from catok.utils.datasets import vae_transforms


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser("CaTok reconstruction inference")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=PROJECT_ROOT / "output" / "catok_b_256",
        help="Model output folder containing models/step* checkpoints",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "catok_b_256.yaml",
        help="Config yaml used to build model",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint path: step dir, *.pkl, or *.safetensors. If omitted, auto-pick latest under model-dir/models",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Single image path to reconstruct",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Directory of images to reconstruct",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./infer_outputs"),
        help="Output directory",
    )
    parser.add_argument(
        "--cfg",
        type=float,
        default=2.0,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=256,
        help="How many tokens participate in reconstruction",
    )
    parser.add_argument(
        "--start-token",
        type=int,
        default=0,
        help="Start token index",
    )
    parser.add_argument(
        "--sample-steps",
        type=int,
        default=25,
        help="Sampling steps",
    )
    parser.add_argument(
        "--vae-path",
        type=str,
        default=str(PROJECT_ROOT / "pretrained" / "mar-vae-kl16"),
        help="Override VAE path",
    )
    parser.add_argument(
        "--disable-repa",
        action="store_true",
        default=True,
        help="Disable REPA encoder during inference for stability/offline use (default: enabled)",
    )
    parser.add_argument(
        "--enable-repa",
        dest="disable_repa",
        action="store_false",
        help="Keep REPA encoder as defined in config",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        default=True,
        help="Use mixed precision on CUDA (default: enabled)",
    )
    parser.add_argument(
        "--no-amp",
        dest="amp",
        action="store_false",
        help="Disable mixed precision",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device",
    )
    parser.add_argument(
        "--save-recon-only",
        action="store_true",
        help="If set, save only reconstruction image (otherwise save side-by-side)",
    )
    return parser.parse_args()


def _step_value(step_dir: Path) -> int:
    match = re.search(r"step(\d+)", step_dir.name)
    return int(match.group(1)) if match else -1


def find_latest_checkpoint(model_dir: Path) -> Optional[Path]:
    models_dir = model_dir / "models"
    if not models_dir.exists():
        return None

    step_dirs = sorted(
        [p for p in models_dir.glob("step*") if p.is_dir()],
        key=_step_value,
    )
    if not step_dirs:
        return None

    # Prefer EMA checkpoint first, then safetensors.
    for step_dir in reversed(step_dirs):
        ema = step_dir / "custom_checkpoint_1.pkl"
        safe = step_dir / "model.safetensors"
        if ema.exists():
            return ema
        if safe.exists():
            return safe
    return step_dirs[-1]


def load_checkpoint(model: torch.nn.Module, ckpt_path: Path) -> Path:
    if ckpt_path.is_dir():
        ema = ckpt_path / "custom_checkpoint_1.pkl"
        safe = ckpt_path / "model.safetensors"
        if ema.exists():
            state = torch.load(str(ema), map_location="cpu")
            load_state_dict(state, model)
            return ema
        if safe.exists():
            load_safetensors(str(safe), model)
            return safe
        raise FileNotFoundError(f"No checkpoint file found in step dir: {ckpt_path}")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if ckpt_path.suffix == ".safetensors":
        load_safetensors(str(ckpt_path), model)
    else:
        state = torch.load(str(ckpt_path), map_location="cpu")
        load_state_dict(state, model)
    return ckpt_path


def build_model(args) -> Tuple[MeanFlowSlot, Path]:
    cfg = OmegaConf.load(str(args.config))
    model_params = cfg.trainer.params.model.params

    if args.vae_path:
        model_params.vae = args.vae_path
    if args.disable_repa:
        model_params.use_repa = False

    model = MeanFlowSlot(**OmegaConf.to_container(model_params, resolve=True))

    ckpt_path = args.checkpoint or find_latest_checkpoint(args.model_dir)
    if ckpt_path is None:
        raise FileNotFoundError(
            f"No checkpoint found under {args.model_dir}/models. "
            "Please provide --checkpoint explicitly."
        )
    ckpt_loaded = load_checkpoint(model, Path(ckpt_path))
    model = model.to(args.device).eval()
    model.enable_nest = True
    return model, ckpt_loaded


def to_uint8(img_tensor: torch.Tensor) -> np.ndarray:
    arr = img_tensor.detach().cpu().clamp(0, 1).numpy()
    arr = (arr.transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
    return arr


def calc_psnr(orig: np.ndarray, recon: np.ndarray, eps: float = 1e-8) -> float:
    x = orig.astype(np.float32) / 255.0
    y = recon.astype(np.float32) / 255.0
    mse = np.mean((x - y) ** 2)
    return float(10.0 * np.log10(1.0 / (mse + eps)))


def collect_images(image: Optional[Path], image_dir: Optional[Path]) -> Iterable[Path]:
    if image is not None:
        if not image.exists():
            raise FileNotFoundError(f"Image not found: {image}")
        return [image]

    if image_dir is not None:
        if not image_dir.exists():
            raise FileNotFoundError(f"Image dir not found: {image_dir}")
        return sorted([p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])

    raise ValueError("Please provide --image or --image-dir")


@torch.no_grad()
def reconstruct_one(model: MeanFlowSlot, img_path: Path, args, transform) -> Tuple[np.ndarray, np.ndarray, float, int, int]:
    image = Image.open(img_path).convert("RGB")
    img = transform(image).unsqueeze(0).to(args.device)

    max_slots = int(model.num_slots)
    num_tokens = int(max(1, min(args.num_tokens, max_slots)))
    start_token = int(max(0, args.start_token))
    if start_token + num_tokens > max_slots:
        start_token = max_slots - num_tokens

    if args.device.startswith("cuda"):
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp)
    else:
        autocast_ctx = torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=False)

    with autocast_ctx:
        recon = model(
            img,
            sample=True,
            cfg=float(args.cfg),
            inference_with_n_slots=int(num_tokens),
            inference_begin=int(start_token),
            sample_steps=int(args.sample_steps),
        )

    orig_np = to_uint8(img[0])
    recon_np = to_uint8(recon[0])
    psnr = calc_psnr(orig_np, recon_np)
    return orig_np, recon_np, psnr, num_tokens, start_token


def save_output(orig_np: np.ndarray, recon_np: np.ndarray, output_path: Path, save_recon_only: bool):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if save_recon_only:
        out = Image.fromarray(recon_np)
    else:
        if orig_np.shape[:2] != recon_np.shape[:2]:
            orig_img = Image.fromarray(orig_np).resize(
                (recon_np.shape[1], recon_np.shape[0]),
                Image.LANCZOS,
            )
            orig_np = np.array(orig_img)
        canvas = np.hstack([orig_np, recon_np])
        out = Image.fromarray(canvas)
    out.save(output_path)


def main():
    args = parse_args()

    model, ckpt_loaded = build_model(args)
    transform = vae_transforms("test", img_size=model.enc_img_size)
    images = list(collect_images(args.image, args.image_dir))
    if not images:
        raise RuntimeError("No images found")

    print(f"Loaded checkpoint: {ckpt_loaded}")
    print(f"Device: {args.device}")
    print(
        "Controls -> "
        f"cfg={args.cfg}, num_tokens={args.num_tokens}, start_token={args.start_token}, sample_steps={args.sample_steps}"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for img_path in images:
        orig_np, recon_np, psnr, n_tok, s_tok = reconstruct_one(model, img_path, args, transform)
        out_name = (
            f"{img_path.stem}_cfg{args.cfg}_n{n_tok}_start{s_tok}_steps{args.sample_steps}.png"
        )
        out_path = args.output_dir / out_name
        save_output(orig_np, recon_np, out_path, args.save_recon_only)
        print(f"[OK] {img_path} -> {out_path} | PSNR={psnr:.2f} dB")


if __name__ == "__main__":
    main()
