<div align="center">
<h2><font size=3>CaTok: Taming Mean Flows for One-Dimensional Causal Image Tokenization</font></h2>
<h4>

[Yitong Chen](https://scholar.google.com/citations?hl=en&user=a40b6HQAAAAJ)<sup>1,2,3</sup>, [Zuxuan Wu](https://zxwu.azurewebsites.net/)<sup>1,2,3</sup>, [Xipeng Qiu](https://xpqiu.github.io/en.html)<sup>1,2,3</sup>, [Yu-Gang Jiang](https://scholar.google.com/citations?user=f3_FP8AAAAAJ&hl=en)<sup>1,3</sup>

<sup>1</sup> Institute of Trustworthy Embodied AI, Fudan University <br>
<sup>2</sup> Shanghai Innovation Institute <br>
<sup>3</sup> Shanghai Key Laboratory of Multimodal Embodied AI

[[`Paper CVPR-26`](#)]
[[`Project Page`](https://sharelab-sii.github.io/catok-web/)]
</h4>
</div>


## Installation

1. Clone repository:
```bash
git clone https://github.com/ShareLab-SII/CaTok
cd CaTok
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation

Please first download [ImageNet](https://www.image-net.org/) on your own path,
then soft-link it to this repo:

```bash
mkdir -p dataset
ln -s /path/to/imagenet ./dataset/imagenet
```

Expected layout:

```text
dataset/
  imagenet/
    train/
    val/
    val256/   # optional, for FID real images
```

The default config uses `./dataset/imagenet/`.

For FID evaluation, it is recommended to preprocess ImageNet validation images
to `256x256` and place them in `./dataset/imagenet/val256`
(you can use this script:
[prepare_imgnet_val.py](https://github.com/LTH14/rcg/blob/main/prepare_imgnet_val.py)).

## Pretrained Dependencies

Create pretrained folder:

```bash
mkdir -p pretrained
```

1. Download MAR VAE (`./pretrained/mar-vae-kl16`):

```bash
huggingface-cli download xwen99/mar-vae-kl16 --local-dir ./pretrained/mar-vae-kl16
```

2. REPA backbone:
- Supported presets in code:
  - `dinov2` -> `facebook/dinov2-base`
  - `dinov3` -> `facebook/dinov3-vitb16-pretrain-lvd1689m`
  - `siglip2` -> `google/siglip2-base-patch16-256`
- Default training config currently uses `repa_encoder: dinov2`.
- `dinov2` and `siglip2` can be downloaded automatically from Hugging Face on first run.
- `dinov3` is supported by codebase, but you should download its checkpoint manually and use a local path.

For local `dinov3` checkpoint, set in config:

```yaml
model:
  params:
    repa_encoder: ./pretrained/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
```

Optional manual pre-cache for HF-based backbones (`dinov2`, `siglip2`):

```bash
python - <<'PY'
from transformers import AutoImageProcessor, AutoModel
models = [
    "facebook/dinov2-base",
    "google/siglip2-base-patch16-256",
]
for m in models:
    AutoImageProcessor.from_pretrained(m)
    AutoModel.from_pretrained(m)
    print(f"{m} cached")
PY
```

To switch REPA backbone, edit `repa_encoder` in config, e.g.:
- `repa_encoder: dinov3`
- `repa_encoder: siglip2`
- or local file path for dinov3:
  - `repa_encoder: ./pretrained/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`

3. FID stats:
- Default file is already in repo: `fid_stats/adm_in256_stats.npz`.

## Train

Current released training example config:
- [configs/catok_b_256.yaml](configs/catok_b_256.yaml)

Run training on 8 GPUs:

```bash
bash scripts/train_8gpu.sh configs/catok_b_256.yaml
```

Or directly:

```bash
torchrun --nproc-per-node=8 train_net.py --cfg configs/catok_b_256.yaml
```

## Evaluation

Example tokenizer evaluation:

```bash
torchrun --nproc-per-node=8 test_net.py \
  --model ./output/catok_b_256 \
  --step 250000 \
  --cfg configs/catok_b_256.yaml \
  --cfg_value 1.0 \
  --test_num_slots 256 \
  --test_num_steps 25
```

## Reconstruction Inference

Use [scripts/infer_recon.py](scripts/infer_recon.py) for controllable reconstruction:
- `--cfg`: classifier-free guidance
- `--num-tokens`: number of tokens used for reconstruction
- `--start-token`: token start index

Example:

```bash
python scripts/infer_recon.py \
  --model-dir ./output/catok_b_256 \
  --config ./configs/catok_b_256.yaml \
  --checkpoint ./output/catok_b_256/models/step250000/custom_checkpoint_1.pkl \
  --image /path/to/your/input_image.webp \
  --cfg 2.0 \
  --num-tokens 256 \
  --start-token 0 \
  --sample-steps 25 \
  --output-dir ./infer_outputs
```

## Citation

```bibtex
@inproceedings{catok2026,
  title={CaTok: Taming Mean Flows for One-Dimensional Causal Image Tokenization},
  author={Chen, Yitong and Wu, Zuxuan and Qiu, Xipeng and Jiang, Yu-Gang},
  booktitle={CVPR},
  year={2026}
}
```

## Acknowledgement and Note

This codebase is built on [Semanticist](https://github.com/visual-gen/semanticist) and inspired by [MeanFlow](https://github.com/haidog-yaqub/MeanFlow).
Most of the repository refactoring and cleanup work was completed by an agent, so if you notice any issues, feel free to reach out.
