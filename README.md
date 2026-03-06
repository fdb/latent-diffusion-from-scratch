# Latent Diffusion Experiments

Paired conditional diffusion models that generate images from pose skeleton inputs. Includes both pixel-space (256x256) and latent-space (32x32x4 via SD 1.5 VAE) variants.

## Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first. All commands use `uv run` — no need to activate a virtualenv.

## Latent-Space Paired Diffusion (Recommended)

Uses a pretrained Stable Diffusion 1.5 VAE to compress images to 32x32x4 latent space before training. The UNet operates on ~48x fewer values than the pixel-space version, dramatically speeding up training and inference.

### 1. Training

Training images should be paired JPGs (target on left, source/skeleton on right) in a single directory.

```bash
# Basic training
uv run python train_latent_paired.py --train_dir datasets/research-week-2025

# With custom settings
uv run python train_latent_paired.py \
  --train_dir datasets/research-week-2025 \
  --num_epochs 50 \
  --batch_size 8 \
  --learning_rate 1e-4

# Resume from checkpoint
uv run python train_latent_paired.py \
  --resume_from output/train_latent_paired_.../checkpoints/checkpoint-0010

# Force re-encode images through VAE (e.g. after changing dataset)
uv run python train_latent_paired.py --recache
```

On first run, all images are encoded through the frozen VAE and cached to `_latent_cache.pt` in the dataset directory. Subsequent runs load from cache instantly.

### 2. Inference

```bash
uv run python inference_latent_paired.py \
  --checkpoint output/train_latent_paired_.../checkpoints/checkpoint-0010 \
  --input example-pose.png \
  --output result.png \
  --steps 20
```

### 3. ONNX Export

Exports three ONNX models for deployment (e.g. in Figment):

```bash
uv run python export_latent_onnx.py \
  --checkpoint_dir output/train_latent_paired_.../checkpoints/checkpoint-0010

# Optional: also export fp16 versions
uv run python export_latent_onnx.py \
  --checkpoint_dir output/train_latent_paired_.../checkpoints/checkpoint-0010 \
  --fp16
```

This produces:
- `vae_encoder.onnx` — encodes 256x256 RGB to 32x32x4 latent
- `unet.onnx` — 8-channel latent UNet
- `vae_decoder.onnx` — decodes 32x32x4 latent back to 256x256 RGB

The VAE scaling factor (0.18215) is baked into the encoder/decoder ONNX models.

### 4. Figment Node

Open `latent-paired-diffusion.fgmt` in [Figment](https://figmentapp.com) and configure the three ONNX model paths. The node runs VAE encoding, DDIM denoising, and VAE decoding entirely on the GPU via WebGPU.

## Pixel-Space Paired Diffusion (Legacy)

The original pixel-space variant operates at 256x256x3 with a 6-channel UNet.

```bash
# Training
uv run python train_paired_256.py --num_epochs 50 --batch_size 4

# Inference
uv run python inference_paired.py \
  --checkpoint output/train_paired_.../checkpoints/checkpoint-0010 \
  --input example-pose.png

# ONNX export (single UNet model)
uv run python export_unet_onnx.py \
  --checkpoint_dir output/train_paired_.../checkpoints/checkpoint-0010
```

Figment node: `paired-diffusion.fgmt`
