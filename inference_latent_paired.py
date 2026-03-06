"""Run latent-space paired conditional diffusion inference on a single image.

Uses a pretrained SD 1.5 VAE for encoding/decoding and a trained latent UNet
for the DDIM denoising loop at 32x32x4 resolution.

Usage:
    uv run python inference_latent_paired.py --checkpoint output/.../checkpoints/checkpoint-0004 --input example-pose.png
    uv run python inference_latent_paired.py --checkpoint output/.../checkpoints/checkpoint-0004 --input example-pose.png --output result.png --steps 20
"""

import argparse
import os

import torch
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DModel
from PIL import Image
from torchvision import transforms

VAE_MODEL_ID = "stabilityai/sd-vae-ft-mse"
VAE_SCALING_FACTOR = 0.18215


def load_model(checkpoint_dir, device):
    model = UNet2DModel.from_config(checkpoint_dir)

    state_dict_path = os.path.join(checkpoint_dir, "unet_state_dict.pt")
    state_dict = torch.load(state_dict_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def generate(model, scheduler, vae, source_image, num_steps=20, seed=42, device="cpu"):
    """Run DDIM sampling in latent space conditioned on a source image.

    Args:
        model: The 8-channel latent UNet.
        scheduler: DDIMScheduler instance.
        vae: Pretrained VAE for encoding/decoding.
        source_image: Tensor of shape [1, 3, 256, 256] in [-1, 1].
        num_steps: Number of DDIM denoising steps.
        seed: Random seed for initial noise.
        device: Torch device.

    Returns:
        Generated image tensor of shape [1, 3, 256, 256] in [-1, 1].
    """
    # Encode conditioning image to latent space
    source_latent = vae.encode(source_image).latent_dist.mean * VAE_SCALING_FACTOR

    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(
        1, 4, 32, 32,
        device=device,
        generator=generator,
    )

    scheduler.set_timesteps(num_steps)
    sample = noise

    for t in scheduler.timesteps:
        model_input = torch.cat([sample, source_latent], dim=1)
        noise_pred = model(model_input, t, return_dict=False)[0]
        sample = scheduler.step(noise_pred, t, sample, return_dict=False)[0]

    # Decode latent back to pixel space
    decoded = vae.decode(sample / VAE_SCALING_FACTOR).sample
    return decoded


def main():
    parser = argparse.ArgumentParser(description="Latent paired diffusion inference")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to checkpoint dir with unet_state_dict.pt",
    )
    parser.add_argument("--input", type=str, required=True, help="Input conditioning image")
    parser.add_argument("--output", type=str, default="out.png", help="Output image path")
    parser.add_argument("--steps", type=int, default=20, help="Number of DDIM steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load VAE
    print(f"Loading VAE from {VAE_MODEL_ID}...")
    vae = AutoencoderKL.from_pretrained(VAE_MODEL_ID).to(device)
    vae.eval()

    # Load UNet
    model = load_model(args.checkpoint, device)
    print(f"Loaded UNet from {args.checkpoint}")

    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
        clip_sample=True,
    )

    transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    source_pil = Image.open(args.input).convert("RGB")
    source_tensor = transform(source_pil).unsqueeze(0).to(device)
    print(f"Input image: {args.input} ({source_pil.size[0]}x{source_pil.size[1]} -> 256x256)")

    print(f"Running latent DDIM sampling with {args.steps} steps (seed={args.seed})...")
    result = generate(model, scheduler, vae, source_tensor, args.steps, args.seed, device)

    # Denormalize [-1, 1] -> [0, 1] and save
    result = (result.clamp(-1, 1) + 1) / 2
    result_pil = transforms.ToPILImage()(result.squeeze(0).cpu())
    result_pil.save(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
