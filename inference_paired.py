"""Run paired conditional diffusion inference on a single image.

Usage:
    uv run python inference_paired.py --checkpoint output/.../checkpoints/checkpoint-0000 --input example-pose.png
    uv run python inference_paired.py --checkpoint output/.../checkpoints/checkpoint-0000 --input example-pose.png --output result.png --steps 50
"""

import argparse

import torch
from diffusers import DDIMScheduler
from PIL import Image
from torchvision import transforms

from train_paired_256 import create_model


def load_model(checkpoint_dir, device):
    import os

    model = create_model(image_size=256)

    state_dict_path = os.path.join(checkpoint_dir, "unet_state_dict.pt")
    state_dict = torch.load(state_dict_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def generate(model, scheduler, source_image, num_steps=20, seed=42, device="cpu"):
    """Run DDIM sampling conditioned on a source image.

    Args:
        model: The 6-channel UNet.
        scheduler: DDIMScheduler instance.
        source_image: Tensor of shape [1, 3, 256, 256] in [-1, 1].
        num_steps: Number of DDIM denoising steps.
        seed: Random seed for initial noise.
        device: Torch device.

    Returns:
        Generated image tensor of shape [1, 3, 256, 256] in [-1, 1].
    """
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(
        1, 3, 256, 256,
        device=device,
        generator=generator,
    )

    scheduler.set_timesteps(num_steps)
    sample = noise

    for t in scheduler.timesteps:
        model_input = torch.cat([sample, source_image], dim=1)
        noise_pred = model(model_input, t, return_dict=False)[0]
        sample = scheduler.step(noise_pred, t, sample, return_dict=False)[0]

    return sample


def main():
    parser = argparse.ArgumentParser(description="Paired diffusion inference")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to checkpoint dir with unet_state_dict.pt",
    )
    parser.add_argument("--input", type=str, required=True, help="Input conditioning image")
    parser.add_argument("--output", type=str, default="out.png", help="Output image path")
    parser.add_argument("--steps", type=int, default=20, help="Number of DDIM steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    model = load_model(args.checkpoint, device)
    print(f"Loaded model from {args.checkpoint}")

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

    print(f"Running DDIM sampling with {args.steps} steps (seed={args.seed})...")
    result = generate(model, scheduler, source_tensor, args.steps, args.seed, device)

    # Denormalize [-1, 1] -> [0, 1] and save
    result = (result.clamp(-1, 1) + 1) / 2
    result_pil = transforms.ToPILImage()(result.squeeze(0).cpu())
    result_pil.save(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
