from diffusers import AutoencoderKL, DDIMPipeline
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from tqdm.auto import tqdm
import argparse

def generate_and_encode_images(pipeline, vae, seed1, seed2, inference_steps, output_dir):
    # Generate first image
    torch.manual_seed(seed1)
    with torch.no_grad():
        image1 = pipeline(num_inference_steps=inference_steps).images[0]
    image1.save(f'{output_dir}/image1.png')

    # Generate second image
    torch.manual_seed(seed2)
    with torch.no_grad():
        image2 = pipeline(num_inference_steps=inference_steps).images[0]
    image2.save(f'{output_dir}/image2.png')

    # Convert to tensors
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    image1_tensor = transform(image1).unsqueeze(0).to('cuda')
    image2_tensor = transform(image2).unsqueeze(0).to('cuda')

    # Encode images to latent space using VAE
    with torch.no_grad():
        latent1 = vae.encode(image1_tensor).latent_dist.sample()
        latent2 = vae.encode(image2_tensor).latent_dist.sample()

    return latent1, latent2

def interpolate(pipeline, vae, latent1, latent2, steps, inference_steps, output_dir):
    # Interpolate in VAE latent space and decode
    for i in tqdm(range(steps)):
        t_scale = i / (steps - 1)

        # Interpolate in VAE latent space
        interpolated_latent = latent1 * (1 - t_scale) + latent2 * t_scale

        # Decode interpolated latent
        with torch.no_grad():
            image = vae.decode(interpolated_latent).sample

        # Save image
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        image = Image.fromarray((image * 255).round().astype("uint8"))
        image.save(f'{output_dir}/frame-{i:04d}.png')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Interpolate between two images using VAE')
    parser.add_argument('checkpoint', type=str, help='Path to the checkpoint')
    parser.add_argument('--n1', type=int, help='Seed of the first image', default=1234)
    parser.add_argument('--n2', type=int, help='Seed of the second image', default=5678)
    parser.add_argument('--steps', type=int, help='Number of steps to interpolate', default=30)
    parser.add_argument('--inference_steps', type=int, help='Number of inference steps', default=50)
    parser.add_argument('--output', type=str, help='Output directory', default='output')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load pipeline and VAE
    pipeline = DDIMPipeline.from_pretrained(args.checkpoint).to('cuda')
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to('cuda')

    # Generate images and get their VAE latents
    latent1, latent2 = generate_and_encode_images(
        pipeline, vae, args.n1, args.n2, args.inference_steps, args.output
    )

    # Interpolate between the latents and generate frames
    interpolate(
        pipeline, vae, latent1, latent2,
        args.steps, args.inference_steps, args.output
    )
