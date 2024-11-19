import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import DDIMPipeline, DDIMInverseScheduler
from PIL import Image
import os
# from accelerate import Accelerator
from tqdm.auto import tqdm
import torchvision
from datetime import datetime
import argparse

def generate_initial_images(pipeline, seed1, seed2, inference_steps, output_dir):
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

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    image1_tensor = transform(image1).unsqueeze(0).to('cuda')
    image2_tensor = transform(image2).unsqueeze(0).to('cuda')

    return image1_tensor, image2_tensor


def interpolate(pipeline, image1_tensor, image2_tensor, steps, inference_steps, output_dir):
    # Get components
    unet = pipeline.unet

    # Create inverse scheduler
    inverse_scheduler = DDIMInverseScheduler.from_config(pipeline.scheduler.config)
    inverse_scheduler.set_timesteps(inference_steps)

    # Get forward scheduler for generation
    forward_scheduler = pipeline.scheduler
    forward_scheduler.set_timesteps(inference_steps)

    # Get latent representations through inverse diffusion
    print("Getting latent for image 1...")
    sample = image1_tensor
    latent1 = sample
    for t in tqdm(inverse_scheduler.timesteps):
        with torch.no_grad():
            noise_pred = unet(sample, t, return_dict=False)[0]
            sample = inverse_scheduler.step(noise_pred, t, sample).prev_sample
    latent1 = sample

    print("Getting latent for image 2...")
    sample = image2_tensor
    latent2 = sample
    for t in tqdm(inverse_scheduler.timesteps):
        with torch.no_grad():
            noise_pred = unet(sample, t, return_dict=False)[0]
            sample = inverse_scheduler.step(noise_pred, t, sample).prev_sample
    latent2 = sample

    # Interpolate between latents and generate frames
    for i in tqdm(range(steps)):
        t_scale = i / (steps - 1)

        # Interpolate in latent space
        interpolated_latent = latent1 * (1 - t_scale) + latent2 * t_scale

        # Generate image from interpolated latent
        sample = interpolated_latent
        for t in forward_scheduler.timesteps:
            with torch.no_grad():
                noise_pred = unet(sample, t, return_dict=False)[0]
                sample = forward_scheduler.step(noise_pred, t, sample).prev_sample

        # Save image
        image = (sample / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        image = Image.fromarray((image * 255).round().astype("uint8"))
        image.save(f'{output_dir}/frame-{i:04d}.png')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Interpolate between two noise samples')
    parser.add_argument('checkpoint', type=str, help='Path to the checkpoint')
    parser.add_argument('--n1', type=int, help='Seed of the first noise sample', default=1234)
    parser.add_argument('--n2', type=int, help='Seed of the second noise sample', default=5678)
    parser.add_argument('--steps', type=int, help='Number of steps to interpolate', default=10)
    parser.add_argument('--inference_steps', type=int, help='Number of inference steps', default=25)
    parser.add_argument('--output', type=str, help='Output directory', default='output')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    pipeline = DDIMPipeline.from_pretrained(args.checkpoint).to('cuda')
    image1_tensor, image2_tensor = generate_initial_images(pipeline, args.n1, args.n2, args.inference_steps, args.output)
    interpolate(pipeline, image1_tensor, image2_tensor, args.steps, args.inference_steps, args.output)
