import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import DDIMPipeline
from PIL import Image
import os
# from accelerate import Accelerator
from tqdm.auto import tqdm
import torchvision
from datetime import datetime
import argparse

def interpolate(pipeline, seed1, seed2, steps, inference_steps, output_dir):
    scheduler = pipeline.scheduler
    unet = pipeline.unet
    scheduler.set_timesteps(inference_steps)

    # Generate two noise patterns
    torch.manual_seed(seed1)
    noise_1 = torch.randn(1, 3, 512, 512).to('cuda')
    torch.manual_seed(seed2)
    noise_2 = torch.randn(1, 3, 512, 512).to('cuda')

    for i in tqdm(range(steps)):
        t_scale = i / (steps - 1)
        sample = noise_1 * (1 - t_scale) + noise_2 * t_scale

        # Store intermediate denoising states
        denoising_states = []

        # First forward pass to get intermediate states
        for t in scheduler.timesteps:
            with torch.no_grad():
                noise_pred = unet(sample, t, return_dict=False)[0]
                sample = scheduler.step(noise_pred, t, sample).prev_sample
                denoising_states.append(sample.clone())

        # Second pass with interpolated intermediate states
        sample = noise_1
        for idx, t in enumerate(scheduler.timesteps):
            with torch.no_grad():
                noise_pred = unet(sample, t, return_dict=False)[0]
                next_sample = scheduler.step(noise_pred, t, sample).prev_sample
                # Interpolate with the stored state
                sample = next_sample * (1 - t_scale) + denoising_states[idx] * t_scale

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
    interpolate(pipeline, args.n1, args.n2, args.steps, args.inference_steps, args.output)
