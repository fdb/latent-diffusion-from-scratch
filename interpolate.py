import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import UNet2DModel, DDIMScheduler, DDIMPipeline
from PIL import Image
import os
# from accelerate import Accelerator
from tqdm.auto import tqdm
import torchvision
from datetime import datetime
import argparse

def interpolate(pipeline, noise_1_seed, noise_2_seed, steps, inference_steps, output_dir):
    torch.manual_seed(noise_1_seed)
    noise_1 = torch.randn(1, 3, 64, 64)
    torch.manual_seed(noise_2_seed)
    noise_2 = torch.randn(1, 3, 64, 64)

    for i in tqdm(range(steps)):
        t = i / (steps - 1)
        interpolated_noise = noise_1 * (1 - t) + noise_2 * t

        with torch.no_grad():
            image = pipeline(num_inference_steps=inference_steps).images[0]
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
