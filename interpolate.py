import torch
from diffusers import DDIMPipeline
from PIL import Image
from tqdm.auto import tqdm
import os
import argparse

def generate_with_cfg(unet, scheduler, noise, timesteps, guidance_scale):
    # Run twice - with and without guidance
    sample = noise

    for t in timesteps:
        # Double the sample for with/without guidance
        latent_model_input = torch.cat([sample] * 2, dim=0)

        with torch.no_grad():
            # Get both predictions
            noise_pred_uncond, noise_pred_cond = unet(
                latent_model_input,
                t,
                return_dict=False
            )[0].chunk(2)

            # Apply guidance
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # Scheduler step
            sample = scheduler.step(noise_pred, t, sample).prev_sample

    return sample

def interpolate(pipeline, noise_1_seed, noise_2_seed, steps, inference_steps, output_dir,
                min_guidance=1.0, max_guidance=7.5):
    scheduler = pipeline.scheduler
    unet = pipeline.unet

    # Set up scheduler timesteps
    scheduler.set_timesteps(inference_steps)

    # Generate two noise patterns
    torch.manual_seed(noise_1_seed)
    noise_1 = torch.randn(1, 3, 512, 512).to('cuda')
    torch.manual_seed(noise_2_seed)
    noise_2 = torch.randn(1, 3, 512, 512).to('cuda')

    for i in tqdm(range(steps)):
        t_scale = i / (steps - 1)

        # Interpolate noise
        interpolated_noise = noise_1 * (1 - t_scale) + noise_2 * t_scale

        # Interpolate guidance scale
        guidance_scale = min_guidance * (1 - t_scale) + max_guidance * t_scale

        # Generate image with current guidance scale
        sample = generate_with_cfg(
            unet,
            scheduler,
            interpolated_noise,
            scheduler.timesteps,
            guidance_scale
        )

        # Save image
        image = (sample / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        image = Image.fromarray((image * 255).round().astype("uint8"))
        image.save(f'{output_dir}/frame-{i:04d}.png')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Interpolate between two noise samples with CFG')
    parser.add_argument('checkpoint', type=str, help='Path to the checkpoint')
    parser.add_argument('--n1', type=int, help='Seed of the first noise sample', default=1234)
    parser.add_argument('--n2', type=int, help='Seed of the second noise sample', default=5678)
    parser.add_argument('--steps', type=int, help='Number of steps to interpolate', default=30)
    parser.add_argument('--inference_steps', type=int, help='Number of inference steps', default=50)
    parser.add_argument('--min_guidance', type=float, help='Minimum guidance scale', default=1.0)
    parser.add_argument('--max_guidance', type=float, help='Maximum guidance scale', default=7.5)
    parser.add_argument('--output', type=str, help='Output directory', default='output')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    pipeline = DDIMPipeline.from_pretrained(args.checkpoint).to('cuda')

    interpolate(
        pipeline,
        args.n1,
        args.n2,
        args.steps,
        args.inference_steps,
        args.output,
        args.min_guidance,
        args.max_guidance
    )
