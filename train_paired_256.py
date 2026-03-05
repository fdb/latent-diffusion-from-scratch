import json
import os
from datetime import datetime

import torch
import torchvision
from accelerate import Accelerator
from diffusers import DDIMPipeline, DDIMScheduler, UNet2DModel
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm


def create_output_dirs(base_dir="output"):
    """Create timestamped output directory structure"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_dir = os.path.join(base_dir, f"train_paired_{timestamp}")
    samples_dir = os.path.join(output_dir, "samples")
    checkpoints_dir = os.path.join(output_dir, "checkpoints")

    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    return output_dir, samples_dir, checkpoints_dir


def save_paired_images(
    model, scheduler, source_images, step, samples_dir, device,
    num_inference_steps=50,
):
    """Generate and save sample images during training.

    Takes source (conditioning) images, runs the reverse diffusion process
    conditioned on them, and saves the results as grids.
    """
    model.eval()
    with torch.no_grad():
        # Start from pure noise for the target side
        noise = torch.randn(
            source_images.shape[0], 3,
            source_images.shape[2], source_images.shape[3],
            device=device,
        )

        # Set up scheduler timesteps
        scheduler.set_timesteps(num_inference_steps)

        sample = noise
        for t in scheduler.timesteps:
            # Concatenate source conditioning with current noisy sample
            model_input = torch.cat([sample, source_images], dim=1)
            noise_pred = model(model_input, t, return_dict=False)[0]
            sample = scheduler.step(noise_pred, t, sample, return_dict=False)[0]

        generated = sample

    model.train()

    # Denormalize from [-1, 1] to [0, 1]
    def denorm(x):
        return (x.clamp(-1, 1) + 1) / 2

    # Build grid: source | generated
    rows = []
    for i in range(generated.shape[0]):
        row = torch.cat([denorm(source_images[i].cpu()), denorm(generated[i].cpu())], dim=2)
        rows.append(row)

    grid = torchvision.utils.make_grid(rows, nrow=1, padding=2)
    grid_image = transforms.ToPILImage()(grid)
    grid_image.save(os.path.join(samples_dir, f"grid_step_{step:06d}.png"))

    # Also save just the generated images
    gen_grid = torchvision.utils.make_grid(
        [denorm(generated[i].cpu()) for i in range(generated.shape[0])], nrow=2
    )
    gen_image = transforms.ToPILImage()(gen_grid)
    gen_image.save(os.path.join(samples_dir, f"generated_step_{step:06d}.png"))


class PairedImageDataset(Dataset):
    """Dataset for paired images (target on left, source/condition on right)."""

    def __init__(self, image_dir, image_size=256):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.image_size = image_size

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(image_path).convert("RGB")

        w, h = image.size
        # Target (real image) is on the LEFT, source (skeleton) is on the RIGHT
        target_image = image.crop((0, 0, w // 2, h))
        source_image = image.crop((w // 2, 0, w, h))

        # Resize both to image_size x image_size
        target_image = target_image.resize(
            (self.image_size, self.image_size), Image.BICUBIC
        )
        source_image = source_image.resize(
            (self.image_size, self.image_size), Image.BICUBIC
        )

        target_image = self.transform(target_image)
        source_image = self.transform(source_image)

        return source_image, target_image


def unwrap_model(model):
    """Unwrap a model from DDP/compile wrappers.

    Works around a bug in some accelerate versions where
    unwrap_model fails with KeyError on '_orig_mod'.
    """
    # Strip DDP / FSDP wrapper
    if hasattr(model, "module"):
        model = model.module
    # Strip torch.compile wrapper
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return model


def train_paired_diffusion(
    train_dir="datasets/research-week-2025",
    base_output_dir="output",
    resume_from=None,
    image_size=256,
    train_batch_size=8,
    num_epochs=100,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    save_image_steps=500,
    save_model_epochs=1,
    num_train_timesteps=1000,
    num_inference_steps=50,
    mixed_precision="fp16",
    seed=42,
):
    # Create output directories
    if resume_from:
        resume_from = resume_from.rstrip("/")
        if not os.path.isdir(resume_from):
            raise ValueError(f"Checkpoint directory {resume_from} not found")

        output_dir = os.path.dirname(os.path.dirname(resume_from))
        samples_dir = os.path.join(output_dir, "samples")
        checkpoints_dir = os.path.join(output_dir, "checkpoints")
        print(f"Resuming training in: {output_dir}")

        config_path = os.path.join(output_dir, "config.json")
        if not os.path.exists(config_path):
            raise ValueError(f"Config file not found at {config_path}")

        with open(config_path, "r") as f:
            saved_config = json.load(f)

        last_checkpoint = os.path.basename(resume_from)
        if last_checkpoint == "checkpoint-final":
            start_epoch = num_epochs - 1
        else:
            try:
                epoch_str = last_checkpoint.split("-")[1]
                start_epoch = int(
                    epoch_str.split("/")[0] if "/" in epoch_str else epoch_str
                )
            except (IndexError, ValueError) as e:
                raise ValueError(
                    f"Invalid checkpoint directory format: {last_checkpoint}"
                ) from e
    else:
        output_dir, samples_dir, checkpoints_dir = create_output_dirs(base_output_dir)
        print(f"Output directory: {output_dir}")
        start_epoch = 0
        saved_config = None

    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )

    torch.manual_seed(seed)

    # Performance: enable cuDNN benchmarking (input size is fixed at 256x256)
    torch.backends.cudnn.benchmark = True
    # Performance: allow TF32 on Ampere+ GPUs for faster matmuls/convolutions
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Initialize UNet model with 6 input channels (3 noisy target + 3 source condition)
    model = UNet2DModel(
        sample_size=image_size,
        in_channels=6,   # 3 channels noisy target + 3 channels source conditioning
        out_channels=3,  # predict noise for the target only
        layers_per_block=2,
        block_out_channels=(128, 256, 384, 512, 768, 1024),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )

    # Performance: use channels_last memory format for Tensor Core optimization
    model = model.to(memory_format=torch.channels_last)

    # Load checkpoint if resuming (must happen before torch.compile)
    if resume_from:
        checkpoint_path = os.path.join(resume_from, "unet_state_dict.pt")
        if os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            print(f"Loaded model weights from: {checkpoint_path}")
        else:
            # Try loading from diffusers format
            from diffusers import UNet2DModel as LoadUNet
            loaded_unet = LoadUNet.from_pretrained(resume_from)
            model.load_state_dict(loaded_unet.state_dict())
            print(f"Loaded model weights from: {resume_from}")

    # Performance: compile the model for fused CUDA kernels (PyTorch 2.0+)
    model = torch.compile(model)

    # Initialize noise scheduler
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
        clip_sample=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Create dataset and dataloader
    dataset = PairedImageDataset(train_dir, image_size)
    train_dataloader = DataLoader(
        dataset, batch_size=train_batch_size, shuffle=True,
        num_workers=4, pin_memory=False, persistent_workers=True,
    )

    # Keep a fixed batch of source images for consistent visualization
    fixed_source_batch = None

    # Save training config
    config = {
        "type": "paired_conditional_diffusion",
        "train_dir": train_dir,
        "image_size": image_size,
        "train_batch_size": train_batch_size,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "mixed_precision": mixed_precision,
        "seed": seed,
        "num_training_images": len(dataset),
        "num_train_timesteps": num_train_timesteps,
        "num_inference_steps": num_inference_steps,
        "model_in_channels": 6,
        "model_out_channels": 3,
    }

    if resume_from:
        saved_config.update(
            {
                "resumed_from": resume_from,
                "resumed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_epochs": saved_config.get("total_epochs", 0) + num_epochs,
            }
        )
        config = saved_config
    else:
        config["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        config["total_epochs"] = num_epochs

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Prepare with accelerator
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    # Training loop
    global_step = start_epoch * len(train_dataloader) if resume_from else 0
    print(f"Starting training from epoch {start_epoch} for {num_epochs} epochs...")
    print(f"Number of training steps per epoch: {len(train_dataloader)}")
    print(f"Dataset size: {len(dataset)} paired images")

    for epoch in range(start_epoch, num_epochs):
        model.train()

        progress_bar = tqdm(
            total=len(train_dataloader),
            disable=not accelerator.is_local_main_process,
            desc=f"Epoch {epoch}/{num_epochs - 1}",
        )

        running_loss = 0.0

        for source_images, target_images in train_dataloader:
            source_images = source_images.to(memory_format=torch.channels_last)
            target_images = target_images.to(memory_format=torch.channels_last)

            # Capture fixed source images for visualization
            if fixed_source_batch is None:
                fixed_source_batch = source_images[:4].clone()

            # Sample noise to add to target images
            noise = torch.randn_like(target_images)
            bs = target_images.shape[0]

            # Sample random timesteps
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bs,),
                device=target_images.device,
            ).long()

            # Add noise to the target images
            noisy_targets = noise_scheduler.add_noise(target_images, noise, timesteps)

            # Concatenate noisy target with source conditioning
            model_input = torch.cat([noisy_targets, source_images], dim=1)

            with accelerator.accumulate(model):
                # Predict the noise
                noise_pred = model(model_input, timesteps, return_dict=False)[0]
                loss = torch.nn.functional.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.detach().item()

            progress_bar.update(1)
            progress_bar.set_postfix(
                {"loss": f"{running_loss / (progress_bar.n + 1):.4f}"}
            )

            global_step += 1

            # Generate and save sample images
            if global_step % save_image_steps == 0 and accelerator.is_main_process:
                print(f"\nGenerating sample images at step {global_step}...")
                unwrapped_model = unwrap_model(model)
                save_paired_images(
                    unwrapped_model,
                    noise_scheduler,
                    fixed_source_batch.to(accelerator.device),
                    global_step,
                    samples_dir,
                    accelerator.device,
                    num_inference_steps=num_inference_steps,
                )
                print(f"Samples saved to {samples_dir}")

        progress_bar.close()

        # Save model checkpoint
        if epoch % save_model_epochs == 0 and accelerator.is_main_process:
            unwrapped_model = unwrap_model(model)
            checkpoint_path = os.path.join(checkpoints_dir, f"checkpoint-{epoch:04d}")
            os.makedirs(checkpoint_path, exist_ok=True)

            # Save UNet state dict (since DDIMPipeline doesn't support 6-channel UNet)
            torch.save(
                unwrapped_model.state_dict(),
                os.path.join(checkpoint_path, "unet_state_dict.pt"),
            )
            # Also save the scheduler config
            noise_scheduler.save_config(checkpoint_path)
            # Save model config for reconstruction
            unwrapped_model.save_config(checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

    # Save final model
    if accelerator.is_main_process:
        unwrapped_model = unwrap_model(model)
        final_path = os.path.join(checkpoints_dir, "checkpoint-final")
        os.makedirs(final_path, exist_ok=True)
        torch.save(
            unwrapped_model.state_dict(),
            os.path.join(final_path, "unet_state_dict.pt"),
        )
        noise_scheduler.save_config(final_path)
        unwrapped_model.save_config(final_path)
        print(f"Saved final model: {final_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume_from",
        type=str,
        help="Path to checkpoint directory to resume from",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="Number of epochs to train",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate",
    )
    args = parser.parse_args()

    train_paired_diffusion(
        resume_from=args.resume_from,
        num_epochs=args.num_epochs,
        train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
