import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import UNet2DModel, DDIMScheduler, DDIMPipeline
from PIL import Image
import os
from accelerate import Accelerator
from tqdm.auto import tqdm
import torchvision
from datetime import datetime

def create_output_dirs(base_dir="output"):
    """Create timestamped output directory structure"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_dir = os.path.join(base_dir, f"train_{timestamp}")
    samples_dir = os.path.join(output_dir, "samples")
    checkpoints_dir = os.path.join(output_dir, "checkpoints")

    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    return output_dir, samples_dir, checkpoints_dir

def save_images(pipeline, step, samples_dir, num_images=4, num_inference_steps=25):
    """Generate and save sample images during training"""
    with torch.no_grad():
        images = pipeline(
            batch_size=num_images,
            generator=torch.manual_seed(42),
            num_inference_steps=num_inference_steps,
        ).images

    # Save individual images
    for i, image in enumerate(images):
        image.save(os.path.join(samples_dir, f"sample_step_{step:06d}_image_{i}.png"))

    # Create a grid of images
    grid = torchvision.utils.make_grid([transforms.ToTensor()(img) for img in images], nrow=2)
    grid_image = transforms.ToPILImage()(grid)
    grid_image.save(os.path.join(samples_dir, f"grid_step_{step:06d}.png"))

class CustomImageDataset(Dataset):
    def __init__(self, image_dir, image_size=512):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(image_path).convert('RGB')
        return self.transform(image)

def train_diffusion(
    train_dir="datasets/yes-to-the-dress-pngs",
    base_output_dir="output",
    resume_from=None,       # Directory containing checkpoint to resume from
    image_size=512,
    train_batch_size=2,
    num_epochs=100,
    gradient_accumulation_steps=1,
    learning_rate=1e-4,
    lr_warmup_steps=500,
    save_image_steps=500,    # Save images every N steps
    save_model_epochs=1,    # Save model every N epochs
    num_train_timesteps=1000,  # Number of timesteps for training
    num_inference_steps=25,   # Number of steps for generating samples
    mixed_precision="fp16",
    seed=42,
):
    # Create output directories with timestamp or use existing for resume
    if resume_from:
        # Remove trailing slash if present
        resume_from = resume_from.rstrip('/')

        # The resume_from path should point to a specific checkpoint directory
        if not os.path.isdir(resume_from):
            raise ValueError(f"Checkpoint directory {resume_from} not found")

        # Go up two levels: from 'checkpoint-XXXX' to the root output directory
        output_dir = os.path.dirname(os.path.dirname(resume_from))
        samples_dir = os.path.join(output_dir, "samples")
        checkpoints_dir = os.path.join(output_dir, "checkpoints")
        print(f"Resuming training in: {output_dir}")

        # Load the config from the root directory
        config_path = os.path.join(output_dir, "config.json")
        if not os.path.exists(config_path):
            raise ValueError(f"Config file not found at {config_path}")

        with open(config_path, "r") as f:
            saved_config = json.load(f)

        # Find the last epoch number from the checkpoint name
        last_checkpoint = os.path.basename(resume_from)
        if last_checkpoint == "checkpoint-final":
            start_epoch = num_epochs - 1  # Start from the last epoch
        else:
            try:
                # Handle both formats: checkpoint-N and checkpoint-NNNN
                epoch_str = last_checkpoint.split("-")[1]
                start_epoch = int(epoch_str.split("/")[0] if "/" in epoch_str else epoch_str)
            except (IndexError, ValueError) as e:
                raise ValueError(f"Invalid checkpoint directory format: {last_checkpoint}. Expected 'checkpoint-N' or 'checkpoint-NNNN'") from e
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

    # Set random seed
    torch.manual_seed(seed)

    # Initialize UNet model
    model = UNet2DModel(
        sample_size=image_size,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )

    # Load checkpoint if resuming
    if resume_from:
        pipeline = DDIMPipeline.from_pretrained(resume_from)
        model.load_state_dict(pipeline.unet.state_dict())
        print(f"Loaded model weights from: {resume_from}")

    # Initialize noise scheduler
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
        clip_sample=True,
    )

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Create dataset and dataloader
    dataset = CustomImageDataset(train_dir, image_size)
    train_dataloader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True)

    # Save or update training config
    config = {
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
    }

    if resume_from:
        # Update the existing config with new timestamp
        saved_config.update({
            "resumed_from": resume_from,
            "resumed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_epochs": saved_config.get("total_epochs", 0) + num_epochs,
        })
        config = saved_config
    else:
        config["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        config["total_epochs"] = num_epochs

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Prepare model, optimizer, and dataloader with accelerator
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    # Training loop
    global_step = start_epoch * len(train_dataloader) if resume_from else 0
    print(f"Starting training from epoch {start_epoch} for {num_epochs} epochs...")
    print(f"Number of training steps per epoch: {len(train_dataloader)}")

    for epoch in range(start_epoch, num_epochs):
        model.train()

        # Progress bar for this epoch
        progress_bar = tqdm(
            total=len(train_dataloader),
            disable=not accelerator.is_local_main_process,
            desc=f"Epoch {epoch}/{num_epochs-1}"
        )

        running_loss = 0.0

        for batch in train_dataloader:
            clean_images = batch

            # Sample noise and add to images
            noise = torch.randn_like(clean_images)
            bs = clean_images.shape[0]

            # Sample timesteps
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = torch.nn.functional.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.detach().item()

            # Update progress bar
            progress_bar.update(1)
            progress_bar.set_postfix({"loss": f"{running_loss/(progress_bar.n+1):.4f}"})

            global_step += 1

            # Generate and save sample images based on steps
            if global_step % save_image_steps == 0 and accelerator.is_main_process:
                print(f"\nGenerating sample images at step {global_step}...")
                pipeline = DDIMPipeline(
                    unet=accelerator.unwrap_model(model),
                    scheduler=noise_scheduler,
                )
                save_images(pipeline, global_step, samples_dir, num_inference_steps=num_inference_steps)
                print(f"Samples saved to {samples_dir}")

        progress_bar.close()

        # Save model checkpoints at epoch level
        if epoch % save_model_epochs == 0 and accelerator.is_main_process:
            pipeline = DDIMPipeline(
                unet=accelerator.unwrap_model(model),
                scheduler=noise_scheduler,
            )
            checkpoint_path = os.path.join(checkpoints_dir, f"checkpoint-{epoch:04d}")
            pipeline.save_pretrained(checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

    # Save final model
    if accelerator.is_main_process:
        pipeline = DDIMPipeline(
            unet=accelerator.unwrap_model(model),
            scheduler=noise_scheduler,
        )
        final_path = os.path.join(checkpoints_dir, "checkpoint-final")
        pipeline.save_pretrained(final_path)
        print(f"Saved final model: {final_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_from", type=str, help="Path to checkpoint directory to resume from (e.g., output/train_2024.../checkpoints/checkpoint-005)")
    args = parser.parse_args()

    train_diffusion(resume_from=args.resume_from)
