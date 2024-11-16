import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from PIL import Image
import numpy as np
from glob import glob
from datetime import datetime

# Autoencoder for getting images into latent space
class Autoencoder(nn.Module):
    def __init__(self, latent_dim=4):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, latent_dim, 3, stride=2, padding=1)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

# U-Net for noise prediction
class UNet(nn.Module):
    def __init__(self, in_channels=4, time_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # Downsampling
        self.down1 = self._make_down_block(in_channels, 64)
        self.down2 = self._make_down_block(64, 128)
        self.down3 = self._make_down_block(128, 256)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1)
        )

        # Upsampling
        self.up1 = self._make_up_block(256 + 256, 128)
        self.up2 = self._make_up_block(128 + 128, 64)
        self.up3 = self._make_up_block(64 + 64, 64)

        # Output
        self.final = nn.Conv2d(64, in_channels, 1)

    def _make_down_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def _make_up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

    def forward(self, x, t):
        # Time embedding
        t = self.time_mlp(t.unsqueeze(-1))

        # Downsampling path with skip connections
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)

        # Bottleneck
        bottleneck = self.bottleneck(d3)

        # Upsampling path with skip connections
        u1 = self.up1(torch.cat([bottleneck, d3], dim=1))
        u2 = self.up2(torch.cat([u1, d2], dim=1))
        u3 = self.up3(torch.cat([u2, d1], dim=1))

        return self.final(u3)

class DiffusionTrainer:
    def __init__(self, autoencoder, unet, device='cuda'):
        self.device = device
        self.autoencoder = autoencoder.to(device)
        self.unet = unet.to(device)
        self.num_timesteps = 1000
        self.beta_start = 0.0001
        self.beta_end = 0.02

        # Create noise schedule
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps).to(device)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # Precalculate diffusion parameters
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # Posterior variance calculation (beta_tilde)
        # For t > 0, calculate posterior variance β_t~
        # self.posterior_variance = self.betas * (1. - self.alphas_cumprod[:-1]) / (1. - self.alphas_cumprod[1:])
        # For t = 0, append the minimum beta value
        # self.posterior_variance = torch.cat([self.posterior_variance, torch.tensor([self.beta_end], device=device)])
        self.posterior_variance = self.betas.clone()

        # Add assertions to catch dimension mismatches
        assert self.betas.shape == (self.num_timesteps,), f"Expected betas shape {self.num_timesteps}, got {self.betas.shape}"
        assert self.alphas_cumprod.shape == (self.num_timesteps,), f"Expected alphas_cumprod shape {self.num_timesteps}, got {self.alphas_cumprod.shape}"
        assert self.posterior_variance.shape == (self.num_timesteps,), f"Expected posterior_variance shape {self.num_timesteps}, got {self.posterior_variance.shape}"

    def get_noise_schedule(self, t):
        return self.sqrt_alphas_cumprod[t], self.sqrt_one_minus_alphas_cumprod[t]

    def add_noise(self, x, t):
        alpha_t, sigma_t = self.get_noise_schedule(t)
        epsilon = torch.randn_like(x)
        return alpha_t.view(-1, 1, 1, 1) * x + sigma_t.view(-1, 1, 1, 1) * epsilon, epsilon

    def train_step(self, x, optimizer):
        self.unet.train()
        batch_size = x.shape[0]

        # Encode images to latent space
        with torch.no_grad():
            latents = self.autoencoder.encode(x)

        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)

        # Add noise to latents
        noisy_latents, noise = self.add_noise(latents, t)

        # Predict noise
        noise_pred = self.unet(noisy_latents, t.float() / self.num_timesteps)

        # Calculate loss
        loss = F.mse_loss(noise_pred, noise)

        # Optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    @torch.no_grad()
    def sample(self, batch_size=1, progressive=False):
        """
        Generate samples from noise using the trained model.
        If progressive=True, return intermediate steps for visualization.
        """
        self.unet.eval()

        # Start from random noise
        shape = (batch_size, 4, 64, 64)  # 4 is latent dimension
        img = torch.randn(shape, device=self.device)

        # Store intermediate steps if progressive
        intermediate_images = []
        timesteps_to_save = set(range(0, self.num_timesteps, self.num_timesteps // 10))

        # Reverse diffusion process
        for i in reversed(range(0, self.num_timesteps)):
            # Progress indicator
            if i % 100 == 0:
                print(f'Sampling timestep {i:4d}', end='\r')

            # Get network prediction
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            predicted_noise = self.unet(img, t.float() / self.num_timesteps)

            alpha = self.alphas[i]
            alpha_hat = self.alphas_cumprod[i]
            beta = self.betas[i]

            if i > 0:
                noise = torch.randn_like(img)
            else:
                noise = torch.zeros_like(img)

            img = (1 / torch.sqrt(alpha)) * (img - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise) + torch.sqrt(beta) * noise

            # Save intermediate result
            if progressive and i in timesteps_to_save:
                # Decode to image space
                with torch.no_grad():
                    decoded = self.autoencoder.decode(img)
                intermediate_images.append(decoded)

        # Decode final latent to image
        with torch.no_grad():
            final_img = self.autoencoder.decode(img)

        if progressive:
            return final_img, intermediate_images
        return final_img

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, image_size=256):
        self.image_paths = image_paths
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        img = self.transform(img)
        # Verify image is 256 x 256
        if list(img.size()) != [3, 256, 256]:
            raise ValueError(f"Image {self.image_paths[idx]} has invalid dimensions {img.size()}")
        return img

def save_samples(trainer, epoch, batch_idx, save_dir, num_samples=4):
    """Generate and save samples during training"""
    os.makedirs(save_dir, exist_ok=True)

    # Generate samples with intermediate steps
    final_samples, intermediate_samples = trainer.sample(
        batch_size=num_samples,
        progressive=True
    )

    # Save final samples
    save_image(
        final_samples * 0.5 + 0.5,  # Denormalize
        f"{save_dir}/samples_e{epoch}_b{batch_idx}.png",
        nrow=2
    )

    # Save progressive generation steps
    progressive_grid = make_grid(
        torch.cat(intermediate_samples, dim=0) * 0.5 + 0.5,
        nrow=num_samples
    )
    save_image(
        progressive_grid,
        f"{save_dir}/progressive_e{epoch}_b{batch_idx}.png"
    )

def train(image_paths, num_epochs=100, batch_size=32, device='cuda',
          sample_interval=500, num_samples=4):
    """
    Training function with periodic sample generation
    """
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'output/training_{timestamp}'
    samples_dir = f'{output_dir}/samples'
    checkpoints_dir = f'{output_dir}/checkpoints'
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Initialize models
    autoencoder = Autoencoder()
    unet = UNet()

    print("Training autoencoder...")
    autoencoder = autoencoder.to(device)
    ae_optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=1e-4)

    # Create dataset and dataloader
    dataset = ImageDataset(image_paths)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Train autoencoder for a few epochs
    for epoch in range(10):  # Adjust number of epochs as needed
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(device)
            ae_optimizer.zero_grad()
            reconstructed = autoencoder(batch)
            loss = F.mse_loss(reconstructed, batch)
            loss.backward()
            ae_optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Autoencoder Epoch {epoch+1}, Loss: {avg_loss:.4f}")

        # Save a sample reconstruction
        if (epoch + 1) % 2 == 0:
            with torch.no_grad():
                sample = batch[:4]  # Take first 4 images
                reconstruction = autoencoder(sample)
                comparison = torch.cat([sample, reconstruction])
                save_image(comparison * 0.5 + 0.5, f'{samples_dir}/autoencoder_progress_epoch_{epoch+1}.png', nrow=4)

    trainer = DiffusionTrainer(autoencoder, unet, device)

    # Optimizer
    optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-4)

    # Training loop
    global_step = 0
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            batch = batch.to(device)
            loss = trainer.train_step(batch, optimizer)
            total_loss += loss
            global_step += 1

            # Print progress
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(dataloader)}, "
                      f"Loss: {loss:.4f}")

            # Generate and save samples
            if global_step % sample_interval == 0:
                print("\nGenerating samples...")
                save_samples(trainer, epoch + 1, batch_idx, samples_dir, num_samples)

        # Calculate and print epoch average loss
        avg_loss = total_loss / len(dataloader)
        print(f"\nEpoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'autoencoder_state_dict': autoencoder.state_dict(),
                'unet_state_dict': unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'global_step': global_step,
                'loss': avg_loss
            }
            torch.save(
                checkpoint,
                f'{checkpoints_dir}/checkpoint_epoch_{epoch+1}.pt'
            )

    return autoencoder, unet, trainer

# Load images from directory
image_paths = glob("datasets/yes-to-the-dress/*.jpg")
autoencoder, unet, trainer = train(
    image_paths,
    num_epochs=100,
    batch_size=32,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    sample_interval=500,
    num_samples=4
)
