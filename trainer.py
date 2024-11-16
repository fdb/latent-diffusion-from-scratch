import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from glob import glob

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
            nn.ConvTranspose2d(latent_dim, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 3, stride=2, padding=1),
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

    def get_noise_schedule(self, t):
        return torch.sqrt(self.alphas_cumprod[t]), torch.sqrt(1 - self.alphas_cumprod[t])

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

# Custom dataset class for your images
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
        return self.transform(img)

# Training function
def train(image_paths, num_epochs=100, batch_size=32, device='cuda'):
    # Initialize models
    autoencoder = Autoencoder()
    unet = UNet()
    trainer = DiffusionTrainer(autoencoder, unet, device)

    # Create dataset and dataloader
    dataset = ImageDataset(image_paths)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Optimizer
    optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-4)

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(device)
            loss = trainer.train_step(batch, optimizer)
            total_loss += loss

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'autoencoder_state_dict': autoencoder.state_dict(),
                'unet_state_dict': unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch
            }
            torch.save(checkpoint, f'checkpoint_epoch_{epoch+1}.pt')

    return autoencoder, unet, trainer

# Load images from directory
image_paths = glob("datasets/yes-to-the-dress/*.jpg")
autoencoder, unet, trainer = train(
    image_paths,
    num_epochs=100,
    batch_size=32,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)
