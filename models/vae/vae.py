import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels)
        )
    def forward(self, x):
        return x + self.block(x)

class Encoder(nn.Module):
    def __init__(self, in_channels=2, latent_channels=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            ResBlock(64),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # 256x256
            ResBlock(128),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), # 128x128
            ResBlock(256),
            nn.Conv2d(256, 256, 4, stride=2, padding=1), # 64x64
            ResBlock(256),
            nn.Conv2d(256, latent_channels * 2, 3, padding=1)
        )

    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, latent_channels=4, out_channels=2):
        super().__init__()
        # 64 -> 128 -> 256 -> 512
        self.net = nn.Sequential(
            nn.Conv2d(latent_channels, 256, 3, padding=1),
            ResBlock(256),
            nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1), # 128x128
            ResBlock(256),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), # 256x256
            ResBlock(128),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), # 512x512
            ResBlock(64),
            nn.Conv2d(64, out_channels, 3, padding=1)
        )

    def forward(self, x):
        return self.net(x)

class AcousticVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(in_channels=2, latent_channels=4)
        self.decoder = Decoder(latent_channels=4, out_channels=2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, sample=True):
        # x shape: (B, 2, 512, 512)
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)

        logvar = torch.clamp(logvar, min=-20.0, max=10.0)
        mu = torch.clamp(mu, min=-100.0, max=100.0)

        if sample:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu
        recon_x = self.decoder(z)
        
        return recon_x, mu, logvar