import torch
from torch import nn


class UniversalAutoencoder(nn.Module):
    """Compressed spatial features by default; legacy vectors remain loadable."""
    def __init__(self, base_channels=32, bottleneck_dim=8192, dropout=0., architecture='spatial_v2'):
        super().__init__()
        if architecture not in ['vector_v1', 'spatial_v2']:
            raise ValueError(f'Unknown architecture: {architecture}')
        if not 0 < bottleneck_dim < 3*128*128:
            raise ValueError('The latent must be smaller than the RGB input')
        self.architecture = architecture
        channels = [base_channels * k for k in ([1, 2, 4, 8] if architecture == 'vector_v1' else [1, 2, 4])]
        layers, previous = [], 3
        for c in channels:
            layers += [nn.Conv2d(previous, c, 4, 2, 1), nn.ReLU(), nn.Dropout2d(dropout)]
            previous = c
        self.encoder = nn.Sequential(*layers)
        if architecture == 'vector_v1':
            self.compress = nn.Linear(channels[-1]*8*8, bottleneck_dim)
            self.expand = nn.Linear(bottleneck_dim, channels[-1]*8*8)
        else:
            if bottleneck_dim % (16*16):
                raise ValueError('Spatial latent size must be divisible by 16*16')
            latent_channels = bottleneck_dim // (16*16)
            self.compress = nn.Conv2d(channels[-1], latent_channels, 1)
            self.expand = nn.Conv2d(latent_channels, channels[-1], 1)
        self.last_channels = channels[-1]
        layers = []
        for c in list(reversed(channels[:-1])) + [3]:
            layers += [nn.ConvTranspose2d(previous, c, 4, 2, 1), nn.Sigmoid() if c == 3 else nn.ReLU()]
            previous = c
        self.decoder = nn.Sequential(*layers)

    def encode(self, x):
        features = self.encoder(x)
        return self.compress(features.flatten(1) if self.architecture == 'vector_v1' else features)

    def forward(self, x):
        z = self.encode(x)
        x = torch.relu(self.expand(z))
        if self.architecture == 'vector_v1':
            x = x.reshape(-1, self.last_channels, 8, 8)
        return self.decoder(x)
