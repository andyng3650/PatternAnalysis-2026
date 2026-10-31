"""
modules.py - Model components for the HipMRI generation project (COMP3710, task 2.4).

v1: shared building blocks (residual block, encoder, decoder) and the baseline
autoencoder. The baseline uses the same encoder and decoder as the VQ models
but has NO codebook, so comparing them isolates what quantisation adds.

Architecture follows the VQ-VAE paper (van den Oord et al., 2017,
arXiv:1711.00937): strided convolutions to downsample, residual blocks
(ReLU, 3x3 conv, ReLU, 1x1 conv, skip connection), transposed convolutions
to upsample.

Shapes for a (B, 1, 256, 128) batch with downsample=4:
    encoder: (B, 1, 256, 128) -> (B, hidden, 64, 32)
    decoder: (B, hidden, 64, 32) -> (B, 1, 256, 128)
"""
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """x + conv1x1(relu(conv3x3(relu(x)))). Keeps the shape unchanged."""

    def __init__(self, channels, res_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, res_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(res_channels, channels, kernel_size=1),
        )

    def forward(self, x):
        return x + self.block(x)


class Encoder(nn.Module):
    """Downsample by `downsample` (2 or 4) with stride-2 convs, then residual blocks."""

    def __init__(self, in_channels, hidden, res_channels=32, n_res=2, downsample=4):
        super().__init__()
        if downsample == 4:
            layers = [
                nn.Conv2d(in_channels, hidden // 2, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden // 2, hidden, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
            ]
        elif downsample == 2:
            layers = [
                nn.Conv2d(in_channels, hidden, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
            ]
        else:
            raise ValueError("downsample must be 2 or 4")
        layers.append(nn.Conv2d(hidden, hidden, kernel_size=3, padding=1))
        layers += [ResidualBlock(hidden, res_channels) for _ in range(n_res)]
        layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    """Residual blocks, then upsample by `upsample` (2 or 4) with transposed convs."""

    def __init__(self, in_channels, out_channels, hidden, res_channels=32, n_res=2,
                 upsample=4):
        super().__init__()
        layers = [nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1)]
        layers += [ResidualBlock(hidden, res_channels) for _ in range(n_res)]
        layers.append(nn.ReLU())
        if upsample == 4:
            layers += [
                nn.ConvTranspose2d(hidden, hidden // 2, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(hidden // 2, out_channels, kernel_size=4, stride=2, padding=1),
            ]
        elif upsample == 2:
            layers.append(
                nn.ConvTranspose2d(hidden, out_channels, kernel_size=4, stride=2, padding=1))
        else:
            raise ValueError("upsample must be 2 or 4")
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class BaselineAE(nn.Module):
    """Plain autoencoder: encoder -> 1x1 conv to a latent of `embed_dim` channels -> decoder.

    Same encoder/decoder as the VQ models, but the latent is continuous
    (no codebook), so it can only reconstruct, not generate.
    """

    def __init__(self, hidden=128, embed_dim=64, res_channels=32, n_res=2):
        super().__init__()
        self.encoder = Encoder(1, hidden, res_channels, n_res, downsample=4)
        self.to_latent = nn.Conv2d(hidden, embed_dim, kernel_size=1)
        self.decoder = Decoder(embed_dim, 1, hidden, res_channels, n_res, upsample=4)

    def forward(self, x):
        z = self.to_latent(self.encoder(x))
        return torch.sigmoid(self.decoder(z))  # outputs in [0, 1], like the inputs


if __name__ == "__main__":
    # Shape test: a fake batch must come back with exactly the input shape
    x = torch.rand(8, 1, 256, 128)
    model = BaselineAE()
    z = model.to_latent(model.encoder(x))
    out = model(x)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"input {tuple(x.shape)} -> latent {tuple(z.shape)} -> output {tuple(out.shape)}")
    print(f"output range {out.min():.3f} to {out.max():.3f}, parameters: {n_params:,}")
    assert out.shape == x.shape, "output shape must match input shape"
    print("Shape test passed.")