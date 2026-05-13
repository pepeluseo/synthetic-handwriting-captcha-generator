"""
diffusion.py
------------
Implements a lightweight Conditional U-Net denoiser for MNIST-sized diffusion training.

The model predicts the noise added to an image at a given diffusion timestep.
It is conditioned on:
- diffusion timestep t
- digit class label y
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(timesteps, dim):
    """
    Create sinusoidal timestep embeddings.

    Args:
        timesteps: Tensor of shape (batch_size,)
        dim: Dimension of the embedding

    Returns:
        Tensor of shape (batch_size, dim)
    """

    device = timesteps.device
    half = dim // 2

    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=device) / half
    )

    args = timesteps[:, None].float() * freqs[None]

    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))

    return emb


class ResidualBlock(nn.Module):
    """
    Residual convolutional block with timestep and class-label conditioning.
    """

    def __init__(self, in_channels, out_channels, time_dim, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)

        # Conditioning layers
        self.time_emb = nn.Linear(time_dim, out_channels)
        self.label_emb = nn.Embedding(num_classes, out_channels)

        # Match residual shortcut dimensions when needed
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t_emb, labels):
        """
        Args:
            x: Feature map tensor
            t_emb: Timestep embedding tensor
            labels: Class labels tensor

        Returns:
            Residual feature map
        """

        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)

        # Add timestep and class-label conditioning
        conditioning = (
            self.time_emb(t_emb)[:, :, None, None]
            + self.label_emb(labels)[:, :, None, None]
        )

        h = h + conditioning

        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)

        return h + self.shortcut(x)


class ConditionalUNet(nn.Module):
    """
    Conditional U-Net for diffusion noise prediction.

    Input:
        x: noisy image tensor of shape (batch_size, 1, 28, 28)
        t: timestep tensor of shape (batch_size,)
        labels: class label tensor of shape (batch_size,)

    Output:
        predicted noise tensor of shape (batch_size, 1, 28, 28)
    """

    def __init__(
        self,
        num_classes=10,
        img_channels=1,
        base_channels=64,
        time_dim=128
    ):
        super().__init__()

        self.time_dim = time_dim

        # Process sinusoidal timestep embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        # Downsampling path
        self.down1 = ResidualBlock(
            img_channels,
            base_channels,
            time_dim,
            num_classes
        )

        self.down2 = ResidualBlock(
            base_channels,
            base_channels * 2,
            time_dim,
            num_classes
        )

        self.down3 = ResidualBlock(
            base_channels * 2,
            base_channels * 4,
            time_dim,
            num_classes
        )

        # Bottleneck
        self.mid = ResidualBlock(
            base_channels * 4,
            base_channels * 4,
            time_dim,
            num_classes
        )

        # Upsampling path with skip connections
        self.up3 = ResidualBlock(
            base_channels * 6,
            base_channels * 2,
            time_dim,
            num_classes
        )

        self.up2 = ResidualBlock(
            base_channels * 3,
            base_channels,
            time_dim,
            num_classes
        )

        self.up1 = ResidualBlock(
            base_channels + img_channels,
            base_channels,
            time_dim,
            num_classes
        )

        # Final noise prediction layer
        self.output = nn.Conv2d(
            base_channels,
            img_channels,
            kernel_size=1
        )

    def forward(self, x, t, labels):
        """
        Forward pass through the Conditional U-Net.

        Args:
            x: Noisy image tensor, shape (batch_size, 1, 28, 28)
            t: Diffusion timestep tensor, shape (batch_size,)
            labels: Class label tensor, shape (batch_size,)

        Returns:
            Predicted noise tensor with the same shape as x.
        """

        # Timestep embedding
        t_emb = timestep_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)

        # -----------------------------
        # Down path
        # -----------------------------
        h1 = self.down1(x, t_emb, labels)                         # 28x28
        h2 = self.down2(F.avg_pool2d(h1, kernel_size=2), t_emb, labels)  # 14x14
        h3 = self.down3(F.avg_pool2d(h2, kernel_size=2), t_emb, labels)  # 7x7

        # -----------------------------
        # Bottleneck
        # -----------------------------
        h_mid = self.mid(h3, t_emb, labels)                       # 7x7

        # -----------------------------
        # Up path with skip connections
        # -----------------------------
        h = F.interpolate(h_mid, scale_factor=2, mode="nearest")   # 14x14
        h = torch.cat([h, h2], dim=1)
        h = self.up3(h, t_emb, labels)

        h = F.interpolate(h, scale_factor=2, mode="nearest")       # 28x28
        h = torch.cat([h, h1], dim=1)
        h = self.up2(h, t_emb, labels)

        # Final skip connection with original noisy image
        h = torch.cat([h, x], dim=1)
        h = self.up1(h, t_emb, labels)

        # Predict noise with same shape as input image
        return self.output(h)