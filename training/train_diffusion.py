"""
train_diffusion.py
------------------
Defines training loop and sampling utilities for the Conditional Diffusion Model.

This module implements:
- A linear beta noise schedule
- Forward diffusion q(x_t | x_0)
- MSE noise-prediction training objective
- Reverse denoising sampling loop from T to 0
"""

import os
import torch
import torch.nn.functional as F
from tqdm import tqdm


def linear_beta_schedule(timesteps):
    """
    Create a linear beta schedule for the diffusion process.

    Args:
        timesteps: Number of diffusion steps.

    Returns:
        Tensor of betas with shape (timesteps,).
    """
    beta_start = 1e-4
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


@torch.no_grad()
def sample_images(
    model,
    device,
    num_samples=16,
    num_classes=10,
    img_size=(1, 28, 28),
    timesteps=200,
    class_labels=None,
):
    """
    Generate images using the reverse diffusion denoising process.

    Args:
        model: Trained ConditionalUNet model.
        device: Torch device.
        num_samples: Number of images to generate.
        num_classes: Number of digit classes.
        img_size: Image shape, default MNIST shape.
        timesteps: Number of reverse diffusion steps.
        class_labels: Optional tensor of labels for class-conditioned sampling.

    Returns:
        imgs: Generated images.
        labels: Class labels used for generation.
    """

    model.eval()

    betas = linear_beta_schedule(timesteps).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    # Start from pure Gaussian noise
    imgs = torch.randn(num_samples, *img_size, device=device)

    # If no labels are provided, cycle through class labels 0-9
    if class_labels is None:
        labels = torch.tensor(
            [i % num_classes for i in range(num_samples)],
            device=device,
            dtype=torch.long
        )
    else:
        labels = class_labels.to(device).long()

    # Reverse diffusion loop: T -> 0
    for t in reversed(range(timesteps)):
        t_tensor = torch.full(
            (num_samples,),
            t,
            device=device,
            dtype=torch.long
        )

        pred_noise = model(imgs, t_tensor, labels)

        alpha_t = alphas[t]
        alpha_bar_t = alphas_cumprod[t]
        beta_t = betas[t]

        if t > 0:
            noise = torch.randn_like(imgs)
        else:
            noise = torch.zeros_like(imgs)

        imgs = (
            (1.0 / torch.sqrt(alpha_t))
            * (
                imgs
                - ((1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t))
                * pred_noise
            )
            + torch.sqrt(beta_t) * noise
        )

    return imgs, labels


def train_diffusion(
    model,
    dataloader,
    device,
    num_classes,
    timesteps=200,
    epochs=20,
    lr=1e-4,
    checkpoint_dir="checkpoints",
    save_every=5,
):
    """
    Train the Conditional Diffusion Model.

    The model learns to predict the noise added to clean images at random timesteps.

    Args:
        model: ConditionalUNet model.
        dataloader: DataLoader with MNIST images and labels.
        device: Torch device.
        num_classes: Number of classes.
        timesteps: Number of diffusion steps.
        epochs: Number of training epochs.
        lr: Learning rate.
        checkpoint_dir: Directory for saving checkpoints.
        save_every: Save checkpoint every N epochs.

    Returns:
        losses: List of average training losses per epoch.
    """

    os.makedirs(checkpoint_dir, exist_ok=True)

    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    betas = linear_beta_schedule(timesteps).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0

        pbar = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{epochs}"
        )

        for imgs, labels in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device).long()

            batch_size = imgs.size(0)

            # Sample random timesteps for each image in the batch
            t = torch.randint(
                0,
                timesteps,
                (batch_size,),
                device=device
            ).long()

            # Sample true Gaussian noise
            noise = torch.randn_like(imgs)

            # Forward diffusion process:
            # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
            sqrt_alpha_bar = torch.sqrt(alphas_cumprod[t])[:, None, None, None]
            sqrt_one_minus_alpha_bar = torch.sqrt(
                1.0 - alphas_cumprod[t]
            )[:, None, None, None]

            noisy_imgs = (
                sqrt_alpha_bar * imgs
                + sqrt_one_minus_alpha_bar * noise
            )

            # Predict the noise added to the image
            pred_noise = model(noisy_imgs, t, labels)

            # MSE loss between predicted noise and actual noise
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            pbar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)

        print(
            f"Epoch {epoch + 1}/{epochs} completed. "
            f"Average Loss: {avg_loss:.4f}"
        )

        # Save checkpoint every N epochs
        if (epoch + 1) % save_every == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f"diffusion_unet_epoch_{epoch + 1}.pt"
            )

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                    "timesteps": timesteps,
                },
                checkpoint_path
            )

            print(f"Saved diffusion checkpoint: {checkpoint_path}")

    # Save final model
    final_path = os.path.join(
        checkpoint_dir,
        "diffusion_unet_final.pt"
    )

    torch.save(
        {
            "epoch": "final",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "losses": losses,
            "timesteps": timesteps,
        },
        final_path
    )

    print(f"Final diffusion model saved at: {final_path}")

    return losses