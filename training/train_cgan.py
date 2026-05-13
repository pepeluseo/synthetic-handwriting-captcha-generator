"""
train_cgan.py
-------------
Reusable training function for Conditional GAN experiments.

This module implements the standard adversarial training loop:
- Discriminator update using real and fake images
- Generator update using BCE loss to fool the discriminator
- Periodic checkpoint saving
"""

import os
import torch


def train_cgan(
    generator,
    discriminator,
    dataloader,
    optimizer_G,
    optimizer_D,
    criterion,
    device,
    z_dim,
    num_classes,
    epochs=50,
    checkpoint_dir="checkpoints",
    save_every=10,
):
    """
    Train a Conditional GAN on class-labeled image data.

    Args:
        generator: Conditional Generator model.
        discriminator: Conditional Discriminator model.
        dataloader: DataLoader providing real images and class labels.
        optimizer_G: Optimizer for the generator.
        optimizer_D: Optimizer for the discriminator.
        criterion: Loss function, usually nn.BCELoss().
        device: torch device, either "cuda" or "cpu".
        z_dim: Dimension of latent noise vector.
        num_classes: Number of conditioning classes.
        epochs: Number of training epochs.
        checkpoint_dir: Directory where model checkpoints are saved.
        save_every: Save checkpoints every N epochs.

    Returns:
        history: Dictionary containing generator and discriminator losses.
    """

    os.makedirs(checkpoint_dir, exist_ok=True)

    generator.train()
    discriminator.train()

    history = {
        "g_loss": [],
        "d_loss": []
    }

    for epoch in range(epochs):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0

        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            batch_size = imgs.size(0)

            # Labels for BCE loss
            valid = torch.ones(batch_size, 1, device=device)
            fake = torch.zeros(batch_size, 1, device=device)

            # -------------------------------------------------
            # 1. Train Generator
            # -------------------------------------------------
            optimizer_G.zero_grad()

            # Sample random noise and random class labels
            z = torch.randn(batch_size, z_dim, device=device)
            gen_labels = torch.randint(
                0,
                num_classes,
                (batch_size,),
                device=device
            )

            # Generate fake images conditioned on labels
            gen_imgs = generator(z, gen_labels)

            # Generator tries to make discriminator classify fake images as real
            g_loss = criterion(discriminator(gen_imgs, gen_labels), valid)

            g_loss.backward()
            optimizer_G.step()

            # -------------------------------------------------
            # 2. Train Discriminator
            # -------------------------------------------------
            optimizer_D.zero_grad()

            # Real image loss
            real_loss = criterion(discriminator(imgs, labels), valid)

            # Fake image loss
            # detach() prevents gradients from flowing into the generator
            fake_loss = criterion(
                discriminator(gen_imgs.detach(), gen_labels),
                fake
            )

            # Average discriminator loss
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()

        avg_g_loss = epoch_g_loss / len(dataloader)
        avg_d_loss = epoch_d_loss / len(dataloader)

        history["g_loss"].append(avg_g_loss)
        history["d_loss"].append(avg_d_loss)

        print(
            f"[Epoch {epoch + 1}/{epochs}] "
            f"D loss: {avg_d_loss:.4f} | "
            f"G loss: {avg_g_loss:.4f}"
        )

        # -------------------------------------------------
        # 3. Save checkpoints
        # -------------------------------------------------
        if (epoch + 1) % save_every == 0:
            generator_path = os.path.join(
                checkpoint_dir,
                f"cgan_generator_epoch_{epoch + 1}.pt"
            )

            discriminator_path = os.path.join(
                checkpoint_dir,
                f"cgan_discriminator_epoch_{epoch + 1}.pt"
            )

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": generator.state_dict(),
                    "optimizer_state_dict": optimizer_G.state_dict(),
                    "g_loss": avg_g_loss,
                },
                generator_path
            )

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": discriminator.state_dict(),
                    "optimizer_state_dict": optimizer_D.state_dict(),
                    "d_loss": avg_d_loss,
                },
                discriminator_path
            )

            print(f"Saved generator checkpoint: {generator_path}")
            print(f"Saved discriminator checkpoint: {discriminator_path}")

    return history