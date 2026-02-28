"""
WGAN-GP (Wasserstein GAN with Gradient Penalty)
Stable GAN training with smooth loss landscape
Part of Hive Mega Hybrid GAN System
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import numpy as np


class WGANGenerator(nn.Module):
    """
    WGAN-GP Generator
    Generates synthetic fraud transactions from noise
    """
    
    def __init__(self, noise_dim=100, output_dim=30, hidden_dims=[256]):
        super().__init__()
        
        # SIMPLIFIED: 100 -> 256 -> 30 (was 100->128->256->128->30)
        layers = []
        prev_dim = noise_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        # Output layer (no activation - continuous values)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
        self.noise_dim = noise_dim
    
    def forward(self, noise):
        """Generate fake fraud from noise"""
        return self.model(noise)
    
    def generate(self, n_samples, device='cuda'):
        """Generate N synthetic fraud samples"""
        noise = torch.randn(n_samples, self.noise_dim).to(device)
        with torch.no_grad():
            samples = self.forward(noise)
        return samples.cpu().numpy()


class WGANCritic(nn.Module):
    """
    WGAN-GP Critic (not discriminator - outputs score, not probability)
    """
    
    def __init__(self, input_dim=30, hidden_dims=[128]):
        super().__init__()
        
        # SIMPLIFIED: 30 -> 128 -> 1 (was 30->128->64->32->1)
        # WEAKENED to give generator more breathing room
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.4)  # Increased dropout to weaken critic
            ])
            prev_dim = hidden_dim
        
        # Output layer (NO sigmoid - Wasserstein distance)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """Output: Real score (higher = more real-like)"""
        return self.model(x)


def compute_gradient_penalty(critic, real_data, fake_data, device='cuda', lambda_gp=10):
    """
    Compute gradient penalty for WGAN-GP
    
    GP = E[(||∇D(x_hat)||₂ - 1)²]
    where x_hat = α*x_real + (1-α)*x_fake
    """
    batch_size = real_data.size(0)
    
    # Random interpolation coefficient
    alpha = torch.rand(batch_size, 1).to(device)
    alpha = alpha.expand_as(real_data)
    
    # Interpolated samples
    interpolated = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)
    
    # Critic score on interpolated
    critic_interpolated = critic(interpolated)
    
    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=critic_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(critic_interpolated).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Gradient penalty
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1) ** 2).mean() * lambda_gp
    
    return gradient_penalty


class WGAN_GP:
    """
    Complete WGAN-GP training system with speed optimizations
    """
    
    def __init__(
        self,
        noise_dim=100,
        data_dim=30,
        lr=0.0002,
        n_critic=5,
        lambda_gp=10,
        use_amp=True,
        device='cuda'
    ):
        self.device = device
        self.noise_dim = noise_dim
        self.n_critic = n_critic
        self.lambda_gp = lambda_gp
        self.use_amp = use_amp
        
        # Initialize networks
        self.generator = WGANGenerator(noise_dim, data_dim).to(device)
        self.critic = WGANCritic(data_dim).to(device)
        
        # Optimizers (RMSprop recommended for WGAN)
        self.optimizer_G = torch.optim.RMSprop(self.generator.parameters(), lr=lr)
        self.optimizer_C = torch.optim.RMSprop(self.critic.parameters(), lr=lr)
        
        # Mixed Precision scaler for 2x speedup
        self.scaler_G = GradScaler() if use_amp else None
        self.scaler_C = GradScaler() if use_amp else None
        
        # Training history
        self.history = {
            'critic_loss': [],
            'generator_loss': [],
            'gradient_penalty': []
        }
    
    def train_epoch(self, real_data_loader, epoch):
        """
        Train for one epoch
        
        Args:
            real_data_loader: DataLoader for real fraud data
            epoch: Current epoch number
        """
        self.generator.train()
        self.critic.train()
        
        epoch_critic_loss = 0
        epoch_gen_loss = 0
        epoch_gp = 0
        n_batches = 0
        
        for batch_idx, real_batch in enumerate(real_data_loader):
            real_batch = real_batch.to(self.device)
            batch_size = real_batch.size(0)
            
            # ===========================
            # Train Critic (n_critic times)
            # ===========================
            for _ in range(self.n_critic):
                self.optimizer_C.zero_grad()
                
                # Generate fake data
                noise = torch.randn(batch_size, self.noise_dim).to(self.device)
                
                if self.use_amp:
                    with autocast():
                        fake_batch = self.generator(noise)
                        
                        # Critic scores
                        critic_real = self.critic(real_batch).mean()
                        critic_fake = self.critic(fake_batch.detach()).mean()
                        
                        # Gradient penalty
                        gp = compute_gradient_penalty(
                            self.critic, real_batch, fake_batch.detach(),
                            self.device, self.lambda_gp
                        )
                        
                        # Wasserstein loss
                        critic_loss = critic_fake - critic_real + gp
                    
                    self.scaler_C.scale(critic_loss).backward()
                    self.scaler_C.step(self.optimizer_C)
                    self.scaler_C.update()
                else:
                    fake_batch = self.generator(noise)
                    critic_real = self.critic(real_batch).mean()
                    critic_fake = self.critic(fake_batch.detach()).mean()
                    gp = compute_gradient_penalty(
                        self.critic, real_batch, fake_batch.detach(),
                        self.device, self.lambda_gp
                    )
                    critic_loss = critic_fake - critic_real + gp
                    critic_loss.backward()
                    self.optimizer_C.step()
                
                epoch_critic_loss += critic_loss.item()
                epoch_gp += gp.item()
            
            # ===========================
            # Train Generator (once)
            # ===========================
            self.optimizer_G.zero_grad()
            
            noise = torch.randn(batch_size, self.noise_dim).to(self.device)
            
            if self.use_amp:
                with autocast():
                    fake_batch = self.generator(noise)
                    gen_loss = -self.critic(fake_batch).mean()
                
                self.scaler_G.scale(gen_loss).backward()
                self.scaler_G.step(self.optimizer_G)
                self.scaler_G.update()
            else:
                fake_batch = self.generator(noise)
                gen_loss = -self.critic(fake_batch).mean()
                gen_loss.backward()
                self.optimizer_G.step()
            
            epoch_gen_loss += gen_loss.item()
            n_batches += 1
        
        # Average losses
        avg_critic_loss = epoch_critic_loss / (n_batches * self.n_critic)
        avg_gen_loss = epoch_gen_loss / n_batches
        avg_gp = epoch_gp / (n_batches * self.n_critic)
        
        # Record history
        self.history['critic_loss'].append(avg_critic_loss)
        self.history['generator_loss'].append(avg_gen_loss)
        self.history['gradient_penalty'].append(avg_gp)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}] "
                  f"C_loss: {avg_critic_loss:.4f} | "
                  f"G_loss: {avg_gen_loss:.4f} | "
                  f"GP: {avg_gp:.4f}")
        
        return avg_critic_loss, avg_gen_loss
    
    def save(self, path):
        """Save model checkpoint"""
        torch.save({
            'generator': self.generator.state_dict(),
            'critic': self.critic.state_dict(),
            'optimizer_G': self.optimizer_G.state_dict(),
            'optimizer_C': self.optimizer_C.state_dict(),
            'history': self.history
        }, path)
        print(f"💾 WGAN-GP saved to {path}")
    
    def load(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        self.optimizer_C.load_state_dict(checkpoint['optimizer_C'])
        self.history = checkpoint['history']
        print(f"📂 WGAN-GP loaded from {path}")


# Test
if __name__ == "__main__":
    print("Testing WGAN-GP...")
    
    # Dummy data
    real_data = torch.randn(1000, 30)
    train_loader = torch.utils.data.DataLoader(real_data, batch_size=64, shuffle=True)
    
    # Initialize
    wgan = WGAN_GP(device='cpu', use_amp=False)
    
    # Train
    for epoch in range(5):
        wgan.train_epoch(train_loader, epoch)
    
    # Generate
    samples = wgan.generator.generate(10, device='cpu')
    print(f"\n✅ Generated {len(samples)} samples")
    print(f"Shape: {samples.shape}")
