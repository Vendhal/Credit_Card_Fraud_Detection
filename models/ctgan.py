"""
CTGAN Wrapper for Hive Ensemble
Conditional Tabular GAN - Specialized for tabular data

Based on: https://github.com/sdv-dev/CTGAN
Paper: "Modeling Tabular Data using Conditional GAN" (Xu et al., 2019)

Key Features:
- Designed specifically for tabular/credit card data
- Handles mixed data types (continuous + categorical)
- Mode-specific normalization
- Conditional generator based on data distribution
"""

import torch
import torch.nn as nn
import numpy as np


class CTGANGenerator(nn.Module):
    """
    CTGAN Generator for tabular data
    
    Architecture optimized for numerical features with proper normalization
    """
    
    def __init__(self, noise_dim=100, data_dim=30, embedding_dim=128):
        super(CTGANGenerator, self).__init__()
        
        self.noise_dim = noise_dim
        self.data_dim = data_dim
        
        # SIMPLIFIED: 100 -> 256 -> 30 (was 100->256->256->128->30)
        # Removed residual blocks - too complex for 393 samples
        self.model = nn.Sequential(
            # Single hidden layer
            nn.Linear(noise_dim, embedding_dim * 2),
            nn.BatchNorm1d(embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Output layer
            nn.Linear(embedding_dim * 2, data_dim)
        )
        
        # Apply Tanh to normalize output to [-1, 1]
        self.activation = nn.Tanh()
    
    def forward(self, noise):
        """Generate samples from noise"""
        output = self.model(noise)
        return self.activation(output)
    
    def generate(self, n_samples, device='cuda'):
        """Generate n_samples"""
        self.eval()
        with torch.no_grad():
            noise = torch.randn(n_samples, self.noise_dim).to(device)
            samples = self(noise)
        return samples.cpu().numpy()


class CTGANDiscriminator(nn.Module):
    """
    CTGAN Discriminator (Critic)
    
    Uses PacGAN technique (processes multiple samples together)
    """
    
    def __init__(self, data_dim=30, embedding_dim=128, pac=10):
        super(CTGANDiscriminator, self).__init__()
        
        self.pac = pac  # Number of samples to pack together
        self.data_dim = data_dim
        
        # Input is pac * data_dim
        input_dim = data_dim * pac
        
        # SIMPLIFIED: 60 -> 128 -> 1 (was 60->256->128->1)
        self.model = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            
            nn.Linear(embedding_dim, 1)
        )
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, data_dim]
        
        Returns:
            scores: [batch_size // pac, 1]
        """
        batch_size = x.size(0)
        
        # Pad if batch size not divisible by pac
        if batch_size % self.pac != 0:
            pad_size = self.pac - (batch_size % self.pac)
            x = torch.cat([x, x[:pad_size]], dim=0)
            batch_size = x.size(0)
        
        # Reshape to [batch_size // pac, pac * data_dim]
        x = x.view(-1, self.pac * self.data_dim)
        
        return self.model(x)


class CTGAN:
    """
    CTGAN Trainer - Simplified version for integration
    
    Full CTGAN has:
    - Mode-specific normalization
    - Conditional vectors
    - PacGAN discriminator
    
    This is a lightweight version adapted for Hive Ensemble
    """
    
    def __init__(
        self,
        data_dim=30,
        noise_dim=100,
        embedding_dim=128,
        pac=10,
        device='cuda',
        use_amp=False
    ):
        self.device = device
        self.use_amp = use_amp
        
        # Initialize models
        self.generator = CTGANGenerator(noise_dim, data_dim, embedding_dim).to(device)
        self.discriminator = CTGANDiscriminator(data_dim, embedding_dim, pac).to(device)
        
        # Optimizers (CTGAN uses Adam with specific params)
        self.optimizer_G = torch.optim.Adam(
            self.generator.parameters(),
            lr=2e-4,
            betas=(0.5, 0.9)
        )
        self.optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=2e-4,
            betas=(0.5, 0.9)
        )
        
        # Loss (CTGAN uses Wasserstein loss with gradient penalty)
        self.criterion = nn.BCEWithLogitsLoss()
        
        # AMP scaler
        if use_amp:
            from torch.cuda.amp import GradScaler
            self.scaler_G = GradScaler()
            self.scaler_D = GradScaler()
    
    def train_epoch(self, data_loader, epoch):
        """
        Train one epoch
        
        Args:
            data_loader: DataLoader with real data
            epoch: Current epoch number
        
        Returns:
            avg_loss_D, avg_loss_G
        """
        self.generator.train()
        self.discriminator.train()
        
        losses_D = []
        losses_G = []
        
        for real_data in data_loader:
            real_data = real_data.to(self.device)
            batch_size = real_data.size(0)
            
            # ====================
            # Train Discriminator
            # ====================
            self.optimizer_D.zero_grad()
            
            # Real samples
            real_labels = torch.ones(batch_size // self.discriminator.pac, 1).to(self.device)
            real_output = self.discriminator(real_data)
            loss_D_real = self.criterion(real_output, real_labels)
            
            # Fake samples
            noise = torch.randn(batch_size, self.generator.noise_dim).to(self.device)
            fake_data = self.generator(noise).detach()
            fake_labels = torch.zeros(batch_size // self.discriminator.pac, 1).to(self.device)
            fake_output = self.discriminator(fake_data)
            loss_D_fake = self.criterion(fake_output, fake_labels)
            
            # Total discriminator loss
            loss_D = loss_D_real + loss_D_fake
            loss_D.backward()
            self.optimizer_D.step()
            
            losses_D.append(loss_D.item())
            
            # ====================
            # Train Generator
            # ====================
            self.optimizer_G.zero_grad()
            
            noise = torch.randn(batch_size, self.generator.noise_dim).to(self.device)
            fake_data = self.generator(noise)
            fake_output = self.discriminator(fake_data)
            
            # Generator tries to fool discriminator
            loss_G = self.criterion(fake_output, real_labels)
            loss_G.backward()
            self.optimizer_G.step()
            
            losses_G.append(loss_G.item())
        
        return np.mean(losses_D), np.mean(losses_G)


# Test
if __name__ == "__main__":
    print("Testing CTGAN...")
    
    # Create model
    ctgan = CTGAN(device='cpu', use_amp=False)
    
    # Generate samples
    samples = ctgan.generator.generate(10, device='cpu')
    print(f"✅ Generated {len(samples)} samples with shape {samples.shape}")
    
    # Test training
    dummy_data = torch.randn(100, 30)
    from torch.utils.data import DataLoader
    loader = DataLoader(dummy_data, batch_size=20)
    
    loss_D, loss_G = ctgan.train_epoch(loader, 0)
    print(f"✅ Training working - D_loss: {loss_D:.3f}, G_loss: {loss_G:.3f}")
    
    print("\n✅ CTGAN test complete!")
