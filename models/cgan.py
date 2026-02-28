"""
Conditional GAN (CGAN) - Controlled Fraud Generation
Generates fraud with specific characteristics based on fuzzy rules
Part of Hive Mega Hybrid GAN System
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Fraud type definitions (based on fuzzy rules)
FRAUD_TYPES = {
    0: 'high_amount_unusual_hour',      # Amount > $500, Hour in [0-6, 22-24]
    1: 'rapid_sequence',                # Multiple transactions < 5 min
    2: 'foreign_location',              # Unusual location patterns (V features)
    3: 'small_test_then_large'          # Small → Large within 1 hour
}

N_FRAUD_TYPES = len(FRAUD_TYPES)


class ConditionalGenerator(nn.Module):
    """
    Conditional GAN Generator
    Generates fraud with specific characteristics
    """
    
    def __init__(self, noise_dim=100, condition_dim=4, output_dim=30, hidden_dims=[256]):
        super().__init__()
        
        self.noise_dim = noise_dim
        self.condition_dim = condition_dim
        input_dim = noise_dim + condition_dim
        
        # SIMPLIFIED: 104 -> 256 -> 30 (was 104->128->256->128->30)
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, noise, condition):
        """
        Generate fraud conditioned on fraud type
        
        Args:
            noise: Random noise [batch_size, noise_dim]
            condition: One-hot encoded fraud type [batch_size, condition_dim]
        
        Returns:
            Synthetic fraud [batch_size, output_dim]
        """
        # Concatenate noise and condition
        x = torch.cat([noise, condition], dim=1)
        return self.model(x)
    
    def generate(self, n_samples, fraud_type=None, device='cuda'):
        """
        Generate N samples of specific fraud type
        
        Args:
            n_samples: Number of samples
            fraud_type: Fraud type (0-3), or None for random
            device: 'cuda' or 'cpu'
        
        Returns:
            Samples [n_samples, output_dim]
        """
        noise = torch.randn(n_samples, self.noise_dim).to(device)
        
        if fraud_type is None:
            # Random fraud types
            fraud_types = torch.randint(0, N_FRAUD_TYPES, (n_samples,))
        else:
            # Specific fraud type
            fraud_types = torch.full((n_samples,), fraud_type)
        
        # One-hot encode
        condition = F.one_hot(fraud_types, num_classes=N_FRAUD_TYPES).float().to(device)
        
        with torch.no_grad():
            samples = self.forward(noise, condition)
        
        return samples.cpu().numpy()


class ConditionalDiscriminator(nn.Module):
    """
    Conditional Discriminator
    Evaluates if fraud matches the specified type
    """
    
    def __init__(self, input_dim=30, condition_dim=4, hidden_dims=[128]):
        super().__init__()
        
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        total_input = input_dim + condition_dim
        
        # SIMPLIFIED: 34 -> 128 -> 1 (was 34->128->64->32->1)
        layers = []
        prev_dim = total_input
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.4)  # Increased to weaken discriminator
            ])
            prev_dim = hidden_dim
        
        # Output (real/fake probability)
        layers.extend([
            nn.Linear(prev_dim, 1),
            nn.Sigmoid()
        ])
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x, condition):
        """
        Classify if transaction is real/fake given condition
        
        Args:
            x: Transaction [batch_size, input_dim]
            condition: Fraud type [batch_size, condition_dim]
        
        Returns:
            Probability [batch_size, 1]
        """
        combined = torch.cat([x, condition], dim=1)
        return self.model(combined)


def condition_matching_loss(generated_fraud, condition, feature_ranges):
    """
    Ensure generated fraud matches the specified condition
    
    Args:
        generated_fraud: Generated transactions [batch_size, 30]
        condition: One-hot fraud type [batch_size, 4]
        feature_ranges: Dict with expected ranges per fraud type
    
    Returns:
        Matching loss (lower = better match)
    """
    loss = 0.0
    batch_size = generated_fraud.size(0)
    
    # Get fraud type indices
    fraud_types = condition.argmax(dim=1)
    
    for i in range(batch_size):
        fraud_type = fraud_types[i].item()
        transaction = generated_fraud[i]
        
        # Check condition-specific constraints
        if fraud_type == 0:  # high_amount_unusual_hour
            amount = transaction[29]  # Assuming Amount is last feature
            # Expect high amount
            if amount < 2.0:  # Normalized scale
                loss += (2.0 - amount) ** 2
        
        elif fraud_type == 1:  # rapid_sequence
            # Rapid sequence is temporal, hard to enforce on single transaction
            # Could check for similar V features (pattern consistency)
            pass
        
        elif fraud_type == 2:  # foreign_location
            # Check V features for unusual patterns
            v_features = transaction[:28]
            unusual_count = (torch.abs(v_features) > 2.0).float().sum()
            if unusual_count < 2:  # Expect at least 2 unusual features
                loss += (2.0 - unusual_count) ** 2
        
        elif fraud_type == 3:  # small_test_then_large
            amount = transaction[29]
            # This is also temporal, skip for now
            pass
    
    return loss / batch_size


class CGAN:
    """
    Complete Conditional GAN training system
    """
    
    def __init__(
        self,
        noise_dim=100,
        condition_dim=4,
        data_dim=30,
        lr=0.0002,
        device='cuda'
    ):
        self.device = device
        self.noise_dim = noise_dim
        self.condition_dim = condition_dim
        
        # Initialize networks
        self.generator = ConditionalGenerator(noise_dim, condition_dim, data_dim).to(device)
        self.discriminator = ConditionalDiscriminator(data_dim, condition_dim).to(device)
        
        # Optimizers
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        
        # Training history
        self.history = {
            'd_loss': [],
            'g_loss': [],
            'matching_loss': []
        }
    
    def train_epoch(self, real_data_loader, fraud_type_labels, epoch):
        """
        Train for one epoch
        
        Args:
            real_data_loader: DataLoader for real fraud
            fraud_type_labels: Fraud type for each real sample
            epoch: Current epoch
        """
        self.generator.train()
        self.discriminator.train()
        
        epoch_d_loss = 0
        epoch_g_loss = 0
        epoch_match_loss = 0
        n_batches = 0
        
        for batch_idx, (real_batch, fraud_types) in enumerate(zip(real_data_loader, fraud_type_labels)):
            real_batch = real_batch.to(self.device)
            
            # One-hot encode fraud types
            condition = F.one_hot(fraud_types, num_classes=N_FRAUD_TYPES).float().to(self.device)
            batch_size = real_batch.size(0)
            
            # ====================
            # Train Discriminator
            # ====================
            self.optimizer_D.zero_grad()
            
            # Real loss
            d_real = self.discriminator(real_batch, condition)
            d_real_loss = F.binary_cross_entropy(d_real, torch.ones_like(d_real))
            
            # Fake loss
            noise = torch.randn(batch_size, self.noise_dim).to(self.device)
            fake_batch = self.generator(noise, condition)
            d_fake = self.discriminator(fake_batch.detach(), condition)
            d_fake_loss = F.binary_cross_entropy(d_fake, torch.zeros_like(d_fake))
            
            # Total discriminator loss
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            self.optimizer_D.step()
            
            # ====================
            # Train Generator
            # ====================
            self.optimizer_G.zero_grad()
            
            # Generate again
            noise = torch.randn(batch_size, self.noise_dim).to(self.device)
            fake_batch = self.generator(noise, condition)
            
            # Fool discriminator
            d_fake = self.discriminator(fake_batch, condition)
            g_adv_loss = F.binary_cross_entropy(d_fake, torch.ones_like(d_fake))
            
            # Condition matching loss
            match_loss = condition_matching_loss(fake_batch, condition, {})
            
            # Total generator loss
            g_loss = g_adv_loss + 0.3 * match_loss
            g_loss.backward()
            self.optimizer_G.step()
            
            # Record
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            epoch_match_loss += match_loss.item()
            n_batches += 1
        
        # Averages
        avg_d_loss = epoch_d_loss / n_batches
        avg_g_loss = epoch_g_loss / n_batches
        avg_match_loss = epoch_match_loss / n_batches
        
        self.history['d_loss'].append(avg_d_loss)
        self.history['g_loss'].append(avg_g_loss)
        self.history['matching_loss'].append(avg_match_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}] "
                  f"D: {avg_d_loss:.4f} | "
                  f"G: {avg_g_loss:.4f} | "
                  f"Match: {avg_match_loss:.4f}")
        
        return avg_d_loss, avg_g_loss
    
    def save(self, path):
        """Save checkpoint"""
        torch.save({
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'optimizer_G': self.optimizer_G.state_dict(),
            'optimizer_D': self.optimizer_D.state_dict(),
            'history': self.history
        }, path)
        print(f"💾 CGAN saved to {path}")
    
    def load(self, path):
        """Load checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D'])
        self.history = checkpoint['history']
        print(f"📂 CGAN loaded from {path}")


# Test
if __name__ == "__main__":
    print("Testing Conditional GAN...")
    
    # Test conditional generator
    cgan = CGAN(device='cpu')
    
    # Generate each fraud type
    for fraud_type in range(N_FRAUD_TYPES):
        samples = cgan.generator.generate(5, fraud_type=fraud_type, device='cpu')
        print(f"\n✅ Generated 5 samples of type {fraud_type}: {FRAUD_TYPES[fraud_type]}")
        print(f"   Shape: {samples.shape}")
    
    print("\n✅ CGAN test complete!")
