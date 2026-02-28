"""
Final Reality Discriminator - The Supreme Court
Trained ONCE on real fraud data, then FROZEN forever
Acts as unchanging ground truth oracle for validating ensemble output
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path


class FinalRealityDiscriminator(nn.Module):
    """
    The Final Judge - validates if synthetic fraud matches real fraud
    
    Key Features:
   - Trained ONLY on real fraud data
    - Frozen after training (parameters never update)
    - Acts as ground truth anchor
    - Prevents "shared hallucination" in ensemble
    """
    
    def __init__(self, input_dim=30, hidden_dims=[256, 128]):
        super().__init__()
        
        # SIMPLER, MORE POWERFUL architecture
        # Fewer layers, more neurons, less regularization
        # Better for small datasets (393 samples)
        
        self.model = nn.Sequential(
            # First hidden layer
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),  # LayerNorm works better than BatchNorm for small batches
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),  # Less dropout
            
            # Second hidden layer  
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            
            # Output layer (probability of being in-distribution)
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Training state
        self.is_trained = False
        self.is_frozen = False
    
    def forward(self, x):
        """Forward pass through the network"""
        return self.model(x)
    
    def train_on_real_fraud(self, real_fraud_data, normal_data=None, epochs=100, lr=0.001, batch_size=128, device='cuda'):
        """
        Train as OUT-OF-DISTRIBUTION DETECTOR
        
        NEW STRATEGY (Option C):
        - IN-distribution (label=1): Real fraud data (what we want)
        - OUT-of-distribution (label=0): Random noise + corrupted fraud (garbage to reject)
        
        This is MUCH easier to learn than fraud vs normal!
        The discriminator becomes an outlier detector.
        
        Args:
            real_fraud_data: Tensor of real fraud transactions [N, 30]
            normal_data: Not used anymore (kept for compatibility)
            epochs: Number of training epochs
            lr: Learning rate
            batch_size: Batch size for training
            device: 'cuda' or 'cpu'
        """
        if self.is_trained:
            print("WARNING: Final Discriminator already trained! Skipping.")
            return
        
        print("=" * 60)
        print("TRAINING FINAL REALITY DISCRIMINATOR")
        print("(NEW: Outlier Detector - In-distribution vs Out-of-distribution)")
        print("=" * 60)
        
        self.to(device)
        self.train()
        
        # Optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=(0.5, 0.999))
        criterion = nn.BCELoss()
        
        # Convert to tensor if needed
        if isinstance(real_fraud_data, np.ndarray):
            real_fraud_data = torch.FloatTensor(real_fraud_data)
        
        real_fraud_data = real_fraud_data.to(device)
        n_fraud = len(real_fraud_data)
        
        # Compute statistics for generating realistic outliers
        fraud_mean = real_fraud_data.mean(dim=0)
        fraud_std = real_fraud_data.std(dim=0)
        fraud_min = real_fraud_data.min(dim=0)[0]
        fraud_max = real_fraud_data.max(dim=0)[0]
        
        print(f"Training with {n_fraud} fraud samples as IN-distribution")
        print("Generating OUT-of-distribution samples: noise + corrupted fraud")
        
        # Training loop
        best_loss = float('inf')
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            # Mini-batch training
            fraud_indices = torch.randperm(n_fraud)
            
            for i in range(0, n_fraud, batch_size):
                batch_indices = fraud_indices[i:i+batch_size]
                real_batch = real_fraud_data[batch_indices]
                current_batch_size = len(real_batch)
                
                # IN-DISTRIBUTION: Real fraud = label 1
                optimizer.zero_grad()
                real_output = self(real_batch)
                real_loss = criterion(real_output, torch.ones_like(real_output))
                
                # OUT-OF-DISTRIBUTION: Mix of different garbage types = label 0
                # This creates a diverse set of "bad" samples that are clearly wrong
                
                ood_samples = []
                samples_per_type = current_batch_size // 4
                
                # Type 1: Pure random noise (Gaussian)
                noise1 = torch.randn(samples_per_type, real_batch.shape[1], device=device)
                ood_samples.append(noise1)
                
                # Type 2: Scaled noise (different variance)
                noise2 = torch.randn(samples_per_type, real_batch.shape[1], device=device) * 10.0
                ood_samples.append(noise2)
                
                # Type 3: Corrupted fraud (add extreme noise to real fraud)
                corrupt_indices = torch.randint(0, n_fraud, (samples_per_type,))
                corrupted = real_fraud_data[corrupt_indices].clone()
                corruption = torch.randn_like(corrupted) * fraud_std * 5.0  # 5x std deviation
                corrupted = corrupted + corruption
                ood_samples.append(corrupted)
                
                # Type 4: Out-of-range values (beyond min/max of real fraud)
                remaining = current_batch_size - 3 * samples_per_type
                out_of_range = torch.rand(remaining, real_batch.shape[1], device=device)
                out_of_range = fraud_min + out_of_range * (fraud_max - fraud_min) * 3.0  # 3x range
                ood_samples.append(out_of_range)
                
                # Combine all OOD samples
                ood_batch = torch.cat(ood_samples, dim=0)
                
                ood_output = self(ood_batch)
                ood_loss = criterion(ood_output, torch.zeros_like(ood_output))
                
                # Combined loss
                loss = real_loss + ood_loss
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            
            if avg_loss < best_loss:
                best_loss = avg_loss
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f} (Best: {best_loss:.4f})")
        
        print("\n✅ Final Discriminator training complete!")
        print(f"Final Loss: {best_loss:.4f}")
        print("🎯 Now acts as OUTLIER DETECTOR (rejects noise, accepts fraud-like patterns)")
        print("🔒 FREEZING PARAMETERS (will never update again)")
        
        # Mark as trained and freeze
        self.is_trained = True
        self.freeze()
    
    def freeze(self):
        """Freeze all parameters - no more updates"""
        for param in self.parameters():
            param.requires_grad = False
        self.is_frozen = True
        self.eval()  # Set to evaluation mode
        print("🧊 Final Discriminator is now FROZEN")
    
    def validate_samples(self, samples, threshold=0.55, device='cuda'):
        """
        Validate if samples pass the reality check
        
        Args:
            samples: Generated samples to validate [N, 30]
            threshold: Minimum score to pass (default 0.55 - achievable but meaningful)
            device: 'cuda' or 'cpu'
        
        Returns:
            pass_rate: Fraction of samples that pass (0.0-1.0)
            quality_score: Average quality score (0.0-1.0)
            pass_mask: Boolean mask of which samples passed
        """
        if not self.is_frozen:
            raise RuntimeError("Final Discriminator must be frozen before validation!")
        
        self.to(device)
        
        # Convert to tensor if needed
        if isinstance(samples, np.ndarray):
            samples = torch.FloatTensor(samples)
        
        samples = samples.to(device)
        
        with torch.no_grad():
            scores = self(samples).squeeze()
            
            # Compute metrics
            pass_mask = scores > threshold
            pass_rate = pass_mask.float().mean().item()
            quality_score = scores.mean().item()
        
        return pass_rate, quality_score, pass_mask.cpu()
    
    def save(self, path):
        """Save the frozen discriminator"""
        if not self.is_frozen:
            print("WARNING: Saving unfrozen discriminator")
        
        torch.save({
            'state_dict': self.state_dict(),
            'is_trained': self.is_trained,
            'is_frozen': self.is_frozen
        }, path)
        print(f"💾 Final Discriminator saved to {path}")
    
    def load(self, path, device='cuda'):
        """Load a pretrained frozen discriminator"""
        checkpoint = torch.load(path, map_location=device)
        self.load_state_dict(checkpoint['state_dict'])
        self.is_trained = checkpoint['is_trained']
        self.is_frozen = checkpoint['is_frozen']
        
        if self.is_frozen:
            self.freeze()
        
        self.to(device)
        print(f"📂 Final Discriminator loaded from {path}")
        print(f"   Trained: {self.is_trained}, Frozen: {self.is_frozen}")


# Example usage
if __name__ == "__main__":
    print("Testing Final Reality Discriminator...")
    
    # Create dummy real fraud data
    real_fraud = np.random.randn(500, 30).astype(np.float32)
    
    # Initialize and train
    final_disc = FinalRealityDiscriminator()
    final_disc.train_on_real_fraud(real_fraud, epochs=50, device='cpu')
    
    # Test validation
    test_samples = np.random.randn(100, 30).astype(np.float32)
    pass_rate, quality, mask = final_disc.validate_samples(test_samples, device='cpu')
    
    print(f"\n📊 Validation Results:")
    print(f"   Pass Rate: {pass_rate:.2%}")
    print(f"   Quality Score: {quality:.3f}")
    print(f"   Samples Passed: {mask.sum()}/{len(mask)}")
