"""
Generator Network for Fraud Transaction GAN

The Generator takes random noise as input and generates synthetic fraudulent 
credit card transactions that mimic the statistical properties of real fraud.
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    Generator network that creates synthetic fraud transactions from random noise
    
    Architecture:
        Input: Random noise vector (latent_dim)
        Hidden layers with progressive expansion
        Output: Synthetic transaction features (output_dim)
    """
    
    def __init__(self, latent_dim=100, output_dim=30, hidden_dims=[256]):
        """
        Args:
            latent_dim: Dimension of random noise input
            output_dim: Number of features to generate (30 for creditcard.csv: Time + V1-V28 + Amount)
            hidden_dims: List of hidden layer dimensions (SIMPLIFIED for small dataset - 393 fraud samples)
        """
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        # SIMPLIFIED architecture: 100 -> 256 -> 30
        # Reason: Deep networks (100->128->256->512->30) overfit on 393 samples
        layers = []
        prev_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.extend([
            nn.Linear(prev_dim, output_dim),
            nn.Tanh()  # Output in range [-1, 1] to match normalized data
        ])
        
        self.model = nn.Sequential(*layers)
        
        # Initialize weights with slightly larger std for faster learning
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0.0, 0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, z):
        """
        Generate synthetic fraud transactions
        
        Args:
            z: Random noise tensor (batch_size, latent_dim)
        
        Returns:
            Synthetic fraud features (batch_size, output_dim)
        """
        return self.model(z)
    
    def generate(self, num_samples, device='cpu'):
        """
        Utility function to generate synthetic samples
        
        Args:
            num_samples: Number of synthetic frauds to generate
            device: Device to generate on ('cpu' or 'cuda')
        
        Returns:
            Numpy array of synthetic fraud transactions
        """
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(device)
            synthetic_fraud = self(z)
        return synthetic_fraud.cpu().numpy()


if __name__ == "__main__":
    # Test Generator
    print("="*50)
    print("Testing Generator")
    print("="*50)
    
    latent_dim = 100
    output_dim = 30
    batch_size = 10
    
    generator = Generator(latent_dim=latent_dim, output_dim=output_dim)
    print(f"\nGenerator Architecture:")
    print(generator)
    
    # Test forward pass
    z = torch.randn(batch_size, latent_dim)
    fake_fraud = generator(z)
    
    print(f"\nInput noise shape: {z.shape}")
    print(f"Generated fraud shape: {fake_fraud.shape}")
    print(f"Output range: [{fake_fraud.min():.2f}, {fake_fraud.max():.2f}]")
    
    # Test generate utility
    synthetic_samples = generator.generate(num_samples=5)
    print(f"\nGenerated {len(synthetic_samples)} synthetic fraud transactions")
    print(f"Sample features shape: {synthetic_samples.shape}")
    
    print("\n[SUCCESS] Generator test complete!")
