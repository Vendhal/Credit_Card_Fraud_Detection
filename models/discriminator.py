"""
Discriminator Network for Fraud Transaction GAN

The Discriminator distinguishes between real and synthetic (fake) fraudulent 
transactions, helping the Generator learn to create realistic fraud patterns.
"""

import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    Discriminator network that classifies transactions as real or fake
    
    Architecture:
        Input: Transaction features (input_dim)
        Hidden layers with progressive compression
        Output: Probability of being real (0-1)
    """
    
    def __init__(self, input_dim=30, hidden_dims=[256], dropout=0.4):
        """
        Args:
            input_dim: Number of input features (30 for creditcard.csv: Time + V1-V28 + Amount)
            hidden_dims: List of hidden layer dimensions (SIMPLIFIED to weaken discriminator)
            dropout: Dropout rate to prevent discriminator from overpowering generator (INCREASED)
        """
        super(Discriminator, self).__init__()
        
        self.input_dim = input_dim
        
        # SIMPLIFIED architecture: 30 -> 256 -> 1
        # Reason: Deep disc (30->512->256->128->1) crushes weak generators on small dataset
        # Higher dropout (0.4) further weakens discriminator
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        # REMOVED Sigmoid for Hybrid Loss (Wasserstein+Hinge+LSGAN)
        # Output raw scores (logits) instead of probabilities
        layers.append(nn.Linear(prev_dim, 1))
        # NO activation - raw output for Hybrid Loss!
        
        self.model = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0.0, 0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        Classify transaction as real or fake
        
        Args:
            x: Transaction features (batch_size, input_dim)
        
        Returns:
            Probability of being real (batch_size, 1)
        """
        return self.model(x)


if __name__ == "__main__":
    # Test Discriminator
    print("="*50)
    print("Testing Discriminator")
    print("="*50)
    
    input_dim = 30
    batch_size = 10
    
    discriminator = Discriminator(input_dim=input_dim)
    print(f"\nDiscriminator Architecture:")
    print(discriminator)
    
    # Test forward pass
    x = torch.randn(batch_size, input_dim)
    prediction = discriminator(x)
    
    print(f"\nInput transaction shape: {x.shape}")
    print(f"Prediction shape: {prediction.shape}")
    print(f"Prediction range: [{prediction.min():.4f}, {prediction.max():.4f}]")
    print(f"Sample predictions: {prediction.squeeze().tolist()[:5]}")
    
    print("\n[SUCCESS] Discriminator test complete!")
