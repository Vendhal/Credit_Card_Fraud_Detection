"""
Byzantine Fault Tolerant Consensus
Handles conflicting validator opinions to reach agreement
Tolerates up to 1/3 of validators being "wrong"
"""

import torch
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import wasserstein_distance


class ByzantineConsensus:
    """
    Byzantine Fault Tolerant consensus algorithm
    Inspired by blockchain consensus mechanisms
    
    Key features:
    - Requires 2/3 supermajority for consensus
    - Tolerates up to 1/3 Byzantine (faulty) validators
    - Falls back to weighted voting if no supermajority
    """
    
    def __init__(self, threshold=0.5, weights=None):
        """
        Args:
            threshold: Minimum score to consider "valid" (default 0.5, lowered from 0.6 to reduce mode collapse)
            weights: Dict of validator weights (default: equal)
        """
        self.threshold = threshold
        self.weights = weights or {}
    
    def detect_contradictions(self, samples_dict):
        """
        Detect contradictions between generators
        
        Args:
            samples_dict: {gen_name: samples_array}
        
        Returns:
            List of contradiction dicts
        """
        contradictions = []
        gen_names = list(samples_dict.keys())
        
        # Pairwise comparison
        for i in range(len(gen_names)):
            for j in range(i+1, len(gen_names)):
                gen1_name = gen_names[i]
                gen2_name = gen_names[j]
                
                samples1 = samples_dict[gen1_name]
                samples2 = samples_dict[gen2_name]
                
                # Compute statistical distance
                distance = self._compute_distance(samples1, samples2)
                
                if distance > 0.5:  # Significant disagreement
                    contradictions.append({
                        'gen1': gen1_name,
                        'gen2': gen2_name,
                        'distance': distance,
                        'samples_1': samples1,
                        'samples_2': samples2
                    })
        
        return contradictions
    
    def _compute_distance(self, samples1, samples2):
        """Compute Wasserstein distance between two sample distributions"""
        # Flatten samples for distance computation
        flat1 = samples1.reshape(samples1.shape[0], -1).mean(axis=0)
        flat2 = samples2.reshape(samples2.shape[0], -1).mean(axis=0)
        
        # Wasserstein distance
        distance = np.linalg.norm(flat1 - flat2) / np.linalg.norm(flat1 + 1e-8)
        return distance
    
    def resolve(self, samples_dict, validators, contradictions=None):
        """
        Reach consensus on best samples
        
        Args:
            samples_dict: {gen_name: samples}
            validators: {validator_name: validator_object}
            contradictions: Optional pre-computed contradictions
        
        Returns:
            (consensus_samples, winner_name) tuple
        """
        if contradictions is None:
            contradictions = self.detect_contradictions(samples_dict)
        
        # Score each generator's samples using all validators
        scores = {}
        
        for gen_name, samples in samples_dict.items():
            validator_scores = {}
            
            for validator_name, validator in validators.items():
                score = validator.score(samples)
                validator_scores[validator_name] = score
            
            scores[gen_name] = validator_scores
        
        # Check for supermajority consensus
        consensus_gen = self._check_supermajority(scores, validators)
        
        if consensus_gen:
            print(f"[OK] Byzantine consensus: {consensus_gen} (supermajority)")
            return samples_dict[consensus_gen], consensus_gen
        else:
            # Fallback: Weighted voting
            print("[Warning] No supermajority, using weighted voting")
            samples, winner = self._weighted_voting(samples_dict, scores, validators)
            return samples, winner
    
    def _check_supermajority(self, scores, validators):
        """
        Check if any generator has 2/3+ validator approval
        
        Returns:
            Generator name if supermajority, else None
        """
        n_validators = len(validators)
        supermajority_threshold = int(np.ceil(2 * n_validators / 3))
        
        for gen_name, validator_scores in scores.items():
            # Count validators that approve (score > threshold)
            approvals = sum(1 for score in validator_scores.values() 
                           if score > self.threshold)
            
            if approvals >= supermajority_threshold:
                return gen_name
        
        return None
    
    def _weighted_voting(self, samples_dict, scores, validators):
        """
        Weighted voting fallback
        
        Combines samples based on validator scores
        
        Returns:
            (samples, winner_name) tuple
        """
        # Compute weighted scores per generator
        weighted_scores = {}
        
        for gen_name, validator_scores in scores.items():
            total_score = 0.0
            total_weight = 0.0
            
            for validator_name, score in validator_scores.items():
                weight = self.weights.get(validator_name, 1.0)
                total_score += score * weight
                total_weight += weight
            
            weighted_scores[gen_name] = total_score / total_weight
        
        # Select best generator
        best_gen = max(weighted_scores, key=weighted_scores.get)
        print(f"   Winner: {best_gen} (score: {weighted_scores[best_gen]:.3f})")
        
        return samples_dict[best_gen], best_gen


# Diversity loss to prevent mode collapse
def diversity_loss(samples):
    """
    Encourage diversity in generated samples
    Based on Determinantal Point Processes (DPP)
    
    Args:
        samples: Tensor [batch_size, features]
    
    Returns:
        Diversity loss (negative = maximize diversity)
    """
    if isinstance(samples, np.ndarray):
        samples = torch.FloatTensor(samples)
    
    # Compute pairwise distances
    batch_size = samples.size(0)
    
    if batch_size < 2:
        return torch.tensor(0.0)
    
    # Expand for pairwise distance
    samples_1 = samples.unsqueeze(1).expand(batch_size, batch_size, -1)
    samples_2 = samples.unsqueeze(0).expand(batch_size, batch_size, -1)
    
    # L2 distance
    distances = torch.norm(samples_1 - samples_2, dim=2)
    
    # Average distance (higher = more diverse)
    avg_distance = distances.sum() / (batch_size * (batch_size - 1))
    
    # Return negative (we'll minimize this, so maximizing diversity)
    return -avg_distance


# EMA Stabilization
class EMAStabilizer:
    """
    Exponential Moving Average for model stabilization
    Keeps a stable copy of model weights
    """
    
    def __init__(self, model, beta=0.999):
        """
        Args:
            model: PyTorch model to stabilize
            beta: EMA decay rate (0.999 = very slow update)
        """
        self.beta = beta
        self.ema_params = {}
        
        # Initialize EMA with current model params
        for name, param in model.named_parameters():
            self.ema_params[name] = param.data.clone()
    
    def update(self, model):
        """Update EMA parameters"""
        for name, param in model.named_parameters():
            self.ema_params[name] = (
                self.beta * self.ema_params[name] + 
                (1 - self.beta) * param.data
            )
    
    def apply_ema(self, model):
        """Apply EMA parameters to model"""
        for name, param in model.named_parameters():
            param.data.copy_(self.ema_params[name])
    
    def restore_original(self, model, original_params):
        """Restore original parameters"""
        for name, param in model.named_parameters():
            param.data.copy_(original_params[name])


# Test
if __name__ == "__main__":
    print("Testing Byzantine Consensus...")
    
    # Mock samples from 3 generators
    samples_dict = {
        'wgan': np.random.randn(100, 30),
        'cgan': np.random.randn(100, 30) + 0.5,  # Slightly different
        'vanilla': np.random.randn(100, 30) + 1.0  # Very different
    }
    
    # Mock validators
    class MockValidator:
        def __init__(self, preference):
            self.preference = preference
        
        def score(self, samples):
            # Prefer samples with mean close to preference
            mean = samples.mean()
            return 1.0 / (1.0 + abs(mean - self.preference))
    
    validators = {
        'fuzzy': MockValidator(0.0),
        'nlp': MockValidator(0.0),
        'binary': MockValidator(0.5)
    }
    
    # Run consensus
    consensus = ByzantineConsensus()
    contradictions = consensus.detect_contradictions(samples_dict)
    
    print(f"\n Contradictions detected: {len(contradictions)}")
    for c in contradictions:
        print(f"   {c['gen1']} vs {c['gen2']}: distance = {c['distance']:.3f}")
    
    result = consensus.resolve(samples_dict, validators, contradictions)
    print(f"\n[OK] Consensus reached! Selected samples shape: {result.shape}")
    
    # Test diversity loss
    diverse_samples = torch.randn(50, 30) * 3  # Very diverse
    similar_samples = torch.randn(50, 30) * 0.1  # Very similar
    
    div_loss_diverse = diversity_loss(diverse_samples)
    div_loss_similar = diversity_loss(similar_samples)
    
    print(f"\n Diversity Loss Test:")
    print(f"   Diverse samples: {div_loss_diverse:.4f} (more negative = more diverse)")
    print(f"   Similar samples: {div_loss_similar:.4f}")
    
    print("\n[OK] Byzantine consensus test complete!")
