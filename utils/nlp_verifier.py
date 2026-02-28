"""
NLP Verifier - Semantic Fraud Pattern Verification
Converts numerical transactions to natural language descriptions
Uses BERT for semantic fraud detection
Novel component for tabular data validation
"""

import torch
import numpy as np

# Optional BERT support
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("[Warning]  transformers not installed - NLP will use rule-based scoring only")
    
import warnings
warnings.filterwarnings('ignore')


def transaction_to_text(transaction, feature_names=None):
    """
    Convert numerical transaction to natural language description
    
    Args:
        transaction: Array or dict of transaction features [30]
        feature_names: Optional list of feature names
    
    Returns:
        Natural language description string
    """
    # Handle different input types
    if isinstance(transaction, dict):
        features = transaction
    elif isinstance(transaction, (np.ndarray, list)):
        # Assume creditcard.csv format: Time, V1-V28, Amount
        if len(transaction) == 30:
            features = {
                'Time': transaction[0],
                'Amount': transaction[29],
                **{f'V{i}': transaction[i] for i in range(1, 29)}
            }
        else:
            raise ValueError(f"Expected 30 features, got {len(transaction)}")
    else:
        features = {f'F{i}': v for i, v in enumerate(transaction)}
    
    # Extract key features
    time = features.get('Time', 0)
    amount = features.get('Amount', 0)
    
    # Convert time (seconds) to hour
    hour = int((time // 3600) % 24)
    
    # Time of day description
    if 0 <= hour < 6:
        time_desc = "late night"
    elif 6 <= hour < 12:
        time_desc = "morning"
    elif 12 <= hour < 18:
        time_desc = "afternoon"
    elif 18 <= hour < 22:
        time_desc = "evening"
    else:
        time_desc = "night"
    
    # Amount description
    if amount < 50:
        amount_desc = "small"
    elif amount < 200:
        amount_desc = "moderate"
    elif amount < 1000:
        amount_desc = "large"
    else:
        amount_desc = "very large"
    
    # Build description
    description = f"A {amount_desc} transaction of ${amount:.2f} during {time_desc} ({hour}:00). "
    
    # Add V-feature patterns (PCA components)
    unusual_features = []
    
    for i in range(1, 29):
        v_key = f'V{i}'
        if v_key in features:
            v_value = features[v_key]
            
            # Flag unusual patterns (>2 std deviations)
            if abs(v_value) > 2.0:
                unusual_features.append(f"V{i}")
            
            # Specific high-impact features
            if i == 4 and abs(v_value) > 2:
                description += "Unusual V4 pattern (high deviation). "
            if i == 14 and abs(v_value) > 2:
                description += "Suspicious V14 signature. "
            if i == 12 and abs(v_value) > 2:
                description += "Anomalous V12 behavior. "
    
    # Overall pattern summary
    if len(unusual_features) > 5:
        description += f"Multiple unusual features detected ({len(unusual_features)} anomalies). "
    elif len(unusual_features) > 2:
        description += f"Some unusual patterns found. "
    
    # Risk indicators
    if (amount > 500 and (hour < 6 or hour > 22)):
        description += "High-risk combination: large amount during unusual hours. "
    
    return description.strip()


class NLPValidator:
    """
    NLP-based fraud validator using BERT
    
    Note: For production, you would fine-tune BERT on fraud descriptions.
    For this implementation, we use a lightweight approach with rule-based scoring.
    """
    
    def __init__(self, use_bert=False, device='cuda'):
        """
        Args:
            use_bert: If True, use actual BERT model (requires fine-tuning)
            device: 'cuda' or 'cpu'
        """
        self.use_bert = use_bert
        self.device = device
        
        if use_bert:
            # Load pre-trained BERT (would need fine-tuning for real fraud detection)
            print("[Warning]  Loading BERT model (this is a placeholder - needs fraud fine-tuning)")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    "distilbert-base-uncased", 
                    num_labels=2
                ).to(device)
                self.model.eval()
                print("[OK] BERT loaded (note: not fine-tuned for fraud)")
            except Exception as e:
                print(f"[Warning]  Could not load BERT: {e}")
                print("   Falling back to rule-based NLP scoring")
                self.use_bert = False
    
    def score(self, samples):
        """
        Score samples using NLP analysis
        
        Args:
            samples: Array of transactions [N, 30] or transaction descriptions
        
        Returns:
            Average fraud likelihood score (0-1)
        """
        # Handle different input formats
        if isinstance(samples, np.ndarray):
            # Numpy array - convert each row to description
            if samples.ndim == 1:
                # Single transaction [30] - reshape to [1, 30]
                samples = samples.reshape(1, -1)
            descriptions = [transaction_to_text(t) for t in samples]
        elif isinstance(samples, list):
            # Check if list of strings or list of arrays
            if len(samples) > 0 and isinstance(samples[0], str):
                # List of descriptions
                descriptions = samples
            else:
                # List of arrays/transactions
                descriptions = [transaction_to_text(t) for t in samples]
        else:
            # Single transaction
            descriptions = [transaction_to_text(samples)]
        
        if self.use_bert:
            return self._score_with_bert(descriptions)
        else:
            return self._score_with_rules(descriptions)
    
    def _score_with_bert(self, descriptions):
        """Score descriptions using BERT"""
        scores = []
        
        for desc in descriptions:
            # Tokenize
            inputs = self.tokenizer(
                desc,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
                fraud_prob = probs[0, 1].item()  # Probability of fraud class
            
            scores.append(fraud_prob)
        
        return np.mean(scores)
    
    def _score_with_rules(self, descriptions):
        """Rule-based scoring (lightweight alternative to BERT)"""
        scores = []
        
        # Fraud indicators
        fraud_keywords = [
            ('unusual', 0.2),
            ('suspicious', 0.3),
            ('anomalous', 0.3),
            ('high-risk', 0.4),
            ('late night', 0.2),
            ('very large', 0.3),
            ('multiple unusual', 0.4),
            ('high deviation', 0.3)
        ]
        
        for desc in descriptions:
            score = 0.5  # Base score
            desc_lower = desc.lower()
            
            # Add score for each keyword
            for keyword, weight in fraud_keywords:
                if keyword in desc_lower:
                    score += weight
            
            # Normalize to [0, 1]
            score = min(1.0, score)
            scores.append(score)
        
        return np.mean(scores)
    
    def validate_transaction(self, transaction):
        """
        Validate single transaction with explanation
        
        Args:
            transaction: Single transaction [30]
        
        Returns:
            (score, description, explanation)
        """
        description = transaction_to_text(transaction)
        score = self.score([transaction])
        
        # Generate explanation
        if score > 0.8:
            explanation = "HIGH FRAUD RISK: Multiple strong fraud indicators detected"
        elif score > 0.6:
            explanation = "MODERATE RISK: Some suspicious patterns found"
        else:
            explanation = "LOW RISK: Transaction appears normal"
        
        return score, description, explanation


def fine_tune_bert_on_fraud(train_data, train_labels, epochs=3, lr=2e-5):
    """
    Fine-tune BERT on fraud detection task
    
    Args:
        train_data: List of transaction descriptions (text)
        train_labels: List of fraud labels (0/1)
        epochs: Training epochs
        lr: Learning rate
    
    Returns:
        Fine-tuned NLPValidator
    """
    print("="*60)
    print("FINE-TUNING BERT FOR FRAUD DETECTION")
    print("="*60)
    print(f"Training samples: {len(train_data)}")
    print(f"Epochs: {epochs}")
    print("="*60)
    
    # Initialize
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2
    )
    
    # Training loop (simplified - would need full PyTorch training in production)
    print("\n[Warning]  Note: This is a placeholder for BERT fine-tuning")
    print("   In production, implement full training loop with:")
    print("   - DataLoader")
    print("   - AdamW optimizer")
    print("   - Learning rate scheduler")
    print("   - Validation set")
    print("   - Early stopping")
    
    # Return validator with "fine-tuned" model (actually not trained in this demo)
    validator = NLPValidator(use_bert=False)  # Use rule-based for now
    print("\n[OK] NLP Validator ready (rule-based mode)")
    
    return validator


# Test
if __name__ == "__main__":
    print("Testing NLP Verifier...")
    
    # Test transaction to text
    sample_transaction = np.random.randn(30)
    sample_transaction[0] = 85000  # Time (23:36)
    sample_transaction[29] = 750.0  # Amount
    
    description = transaction_to_text(sample_transaction)
    print(f"\n Transaction Description:")
    print(f"   {description}")
    
    # Test NLP validator
    validator = NLPValidator(use_bert=False)
    score = validator.score(np.random.randn(10, 30))
    print(f"\n Fraud Score: {score:.3f}")
    
    # Validate single transaction
    score, desc, explanation = validator.validate_transaction(sample_transaction)
    print(f"\n🔍 Single Transaction Validation:")
    print(f"   Score: {score:.3f}")
    print(f"   {explanation}")
    print(f"   Description: {desc}")
    
    print("\n[OK] NLP Verifier test complete!")
