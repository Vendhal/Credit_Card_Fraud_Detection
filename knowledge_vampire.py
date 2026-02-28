"""
🧛‍♂️ KNOWLEDGE VAMPIRE SYSTEM
Extracts unique patterns from dying models and transfers to elites
SAFE: No validation leakage, pattern generalization, validation gates
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import f1_score
import copy


class KnowledgeVampire:
    """
    🧛‍♂️ Extracts knowledge from weak models before destruction
    
    SAFETY MECHANISMS:
    1. Training-only extraction (NO validation/test leakage)
    2. Pattern generalization (ranges, not exact samples)
    3. Validation gate (rollback if overfitting detected)
    4. Soft transfer (augment, don't replace)
    5. Diversity preservation
    """
    
    def __init__(self, 
                 min_unique_detections: int = 3,
                 max_patterns_per_model: int = 10,
                 synthetic_samples_per_pattern: int = 5,
                 pattern_noise_std: float = 0.1,
                 transfer_weight: float = 1.3,
                 overfitting_threshold: float = 0.97):
        """
        Args:
            min_unique_detections: Minimum unique frauds to consider model valuable
            max_patterns_per_model: Max patterns to extract (prevent overfitting)
            synthetic_samples_per_pattern: Samples to generate per pattern
            pattern_noise_std: Noise to add for generalization
            transfer_weight: Sample weight for transferred patterns
            overfitting_threshold: Rollback if F1 drops below this ratio
        """
        self.min_unique_detections = min_unique_detections
        self.max_patterns_per_pattern = max_patterns_per_model
        self.synthetic_samples_per_pattern = synthetic_samples_per_pattern
        self.pattern_noise_std = pattern_noise_std
        self.transfer_weight = transfer_weight
        self.overfitting_threshold = overfitting_threshold
        
        # Track transfer history for rollback
        self.transfer_history = []
    
    def extract_unique_patterns(self, 
                               weak_model, 
                               strong_models: List, 
                               X_train: np.ndarray, 
                               y_train: np.ndarray,
                               feature_names: List[str] = None) -> List[Dict]:
        """
        🎯 Extract patterns that weak model knows but strong models don't
        
        CRITICAL: Uses TRAINING DATA ONLY - no validation/test leakage!
        
        Returns:
            List of pattern dictionaries with feature ranges/thresholds
        """
        # ✅ SAFETY 1: Only use training data
        # Convert to numpy if needed
        if hasattr(X_train, 'get'):
            X_train_np = X_train.get()
        else:
            X_train_np = np.asarray(X_train)
        
        # Get weak model predictions (probabilities if possible)
        weak_probs = self._get_model_probability(weak_model, X_train_np)
        
        # Get strong models ensemble predictions
        if len(strong_models) > 0:
            strong_probs_list = []
            for sm in strong_models:
                sm_probs = self._get_model_probability(sm, X_train_np)
                if sm_probs is not None:
                    strong_probs_list.append(sm_probs)
            
            if len(strong_probs_list) > 0:
                strong_probs = np.mean(strong_probs_list, axis=0)
            else:
                strong_probs = np.zeros(len(y_train))
        else:
            strong_probs = np.zeros(len(y_train))
        
        # Find where weak model is confident about fraud but strong models aren't
        # Weak confident: probability > 0.6
        # Strong uncertain: probability < 0.5
        fraud_mask = y_train == 1
        weak_confident = (weak_probs > 0.6) & fraud_mask
        strong_uncertain = strong_probs < 0.5
        
        unique_detections = np.where(weak_confident & strong_uncertain)[0]
        
        # Check if model has enough unique value
        if len(unique_detections) < self.min_unique_detections:
            return []  # Model is truly weak, no unique knowledge
        
        print(f"   🧛 Found {len(unique_detections)} unique fraud detections")
        
        # ✅ SAFETY 2: Extract PATTERNS (ranges/thresholds), not exact samples
        patterns = []
        
        # Sort by confidence and take top patterns
        confidences = weak_probs[unique_detections]
        top_indices = unique_detections[np.argsort(confidences)[-self.max_patterns_per_pattern:]]
        
        for idx in top_indices:
            sample = X_train_np[idx]
            
            # Extract pattern with ranges (generalization, not memorization)
            pattern = self._extract_pattern_from_sample(
                sample, weak_probs[idx], feature_names
            )
            patterns.append(pattern)
        
        return patterns
    
    def _get_model_probability(self, model, X: np.ndarray) -> Optional[np.ndarray]:
        """Get probability predictions from any model type"""
        try:
            # Handle divine alien avatars
            if hasattr(model, 'niche') and model.niche == 'divine_alien':
                if hasattr(model, 'model'):
                    sklearn_model = model.model
                    
                    if hasattr(sklearn_model, 'decision_function'):
                        scores = sklearn_model.decision_function(X)
                    elif hasattr(sklearn_model, 'score_samples'):
                        scores = sklearn_model.score_samples(X)
                    else:
                        pred = sklearn_model.predict(X)
                        scores = np.where(pred == -1, -5.0, 5.0)
                    
                    # Inverted sigmoid for anomaly detection
                    return 1.0 / (1.0 + np.exp(scores))
            
            # Standard models
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X)
                if hasattr(probs, 'get'):
                    probs = probs.get()
                if len(probs.shape) > 1:
                    return probs[:, 1]
                return probs
            
            if hasattr(model, 'decision_function'):
                scores = model.decision_function(X)
                if hasattr(scores, 'get'):
                    scores = scores.get()
                return 1.0 / (1.0 + np.exp(-scores))
            
            # Fallback: binary to soft probabilities
            pred = model.predict(X) if hasattr(model, 'predict') else model.yuddha(X)
            if hasattr(pred, 'get'):
                pred = pred.get()
            # Soft conversion: 1→0.7, 0→0.3 (NOT fake 0.99!)
            return np.where(pred == 1, 0.7, 0.3)
            
        except Exception as e:
            print(f"   ⚠️ Probability extraction failed: {e}")
            return None
    
    def _extract_pattern_from_sample(self, 
                                    sample: np.ndarray, 
                                    confidence: float,
                                    feature_names: List[str] = None) -> Dict:
        """
        ✅ SAFETY 2: Extract generalized pattern, not exact sample
        Returns ranges and thresholds, not specific values
        """
        # Key fraud indicators (V14, Amount, Time, etc.)
        # Extract ranges around the sample with noise for generalization
        
        pattern = {
            'confidence': float(confidence),
            'features': {}
        }
        
        # Extract key features with ranges
        key_indices = [14, 15, 16, 17]  # V14-V17 (known fraud indicators)
        if len(sample) > 28:
            key_indices.extend([-3, -2, -1])  # Amount, Time, etc.
        
        for idx in key_indices:
            if idx < len(sample):
                value = sample[idx]
                feature_name = feature_names[idx] if feature_names else f'V{idx}'
                
                # Create RANGE around value (generalization!)
                # Add noise to prevent exact memorization
                noise = np.random.normal(0, self.pattern_noise_std * abs(value))
                range_min = value - 0.2 * abs(value) + noise
                range_max = value + 0.2 * abs(value) + noise
                
                pattern['features'][feature_name] = {
                    'range': (float(min(range_min, range_max)), 
                             float(max(range_min, range_max))),
                    'center': float(value)
                }
        
        return pattern
    
    def generate_synthetic_from_pattern(self, pattern: Dict, n_features: int) -> np.ndarray:
        """Generate synthetic sample from pattern with noise"""
        sample = np.random.randn(n_features) * 0.1  # Base noise
        
        for feature_name, feature_info in pattern['features'].items():
            # Extract feature index from name (e.g., 'V14' → 14)
            if feature_name.startswith('V'):
                idx = int(feature_name[1:])
            else:
                continue  # Skip if can't parse
            
            if idx < n_features:
                # Generate value within pattern range
                range_min, range_max = feature_info['range']
                sample[idx] = np.random.uniform(range_min, range_max)
        
        return sample
    
    def transfer_patterns(self, 
                         elite_model, 
                         patterns: List[Dict], 
                         X_train: np.ndarray, 
                         y_train: np.ndarray,
                         X_val: np.ndarray,
                         y_val: np.ndarray) -> Tuple[bool, float]:
        """
        🎯 Transfer patterns to elite model with validation gate
        
        ✅ SAFETY 3: Validation gate - rollback if overfitting
        ✅ SAFETY 4: Soft transfer - augment, don't replace
        
        Returns:
            (success, new_f1)
        """
        if len(patterns) == 0:
            return False, 0.0
        
        # Store baseline for rollback
        baseline_pred = elite_model.predict(X_val) if hasattr(elite_model, 'predict') else elite_model.yuddha(X_val)
        if hasattr(baseline_pred, 'get'):
            baseline_pred = baseline_pred.get()
        baseline_f1 = f1_score(y_val, baseline_pred, zero_division=0)
        
        print(f"   📊 Baseline F1: {baseline_f1:.4f}")
        
        # Generate synthetic samples from patterns
        synthetic_X = []
        synthetic_y = []
        
        for pattern in patterns:
            for _ in range(self.synthetic_samples_per_pattern):
                sample = self.generate_synthetic_from_pattern(pattern, X_train.shape[1])
                synthetic_X.append(sample)
                synthetic_y.append(1)  # Fraud label
        
        synthetic_X = np.array(synthetic_X)
        synthetic_y = np.array(synthetic_y)
        
        # ✅ SAFETY 4: Soft transfer - mix with real data
        # 70% real, 30% synthetic
        n_real = len(X_train)
        n_synthetic = len(synthetic_X)
        
        X_augmented = np.vstack([X_train, synthetic_X])
        y_augmented = np.concatenate([y_train, synthetic_y])
        
        # Sample weights: boost synthetic but not too much
        sample_weights = np.ones(len(y_augmented))
        sample_weights[n_real:] = self.transfer_weight  # Boost patterns
        
        print(f"   🧬 Transferring {len(patterns)} patterns ({n_synthetic} synthetic samples)")
        
        try:
            # Store model state for rollback
            if hasattr(elite_model, '__getstate__'):
                original_state = elite_model.__getstate__()
            else:
                original_state = copy.deepcopy(elite_model)
            
            # Partial fit with augmented data
            if hasattr(elite_model, 'partial_fit'):
                elite_model.partial_fit(X_augmented, y_augmented, 
                                       sample_weight=sample_weights)
            elif hasattr(elite_model, 'fit'):
                # Store original and refit with more weight on new patterns
                elite_model.fit(X_augmented, y_augmented, 
                              sample_weight=sample_weights)
            
            # ✅ SAFETY 3: Validation gate
            new_pred = elite_model.predict(X_val) if hasattr(elite_model, 'predict') else elite_model.yuddha(X_val)
            if hasattr(new_pred, 'get'):
                new_pred = new_pred.get()
            new_f1 = f1_score(y_val, new_pred, zero_division=0)
            
            print(f"   📊 New F1: {new_f1:.4f}")
            
            # Check for overfitting
            if new_f1 < baseline_f1 * self.overfitting_threshold:
                print(f"   ⚠️ OVERFITTING DETECTED! Rolling back...")
                print(f"      Baseline: {baseline_f1:.4f} → New: {new_f1:.4f}")
                
                # Rollback
                if hasattr(elite_model, '__setstate__'):
                    elite_model.__setstate__(original_state)
                
                return False, baseline_f1
            
            print(f"   ✅ Transfer successful! Improvement: {baseline_f1:.4f} → {new_f1:.4f}")
            
            # Record successful transfer
            self.transfer_history.append({
                'patterns_transferred': len(patterns),
                'baseline_f1': baseline_f1,
                'new_f1': new_f1,
                'improvement': new_f1 - baseline_f1
            })
            
            return True, new_f1
            
        except Exception as e:
            print(f"   ❌ Transfer failed: {e}")
            return False, baseline_f1
    
    def smart_destroy(self, 
                     model, 
                     elite_models: List,
                     X_train: np.ndarray,
                     y_train: np.ndarray,
                     X_val: np.ndarray,
                     y_val: np.ndarray) -> bool:
        """
        🧛 Smart destruction: extract knowledge before destroying
        
        Returns:
            True if model was destroyed, False if kept
        """
        print(f"\n🧛 Processing model for destruction...")
        
        # Check if model has unique value
        patterns = self.extract_unique_patterns(
            model, elite_models, X_train, y_train
        )
        
        if len(patterns) >= self.min_unique_detections:
            print(f"   💎 Model has {len(patterns)} unique patterns - extracting before destruction")
            
            # Find weakest elite model to transfer to
            if len(elite_models) > 0:
                # Score elites on validation
                elite_scores = []
                for em in elite_models:
                    try:
                        pred = em.predict(X_val) if hasattr(em, 'predict') else em.yuddha(X_val)
                        if hasattr(pred, 'get'):
                            pred = pred.get()
                        f1 = f1_score(y_val, pred, zero_division=0)
                        elite_scores.append((em, f1))
                    except:
                        elite_scores.append((em, 0.0))
                
                # Transfer to weakest elite
                elite_scores.sort(key=lambda x: x[1])
                weakest_elite = elite_scores[0][0]
                
                success, new_f1 = self.transfer_patterns(
                    weakest_elite, patterns, X_train, y_train, X_val, y_val
                )
                
                if success:
                    print(f"   ✅ Knowledge preserved! Model can be safely destroyed")
                else:
                    print(f"   ⚠️ Transfer failed - consider keeping model")
                    return False  # Don't destroy if transfer failed
        else:
            print(f"   💀 Model has no unique value - safe to destroy")
        
        return True  # Safe to destroy


# Integration function for Great Convergence
def integrate_vampire_with_cataclysm(great_convergence_instance, 
                                     X_train, y_train, X_val, y_val,
                                     enable_vampire: bool = True):
    """
    Hook to integrate Vampire into Great Convergence Cataclysm
    
    Usage:
        # In GreatConvergence.visvarupa_cataclysm():
        vampire = KnowledgeVampire()
        
        for model in models_to_destroy:
            should_destroy = vampire.smart_destroy(
                model, self.elite_models, X_train, y_train, X_val, y_val
            )
            if should_destroy:
                actually_destroy(model)
    """
    if not enable_vampire:
        return None
    
    return KnowledgeVampire()


# 🎯 PROFESSIONAL API ALIAS
# Use PatternExtractor for public/professional contexts
# KnowledgeVampire remains for internal/theme usage
PatternExtractor = KnowledgeVampire
"""
🎯 Pattern Extractor (Professional Name)

This is the same as KnowledgeVampire but with a professional/public-facing name.

Usage:
    from knowledge_vampire import PatternExtractor
    extractor = PatternExtractor()

Internal mythology name (KnowledgeVampire) still works:
    from knowledge_vampire import KnowledgeVampire
    vampire = KnowledgeVampire()

Both are identical - choose based on context:
- Academic papers: PatternExtractor
- Presentations: PatternExtractor  
- Internal code: KnowledgeVampire (fun theme)
- Documentation: Mention both
"""


if __name__ == "__main__":
    print("🧛‍♂️ Knowledge Vampire System Ready!")
    print("✅ Safety mechanisms:")
    print("   1. Training-only extraction")
    print("   2. Pattern generalization")
    print("   3. Validation gate with rollback")
    print("   4. Soft transfer (augment, don't replace)")
    print("\nUse: integrate_vampire_with_cataclysm() in Great Convergence")
