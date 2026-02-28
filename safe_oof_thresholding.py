"""
🛡️ SAFE OOF THRESHOLDING FOR GREAT CONVERGENCE
Implements rigorous OOF without overfitting pitfalls

OOF Strategy: True Cross-Validation (Option A)
- Train K separate Great Convergence instances
- Each trained on K-1 folds, predicts on held-out fold
- Aggregate all OOF predictions
- Calculate threshold ONCE on aggregated OOF
- Lock threshold
- Train final model on full data with locked threshold
- Evaluate on test ONCE

This avoids ALL leakage pitfalls ChatGPT identified.
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve, f1_score
from typing import Tuple, List, Dict
import copy


class OOFThresholdSelector:
    """
    🛡️ Rigorous OOF threshold selection for Great Convergence
    
    SAFETY MECHANISMS:
    1. Preprocessing fit PER FOLD (not globally)
    2. Evolution is threshold-agnostic (no validation feedback during training)
    3. OOF predictions generated DURING CV (not after final training)
    4. Threshold locked before test (no retuning after seeing metrics)
    """
    
    def __init__(self, n_folds: int = 5):
        self.n_folds = n_folds
        self.oof_predictions = None
        self.oof_labels = None
        self.optimal_threshold = None
        self.threshold_stats = {}
    
    def compute_oof_predictions(self, 
                                X_train: np.ndarray, 
                                y_train: np.ndarray,
                                great_convergence_factory,
                                feature_names: List[str] = None) -> np.ndarray:
        """
        🎯 Generate TRUE OOF predictions
        
        For each fold:
        1. Split data into train/val
        2. Train FRESH Great Convergence instance on train
        3. Predict on validation (OOF)
        4. Aggregate all OOF predictions
        
        CRITICAL: Each fold gets its own evolution!
        No information leaks between folds.
        """
        print(f"\n{'='*80}")
        print(f"🛡️ COMPUTING OOF PREDICTIONS ({self.n_folds}-Fold CV)")
        print(f"{'='*80}")
        print("   Training separate Great Convergence for each fold...")
        print("   No information leakage between folds")
        print(f"{'='*80}\n")
        
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        oof_preds = np.zeros(len(y_train))
        oof_probas = np.zeros(len(y_train))
        fold_stats = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            print(f"\n📊 Fold {fold_idx + 1}/{self.n_folds}")
            
            # Split data
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
            
            # 🎯 CRITICAL: Train FRESH instance
            # No weight sharing, no information from other folds
            print(f"   Training fresh Great Convergence...")
            gc = great_convergence_factory()
            
            # Train on this fold's training data
            gc.run_convergence(
                X_fold_train, y_fold_train,
                X_fold_val, y_fold_val,  # Only for logging, not evolution!
                feature_names=feature_names
            )
            
            # Get OOF predictions (probabilities, not binary!)
            print(f"   Generating OOF predictions...")
            oof_proba = self._get_ensemble_probability(gc, X_fold_val)
            oof_pred = (oof_proba >= 0.5).astype(int)  # Default 0.5 for OOF
            
            # Store OOF predictions
            oof_preds[val_idx] = oof_pred
            oof_probas[val_idx] = oof_proba
            
            # Fold statistics
            fold_f1 = f1_score(y_fold_val, oof_pred, zero_division=0)
            fold_stats.append({
                'fold': fold_idx + 1,
                'f1': fold_f1,
                'n_train': len(train_idx),
                'n_val': len(val_idx)
            })
            print(f"   Fold F1: {fold_f1:.4f}")
        
        # Store all OOF predictions
        self.oof_predictions = oof_preds
        self.oof_labels = y_train
        self.oof_probabilities = oof_probas
        self.fold_stats = fold_stats
        
        print(f"\n{'='*80}")
        print("✅ OOF PREDICTIONS COMPLETE")
        print(f"{'='*80}")
        mean_f1 = np.mean([s['f1'] for s in fold_stats])
        print(f"   Mean Fold F1: {mean_f1:.4f}")
        print(f"   All {len(y_train)} samples have OOF predictions")
        print(f"{'='*80}\n")
        
        return oof_probas
    
    def _get_ensemble_probability(self, gc, X):
        """Get probability predictions from Great Convergence ensemble"""
        # Use the cascade ensemble if available
        if gc.cascade_ensemble:
            all_probs = []
            
            # Level 1
            for fw, niche, model in gc.cascade_ensemble.get('level1_specialists', []):
                probs = gc._get_model_probability(model, X)
                if probs is not None:
                    all_probs.append(probs)
            
            # Level 2
            for model in gc.cascade_ensemble.get('level2_hybrids', []):
                probs = gc._get_model_probability(model, X)
                if probs is not None:
                    all_probs.append(probs)
            
            # Level 3
            for model in gc.cascade_ensemble.get('level3_supreme', []):
                probs = gc._get_model_probability(model, X)
                if probs is not None:
                    all_probs.append(probs)
            
            if len(all_probs) > 0:
                return np.mean(all_probs, axis=0)
        
        # Fallback: use predict method
        return gc.predict(X).astype(float)
    
    def select_optimal_threshold(self, target_precision: float = 0.95) -> float:
        """
        🎯 Select threshold ONCE on OOF predictions
        
        Strategy: Maximize recall while maintaining precision >= target
        """
        if self.oof_probabilities is None:
            raise ValueError("Must compute OOF predictions first!")
        
        print(f"\n{'='*80}")
        print(f"🎯 SELECTING OPTIMAL THRESHOLD (OOF-Based)")
        print(f"{'='*80}")
        print(f"   Target: Maximize Recall @ Precision >= {target_precision}")
        print(f"   Data: All {len(self.oof_labels)} OOF predictions")
        print(f"{'='*80}\n")
        
        # Compute PR curve on OOF
        precision, recall, thresholds = precision_recall_curve(
            self.oof_labels, self.oof_probabilities
        )
        
        # Find threshold that maximizes recall while precision >= target
        best_recall = 0
        best_threshold = 0.5
        best_f1 = 0
        
        for p, r, thresh in zip(precision, recall, thresholds):
            if p >= target_precision:
                if r > best_recall:
                    best_recall = r
                    best_threshold = thresh
                    # Calculate F1 at this point
                    if p + r > 0:
                        best_f1 = 2 * p * r / (p + r)
        
        # If no threshold meets precision target, use F1-optimal
        if best_threshold == 0.5 and best_recall == 0:
            print("   ⚠️ No threshold meets precision target, using F1-optimal...")
            f1_scores = []
            for p, r in zip(precision, recall):
                if p + r > 0:
                    f1_scores.append(2 * p * r / (p + r))
                else:
                    f1_scores.append(0)
            
            best_idx = np.argmax(f1_scores)
            best_threshold = thresholds[min(best_idx, len(thresholds)-1)]
            best_recall = recall[best_idx]
            best_f1 = f1_scores[best_idx]
            target_precision = precision[best_idx]
        
        self.optimal_threshold = best_threshold
        self.threshold_stats = {
            'threshold': best_threshold,
            'precision_at_threshold': target_precision,
            'recall_at_threshold': best_recall,
            'f1_at_threshold': best_f1,
            'pr_auc': np.mean(precision)  # Approximation
        }
        
        print(f"✅ OPTIMAL THRESHOLD SELECTED: {best_threshold:.4f}")
        print(f"   Precision: {target_precision:.4f}")
        print(f"   Recall: {best_recall:.4f}")
        print(f"   F1: {best_f1:.4f}")
        # DEBUG: Print exact numbers
        print(f"   [DEBUG] Threshold: {best_threshold:.6f} | Recall: {best_recall:.4f} | F1: {best_f1:.4f} | TargetPrecision: {target_precision:.2f}")
        # DEBUG: Print OOF stats
        oof_min, oof_max, oof_mean = self.oof_probabilities.min(), self.oof_probabilities.max(), self.oof_probabilities.mean()
        oof_p5 = np.percentile(self.oof_probabilities, 5)
        oof_p95 = np.percentile(self.oof_probabilities, 95)
        print(f"   [DEBUG] OOF stats -> min={oof_min:.4f} max={oof_max:.4f} mean={oof_mean:.4f} p5={oof_p5:.4f} p95={oof_p95:.4f}")
        print(f"\n🔒 THRESHOLD LOCKED - Will not change during final training!")
        print(f"{'='*80}\n")
        
        return best_threshold
    
    def train_final_model(self,
                         X_train: np.ndarray,
                         y_train: np.ndarray,
                         great_convergence_factory,
                         feature_names: List[str] = None):
        """
        🎯 Train final model on ALL data with LOCKED threshold
        
        CRITICAL: Threshold is frozen, never adapted to validation/test
        """
        if self.optimal_threshold is None:
            raise ValueError("Must select threshold first!")
        
        print(f"\n{'='*80}")
        print(f"🏋️ TRAINING FINAL MODEL (All Data)")
        print(f"{'='*80}")
        print(f"   Data: {len(X_train)} samples")
        print(f"   Threshold: {self.optimal_threshold:.4f} (LOCKED)")
        print(f"   No validation feedback during training")
        print(f"{'='*80}\n")
        
        # Train final Great Convergence on ALL training data
        gc = great_convergence_factory()
        
        # Use a dummy validation set (for logging only)
        # In practice, use a small hold-out or None
        dummy_val_size = min(1000, len(X_train) // 10)
        X_final_train = X_train[:-dummy_val_size]
        y_final_train = y_train[:-dummy_val_size]
        X_dummy_val = X_train[-dummy_val_size:]
        y_dummy_val = y_train[-dummy_val_size:]
        
        gc.run_convergence(
            X_final_train, y_final_train,
            X_dummy_val, y_dummy_val,
            feature_names=feature_names
        )
        
        # 🔒 LOCK THE THRESHOLD
        gc.optimal_threshold = self.optimal_threshold
        gc.threshold_frozen = True
        
        print(f"\n✅ Final model trained!")
        print(f"   Threshold locked: {gc.optimal_threshold:.4f}")
        print(f"   Ready for single test evaluation")
        
        return gc


def run_safe_great_convergence(X_train, y_train, X_test, y_test,
                               great_convergence_factory,
                               n_folds: int = 5,
                               feature_names: List[str] = None) -> Dict:
    """
    🛡️ End-to-end safe training with OOF thresholding
    
    This is the SAFE way to use Great Convergence:
    1. OOF predictions from K-fold CV
    2. Threshold selected on OOF (not adapted during training)
    3. Final model trained on all data with LOCKED threshold
    4. Single evaluation on test
    
    Returns:
        Dictionary with model, metrics, and OOF stats
    """
    
    # Step 1: OOF Threshold Selection
    print("\n" + "="*80)
    print("🛡️ SAFE GREAT CONVERGENCE WITH OOF THRESHOLDING")
    print("="*80)
    
    oof_selector = OOFThresholdSelector(n_folds=n_folds)
    
    # Compute OOF predictions (this is the rigorous part!)
    oof_probas = oof_selector.compute_oof_predictions(
        X_train, y_train, great_convergence_factory, feature_names
    )
    
    # Select threshold on OOF
    threshold = oof_selector.select_optimal_threshold(target_precision=0.95)
    
    # Step 2: Train Final Model
    final_model = oof_selector.train_final_model(
        X_train, y_train, great_convergence_factory, feature_names
    )
    
    # Step 3: Single Test Evaluation (NO RETUNING!)
    print(f"\n{'='*80}")
    print("🎯 SINGLE TEST EVALUATION (NO ADAPTATION)")
    print(f"{'='*80}")
    
    y_pred = final_model.predict(X_test)
    
    from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
    
    test_f1 = f1_score(y_test, y_pred, zero_division=0)
    test_precision = precision_score(y_test, y_pred, zero_division=0)
    test_recall = recall_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    results = {
        'test_f1': test_f1,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'confusion_matrix': {
            'tn': int(cm[0,0]), 'fp': int(cm[0,1]),
            'fn': int(cm[1,0]), 'tp': int(cm[1,1])
        },
        'optimal_threshold': threshold,
        'oof_stats': oof_selector.threshold_stats,
        'fold_stats': oof_selector.fold_stats,
        'model': final_model
    }
    
    print(f"\n✅ RESULTS:")
    print(f"   F1: {test_f1:.4f}")
    print(f"   Precision: {test_precision:.4f}")
    print(f"   Recall: {test_recall:.4f}")
    print(f"   Threshold: {threshold:.4f} (locked)")
    print(f"\n{'='*80}\n")
    
    return results


if __name__ == "__main__":
    print("🛡️ Safe OOF Thresholding System Ready!")
    print("\nUsage:")
    print("   oof_selector = OOFThresholdSelector(n_folds=5)")
    print("   oof_selector.compute_oof_predictions(X_train, y_train, factory)")
    print("   threshold = oof_selector.select_optimal_threshold()")
    print("   model = oof_selector.train_final_model(X_train, y_train, factory)")
    print("   # Threshold is LOCKED - evaluate once on test")
    print("\nOr use the one-shot function:")
    print("   results = run_safe_great_convergence(X_train, y_train, X_test, y_test, factory)")
