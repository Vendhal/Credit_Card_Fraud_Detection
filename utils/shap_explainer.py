"""
SHAP Explainer for ML Model Interpretability

This module provides SHAP-based explanations for the Random Forest classifier,
showing which features contribute most to fraud predictions.
"""

import numpy as np
import shap
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier


class SHAPExplainer:
    """SHAP explainer for fraud detection ML model"""
    
    def __init__(self, model, background_data=None, feature_names=None):
        """
        Initialize SHAP explainer
        
        Args:
            model: Trained sklearn model (Random Forest)
            background_data: Background dataset for SHAP (sample of training data)
            feature_names: List of feature names for clarity
        """
        self.model = model
        self.feature_names = feature_names or [f'Feature_{i}' for i in range(30)]
        
        # Initialize TreeExplainer (optimized for Random Forest)
        if background_data is not None:
            # Use background data for faster computation
            self.explainer = shap.TreeExplainer(model, data=background_data)
        else:
            self.explainer = shap.TreeExplainer(model)
        
        print(f"[SUCCESS] SHAP TreeExplainer initialized for {type(model).__name__}")
    
    def explain_instance(self, transaction, show_plot=False):
        """
        Generate SHAP explanation for a single transaction
        
        Args:
            transaction: Feature array (1D or 2D)
            show_plot: Whether to display waterfall plot
        
        Returns:
            dict with SHAP values and top contributing features
        """
        # Ensure 2D shape
        if transaction.ndim == 1:
            transaction = transaction.reshape(1, -1)
        
        # Compute SHAP values
        shap_values = self.explainer.shap_values(transaction)
        print(f"[DEBUG] shap_values type: {type(shap_values)}")
        if hasattr(shap_values, 'shape'):
            print(f"[DEBUG] shap_values shape: {shap_values.shape}")
        elif isinstance(shap_values, list):
             print(f"[DEBUG] shap_values list len: {len(shap_values)}")
        
        # For binary classification, take fraud class (class 1)
        if isinstance(shap_values, list):
            shap_values_fraud = shap_values[1][0]  # Class 1 (fraud)
        elif hasattr(shap_values, 'ndim') and shap_values.ndim == 3:
            # (1, features, classes) -> take class 1
            shap_values_fraud = shap_values[0, :, 1]
        else:
            shap_values_fraud = shap_values[0]
        
        # Get base value (expected model output)
        base_value = self.explainer.expected_value
        if isinstance(base_value, list):
             base_value = base_value[1]  # Fraud class
        elif hasattr(base_value, 'ndim') and base_value.ndim >= 1 and base_value.shape[0] > 1:
             # If scalar array for binary classification, take 2nd element (fraud)
             if base_value.shape[0] == 2:
                 base_value = base_value[1]
             else:
                 # Fallback
                 base_value = base_value[-1]
        
        # Get top contributing features
        top_features = self._get_top_features(shap_values_fraud, transaction[0], n=5)
        
        # Create waterfall plot if requested
        if show_plot:
            self._plot_waterfall(shap_values_fraud, transaction[0], base_value)
        
        return {
            'shap_values': shap_values_fraud,
            'base_value': base_value,
            'top_features': top_features,
            'prediction': base_value + np.sum(shap_values_fraud)  # Native numpy
        }
    
    def _get_top_features(self, shap_values, feature_values, n=5):
        """Get top N contributing features - FIXED VERSION"""
        # Get absolute SHAP values for ranking
        abs_shap = np.abs(shap_values)
        top_indices = np.argsort(abs_shap)[-n:][::-1]
        
        top_features = []
        for idx in top_indices:
            idx = int(idx)
            # ROBUST scalar extraction - handles all numpy types
            val = np.asarray(feature_values[idx]).flatten()[0]
            shp = np.asarray(shap_values[idx]).flatten()[0]
            
            top_features.append({
                'feature': self.feature_names[idx],
                'val_raw': val,  # Store raw for debug
                'value': float(val),
                'shap_value': float(shp),
                'contribution': 'increases' if float(shp) > 0 else 'decreases'
            })
        
        return top_features
    
    def _plot_waterfall(self, shap_values, feature_values, base_value):
        """Create SHAP waterfall plot"""
        # NOTE: Waterfall plots are not critical, skip for now to avoid complexity
        print("[INFO] Waterfall plot generation skipped")
        pass
    
    def plot_summary(self, X_test, max_display=20):
        """
        Create SHAP summary plot showing feature importance across all predictions
        
        Args:
            X_test: Test dataset
            max_display: Number of features to display
        """
        print("[INFO] Generating SHAP summary plot...")
        
        # Calculate SHAP values for test set
        shap_values = self.explainer.shap_values(X_test)
        
        # For binary classification
        if isinstance(shap_values, list):
            shap_values_plot = shap_values[1]  # Fraud class
        elif hasattr(shap_values, 'ndim') and shap_values.ndim == 3:
             # (samples, features, classes) -> take class 1
            shap_values_plot = shap_values[:, :, 1]
        else:
            shap_values_plot = shap_values
        
        # Create summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values_plot, X_test, 
                         feature_names=self.feature_names,
                         max_display=max_display, show=False)
        
        # Save plot
        os.makedirs('outputs', exist_ok=True)
        plt.tight_layout()
        plt.savefig('outputs/shap_summary.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("[SUCCESS] SHAP summary plot saved to outputs/shap_summary.png")
    
    def format_explanation(self, explanation_dict):
        """
        Format SHAP explanation as human-readable text
        
        Args:
            explanation_dict: Output from explain_instance()
        
        Returns:
            Formatted string explanation
        """
        top_features = explanation_dict['top_features']
        base = explanation_dict['base_value']
        pred = explanation_dict['prediction']
        
        lines = []
        lines.append("="*60)
        lines.append("SHAP EXPLANATION")
        lines.append("="*60)
        # Explicitly cast to float to avoid numpy formatting errors
        lines.append(f"Base value (average): {float(base):.4f}")
        lines.append(f"Final prediction: {float(pred):.4f}")
        lines.append(f"")
        lines.append("Top Contributing Features:")
        lines.append("-"*60)
        
        for i, feat in enumerate(top_features, 1):
            lines.append(f"{i}. {feat['feature']}")
            lines.append(f"   Value: {feat['value']:.4f}")
            lines.append(f"   SHAP: {feat['shap_value']:.4f} ({feat['contribution']} fraud risk)")
            lines.append("")
        
        return "\n".join(lines)


if __name__ == "__main__":
    # Test SHAP Explainer
    print("="*70)
    print("[TEST] Testing SHAP Explainer")
    print("="*70)
    
    # Create dummy Random Forest for testing
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=1000, n_features=30, n_informative=15, 
                               n_redundant=5, random_state=42)
    
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X[:800], y[:800])
    
    # Initialize SHAP explainer
    explainer = SHAPExplainer(rf, background_data=X[:100])
    
    # Explain single instance
    test_sample = X[900]
    explanation = explainer.explain_instance(test_sample, show_plot=False)
    
    print("\n" + explainer.format_explanation(explanation))
    
    # Create summary plot
    explainer.plot_summary(X[800:900], max_display=15)
    
    print("\n[SUCCESS] SHAP explainer test complete!")
