"""
Comprehensive Metrics Module for GAN + Fuzzy Fraud Detection

This module provides:
1. GAN Quality Metrics (MMD, correlation, KDE overlap)
2. Classification Metrics (PR-AUC, F1, F2, MCC, recall, precision)
3. Hybrid System Comparison
4. Visualization Generation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.metrics import (
    precision_recall_curve, auc, roc_curve, roc_auc_score,
    f1_score, precision_score, recall_score, confusion_matrix,
    matthews_corrcoef, classification_report
)
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')


class GANQualityMetrics:
    """Metrics for evaluating synthetic fraud quality"""
    
    @staticmethod
    def maximum_mean_discrepancy(X_real, X_synthetic, kernel='rbf', gamma=1.0):
        """
        Maximum Mean Discrepancy - measures distribution similarity
        
        Args:
            X_real: Real fraud samples
            X_synthetic: Synthetic fraud samples
            kernel: 'rbf' or 'linear'
            gamma: RBF kernel parameter
        
        Returns:
            MMD value (lower is better, 0 = identical distributions)
        """
        n_real = len(X_real)
        n_synth = len(X_synthetic)
        
        # Compute kernel matrices
        if kernel == 'rbf':
            # Real-Real
            K_RR = np.exp(-gamma * cdist(X_real, X_real, 'sqeuclidean'))
            # Synthetic-Synthetic
            K_SS = np.exp(-gamma * cdist(X_synthetic, X_synthetic, 'sqeuclidean'))
            # Real-Synthetic
            K_RS = np.exp(-gamma * cdist(X_real, X_synthetic, 'sqeuclidean'))
        else:  # linear kernel
            K_RR = X_real @ X_real.T
            K_SS = X_synthetic @ X_synthetic.T
            K_RS = X_real @ X_synthetic.T
        
        # MMD^2
        mmd_squared = (K_RR.sum() / (n_real * n_real) + 
                      K_SS.sum() / (n_synth * n_synth) - 
                      2 * K_RS.sum() / (n_real * n_synth))
        
        return np.sqrt(max(mmd_squared, 0))
    
    @staticmethod
    def correlation_matrix_distance(X_real, X_synthetic):
        """
        Frobenius norm of correlation matrix difference
        
        Returns:
            Distance value (lower is better)
        """
        corr_real = np.corrcoef(X_real.T)
        corr_synth = np.corrcoef(X_synthetic.T)
        
        # Frobenius norm
        distance = np.linalg.norm(corr_real - corr_synth, 'fro')
        
        return distance
    
    @staticmethod
    def statistical_similarity(X_real, X_synthetic):
        """
        Compare statistical properties (mean, std)
        
        Returns:
            Dict with mean/std differences
        """
        mean_real = X_real.mean(axis=0)
        mean_synth = X_synthetic.mean(axis=0)
        std_real = X_real.std(axis=0)
        std_synth = X_synthetic.std(axis=0)
        
        mean_diff = np.abs(mean_real - mean_synth).mean()
        std_diff = np.abs(std_real - std_synth).mean()
        
        return {
            'mean_difference': mean_diff,
            'std_difference': std_diff,
            'mean_real': mean_real.mean(),
            'mean_synthetic': mean_synth.mean(),
            'std_real': std_real.mean(),
            'std_synthetic': std_synth.mean()
        }
    
    @staticmethod
    def kolmogorov_smirnov_test(X_real, X_synthetic):
        """
        KS test for each feature
        
        Returns:
            Average p-value (>0.05 is good)
        """
        p_values = []
        for i in range(X_real.shape[1]):
            statistic, p_value = stats.ks_2samp(X_real[:, i], X_synthetic[:, i])
            p_values.append(p_value)
        
        return {
            'avg_p_value': np.mean(p_values),
            'features_passed': np.sum(np.array(p_values) > 0.05),
            'total_features': len(p_values)
        }


class FraudDetectionMetrics:
    """Classification metrics for fraud detection"""
    
    @staticmethod
    def compute_all_metrics(y_true, y_pred, y_pred_proba=None):
        """
        Compute comprehensive classification metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
        
        Returns:
            Dict with all metrics
        """
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Basic metrics
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # F2-Score (weights recall 2x)
        f2 = (5 * precision * recall) / (4 * precision + recall) if (precision + recall) > 0 else 0
        
        # Specificity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # False Positive Rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # Matthews Correlation Coefficient
        mcc = matthews_corrcoef(y_true, y_pred)
        
        metrics = {
            'confusion_matrix': {
                'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn)
            },
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1,
            'f2_score': f2,
            'fpr': fpr,
            'mcc': mcc
        }
        
        # If probabilities available, compute AUC metrics
        if y_pred_proba is not None:
            # ROC-AUC
            try:
                roc_auc = roc_auc_score(y_true, y_pred_proba)
                metrics['roc_auc'] = roc_auc
            except:
                metrics['roc_auc'] = None
            
            # PR-AUC (better for imbalanced data)
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba)
            pr_auc = auc(recall_curve, precision_curve)
            metrics['pr_auc'] = pr_auc
        
        return metrics
    
    @staticmethod
    def cost_sensitive_evaluation(y_true, y_pred, fn_cost=500, fp_cost=5):
        """
        Cost-sensitive accuracy (fraud loss is expensive)
        
        Args:
            fn_cost: Cost of missing fraud (default $500)
            fp_cost: Cost of false alarm (default $5)
        
        Returns:
            Total cost and cost per transaction
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        total_cost = (fn * fn_cost) + (fp * fp_cost)
        cost_per_transaction = total_cost / len(y_true)
        
        # Cost reduction vs baseline (predict all normal)
        baseline_cost = y_true.sum() * fn_cost
        cost_reduction = ((baseline_cost - total_cost) / baseline_cost * 100) if baseline_cost > 0 else 0
        
        return {
            'total_cost': total_cost,
            'cost_per_transaction': cost_per_transaction,
            'baseline_cost': baseline_cost,
            'cost_reduction_pct': cost_reduction,
            'fn_count': int(fn),
            'fp_count': int(fp),
            'fn_cost_total': fn * fn_cost,
            'fp_cost_total': fp * fp_cost
        }
    
    @staticmethod
    def detection_at_fpr(y_true, y_pred_proba, target_fpr=0.02):
        """
        Detection rate at specific FPR threshold
        
        Args:
            target_fpr: Target false positive rate (default 2%)
        
        Returns:
            Detection rate and threshold
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        
        # Find threshold closest to target FPR
        idx = np.argmin(np.abs(fpr - target_fpr))
        
        return {
            'detection_rate': tpr[idx],
            'actual_fpr': fpr[idx],
            'threshold': thresholds[idx]
        }


class HybridSystemMetrics:
    """Metrics for hybrid ML + Fuzzy system"""
    
    @staticmethod
    def compare_models(y_true, ml_pred, fuzzy_pred, hybrid_pred, 
                      ml_proba=None, fuzzy_proba=None, hybrid_proba=None):
        """
        Compare Pure ML vs Pure Fuzzy vs Hybrid
        
        Returns:
            Comparison dict showing improvement
        """
        ml_metrics = FraudDetectionMetrics.compute_all_metrics(y_true, ml_pred, ml_proba)
        fuzzy_metrics = FraudDetectionMetrics.compute_all_metrics(y_true, fuzzy_pred, fuzzy_proba)
        hybrid_metrics = FraudDetectionMetrics.compute_all_metrics(y_true, hybrid_pred, hybrid_proba)
        
        # Calculate improvements
        f1_improvement_ml = ((hybrid_metrics['f1_score'] - ml_metrics['f1_score']) / 
                             ml_metrics['f1_score'] * 100 if ml_metrics['f1_score'] > 0 else 0)
        
        f1_improvement_fuzzy = ((hybrid_metrics['f1_score'] - fuzzy_metrics['f1_score']) / 
                               fuzzy_metrics['f1_score'] * 100 if fuzzy_metrics['f1_score'] > 0 else 0)
        
        return {
            'ml_only': ml_metrics,
            'fuzzy_only': fuzzy_metrics,
            'hybrid': hybrid_metrics,
            'improvements': {
                'f1_improvement_over_ml': f1_improvement_ml,
                'f1_improvement_over_fuzzy': f1_improvement_fuzzy,
                'mcc_ml': ml_metrics['mcc'],
                'mcc_fuzzy': fuzzy_metrics['mcc'],
                'mcc_hybrid': hybrid_metrics['mcc']
            }
        }
    
    @staticmethod
    def ml_fuzzy_correlation(ml_scores, fuzzy_scores):
        """
        Measure correlation between ML and Fuzzy scores
        
        Returns:
            Pearson and Spearman correlations
        """
        pearson = np.corrcoef(ml_scores, fuzzy_scores)[0, 1]
        spearman, _ = stats.spearmanr(ml_scores, fuzzy_scores)
        
        return {
            'pearson_correlation': pearson,
            'spearman_correlation': spearman,
            'interpretation': 'High correlation' if abs(pearson) > 0.7 else 
                            'Moderate correlation' if abs(pearson) > 0.4 else 
                            'Low correlation (complementary signals)'
        }


class MetricsVisualizer:
    """Generate visualizations for metrics"""
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, save_path='outputs/confusion_matrix.png'):
        """Plot confusion matrix"""
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Fraud'],
                   yticklabels=['Normal', 'Fraud'])
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Confusion matrix saved to: {save_path}")
    
    @staticmethod
    def plot_roc_pr_curves(y_true, y_pred_proba, save_path='outputs/roc_pr_curves.png'):
        """Plot ROC and PR curves"""
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        ax1.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate', fontsize=12)
        ax1.set_ylabel('True Positive Rate', fontsize=12)
        ax1.set_title('ROC Curve', fontsize=14, fontweight='bold')
        ax1.legend(loc="lower right")
        ax1.grid(alpha=0.3)
        
        # PR Curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        ax2.plot(recall, precision, color='darkorange', lw=2,
                label=f'PR curve (AUC = {pr_auc:.3f})')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Recall', fontsize=12)
        ax2.set_ylabel('Precision', fontsize=12)
        ax2.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        ax2.legend(loc="lower left")
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[INFO] ROC and PR curves saved to: {save_path}")
    
    @staticmethod
    def plot_tsne_comparison(X_real, X_synthetic, save_path='outputs/tsne_comparison.png'):
        """Plot t-SNE visualization of real vs synthetic fraud"""
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        print("[INFO] Computing t-SNE (this may take a moment)...")
        
        # Combine data
        X_combined = np.vstack([X_real, X_synthetic])
        labels = np.array(['Real'] * len(X_real) + ['Synthetic'] * len(X_synthetic))
        
        # Compute t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_tsne = tsne.fit_transform(X_combined)
        
        # Plot
        plt.figure(figsize=(10, 8))
        for label, color, marker in [('Real', 'blue', 'o'), ('Synthetic', 'red', '^')]:
            mask = labels == label
            plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                       c=color, label=label, alpha=0.6, s=50, marker=marker)
        
        plt.title('t-SNE: Real vs Synthetic Fraud', fontsize=16, fontweight='bold')
        plt.xlabel('t-SNE Component 1', fontsize=12)
        plt.ylabel('t-SNE Component 2', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[INFO] t-SNE comparison saved to: {save_path}")
    
    @staticmethod
    def plot_correlation_heatmap(X_real, X_synthetic, save_path='outputs/correlation_comparison.png'):
        """Compare correlation matrices"""
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        corr_real = np.corrcoef(X_real.T)
        corr_synth = np.corrcoef(X_synthetic.T)
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        
        # Real
        sns.heatmap(corr_real, cmap='coolwarm', center=0, ax=ax1, 
                   cbar_kws={'label': 'Correlation'}, vmin=-1, vmax=1)
        ax1.set_title('Real Fraud Correlations', fontsize=14, fontweight='bold')
        
        # Synthetic
        sns.heatmap(corr_synth, cmap='coolwarm', center=0, ax=ax2,
                   cbar_kws={'label': 'Correlation'}, vmin=-1, vmax=1)
        ax2.set_title('Synthetic Fraud Correlations', fontsize=14, fontweight='bold')
        
        # Difference
        diff = np.abs(corr_real - corr_synth)
        sns.heatmap(diff, cmap='Reds', ax=ax3,
                   cbar_kws={'label': 'Absolute Difference'})
        ax3.set_title('Correlation Difference', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Correlation comparison saved to: {save_path}")
    
    @staticmethod
    def plot_hybrid_comparison(comparison_results, save_path='outputs/hybrid_comparison.png'):
        """Bar chart comparing Pure ML, Pure Fuzzy, Hybrid"""
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        ml = comparison_results['ml_only']
        fuzzy = comparison_results['fuzzy_only']
        hybrid = comparison_results['hybrid']
        
        metrics = ['f1_score', 'recall', 'precision', 'mcc']
        metric_labels = ['F1-Score', 'Recall', 'Precision', 'MCC']
        
        ml_scores = [ml[m] for m in metrics]
        fuzzy_scores = [fuzzy[m] for m in metrics]
        hybrid_scores = [hybrid[m] for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width, ml_scores, width, label='Pure ML', color='skyblue')
        ax.bar(x, fuzzy_scores, width, label='Pure Fuzzy', color='lightcoral')
        ax.bar(x + width, hybrid_scores, width, label='Hybrid (ML+Fuzzy)', color='lightgreen')
        
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Comparison: Pure ML vs Pure Fuzzy vs Hybrid', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontsize=11)
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.0])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Hybrid comparison saved to: {save_path}")


def print_metrics_report(metrics, title="Metrics Report"):
    """Pretty print metrics"""
    print("\n" + "="*70)
    print(f"[REPORT] {title}")
    print("="*70)
    
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"\n{key.upper().replace('_', ' ')}:")
            for k, v in value.items():
                if isinstance(v, (int, float)):
                    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
                else:
                    print(f"  {k}: {v}")
        elif isinstance(value, (int, float)):
            print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
        else:
            print(f"{key}: {value}")
    
    print("="*70)


if __name__ == "__main__":
    # Test metrics module
    print("="*70)
    print("[TEST] Testing Comprehensive Metrics Module")
    print("="*70)
    
    # Generate dummy data
    np.random.seed(42)
    n_samples = 1000
    
    # Simulated real vs synthetic fraud
    X_real = np.random.randn(n_samples, 30)
    X_synthetic = np.random.randn(n_samples, 30) * 0.95  # Slightly different
    
    # Simulated predictions
    y_true = np.random.binomial(1, 0.3, n_samples)
    y_pred_ml = np.random.binomial(1, 0.3, n_samples)
    y_pred_fuzzy = np.random.binomial(1, 0.3, n_samples)
    y_pred_hybrid = np.random.binomial(1, 0.3, n_samples)
    y_proba = np.random.rand(n_samples)
    
    # Test GAN metrics
    print("\n[1] GAN Quality Metrics:")
    mmd = GANQualityMetrics.maximum_mean_discrepancy(X_real, X_synthetic)
    print(f"   MMD: {mmd:.6f}")
    
    corr_dist = GANQualityMetrics.correlation_matrix_distance(X_real, X_synthetic)
    print(f"   Correlation Distance: {corr_dist:.6f}")
    
    stats_sim = GANQualityMetrics.statistical_similarity(X_real, X_synthetic)
    print(f"   Mean Difference: {stats_sim['mean_difference']:.6f}")
    
    # Test classification metrics
    print("\n[2] Classification Metrics:")
    metrics = FraudDetectionMetrics.compute_all_metrics(y_true, y_pred_ml, y_proba)
    print_metrics_report(metrics, "Classification Performance")
    
    # Test hybrid comparison
    print("\n[3] Hybrid System Comparison:")
    comparison = HybridSystemMetrics.compare_models(
        y_true, y_pred_ml, y_pred_fuzzy, y_pred_hybrid,
        y_proba, y_proba, y_proba
    )
    print(f"   F1 Improvement over ML: {comparison['improvements']['f1_improvement_over_ml']:.2f}%")
    
    # Test visualizations
    print("\n[4] Generating Visualizations...")
    MetricsVisualizer.plot_confusion_matrix(y_true, y_pred_ml)
    MetricsVisualizer.plot_roc_pr_curves(y_true, y_proba)
    
    print("\n[SUCCESS] Metrics module test complete!")
