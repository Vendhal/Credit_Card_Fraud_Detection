"""
Complete Evaluation Script - Tests GAN quality and detection performance

Run this after training to get comprehensive evaluation report.
"""

import numpy as np
import pickle
from utils.metrics import (
    GANQualityMetrics, FraudDetectionMetrics, HybridSystemMetrics,
    MetricsVisualizer, print_metrics_report
)
from data.data_loader import load_tabular_data, normalize_features


def evaluate_complete_system():
    """Run complete evaluation of GAN + Hybrid Detection System"""
    
    print("="*80)
    print("[INFO] COMPLETE SYSTEM EVALUATION")
    print("="*80)
    
    # ========== LOAD DATA ==========
    print("\n[INFO] Loading data...")
    data = load_tabular_data('Dataset/creditcard.csv', label_column='Class')
    data = normalize_features(data, fit_on='all_train')
    
    # ========== LOAD MODELS ==========
    print("[INFO] Loading trained models...")
    
    try:
        # Load GAN - Try Hive synthetic data first
        synthetic_pos = np.load('outputs/hive_synthetic.npy')
        print(f"[SUCCESS] Loaded {len(synthetic_pos)} HIVE synthetic samples")
    except FileNotFoundError:
        print("[Warning] Hive synthetic data not found, trying vanilla...")
        try:
            synthetic_pos = np.load('outputs/vanilla_synthetic.npy')
            print(f"[SUCCESS] Loaded {len(synthetic_pos)} VANILLA synthetic samples")
        except FileNotFoundError:
            print("[ERROR] Neither Hive nor Vanilla synthetic data found. Train GAN first!")
            return
        except Exception as e:
            print(f"[ERROR] Error loading vanilla synthetic data: {e}. Train GAN first!")
            return
    except Exception as e:
        print(f"[ERROR] Error loading hive synthetic data: {e}. Train GAN first!")
        return
    
    
    # Load ML classifier - Try GAN-specific models first
    classifier_loaded = False
    classifier_type = "unknown"
    
    try:
        # Try Hive GAN classifier first (NEW folder structure)
        with open('checkpoints/hive/rf_classifier_hive.pkl', 'rb') as f:
            rf_classifier = pickle.load(f)
        print("[SUCCESS] Loaded Hive GAN classifier (checkpoints/hive/rf_classifier_hive.pkl)")
        classifier_type = "Hive GAN"
        classifier_loaded = True
    except FileNotFoundError:
        try:
            # Try Vanilla GAN classifier (NEW folder structure)
            with open('checkpoints/vanilla/rf_classifier_vanilla.pkl', 'rb') as f:
                rf_classifier = pickle.load(f)
            print("[SUCCESS] Loaded Vanilla GAN classifier (checkpoints/vanilla/rf_classifier_vanilla.pkl)")
            classifier_type = "Vanilla GAN"
            classifier_loaded = True
        except FileNotFoundError:
            try:
                # Fallback 1: Try old flat structure - Hive
                with open('checkpoints/rf_classifier_hive.pkl', 'rb') as f:
                    rf_classifier = pickle.load(f)
                print("[SUCCESS] Loaded Hive GAN classifier (LEGACY: checkpoints/rf_classifier_hive.pkl)")
                classifier_type = "Hive GAN"
                classifier_loaded = True
            except FileNotFoundError:
                print("[ERROR] No RF classifier found. Train models first!")
                print("       Run: python main.py --use-vanilla  OR  python main.py --use-hive")
                return
    
    if not classifier_loaded:
        print("[ERROR] Failed to load classifier")
        return
    
    try:
        # Load hybrid detector
        with open('checkpoints/hybrid_detector.pkl', 'rb') as f:
            hybrid_detector = pickle.load(f)
        print("[SUCCESS] Loaded Hybrid Detector")
    except:
        print("[ERROR] Hybrid detector not found.")
        hybrid_detector = None
    
    # ========== EVALUATE GAN QUALITY ==========
    print("\n" + "="*80)
    print("[1] GAN QUALITY EVALUATION")
    print("="*80)
    
    real_pos = data['pos_test'][:len(synthetic_pos)]  # Match sizes
    
    # MMD
    print("\n[INFO] Computing Maximum Mean Discrepancy...")
    mmd = GANQualityMetrics.maximum_mean_discrepancy(real_pos, synthetic_pos)
    print(f"   MMD (RBF kernel): {mmd:.6f} {'[SUCCESS] Good' if mmd < 0.1 else '[WARNING] Needs improvement'}")
    
    # Correlation distance
    print("\n[INFO] Computing Correlation Matrix Distance...")
    corr_dist = GANQualityMetrics.correlation_matrix_distance(real_pos, synthetic_pos)
    print(f"   Correlation Distance: {corr_dist:.6f} {'[SUCCESS] Good' if corr_dist < 5.0 else '[WARNING] Needs improvement'}")
    
    # Statistical similarity
    print("\n[INFO] Computing Statistical Similarity...")
    stats = GANQualityMetrics.statistical_similarity(real_pos, synthetic_pos)
    print_metrics_report(stats, "Statistical Comparison")
    
    # KS test
    print("\n[INFO] Running Kolmogorov-Smirnov Test...")
    ks_results = GANQualityMetrics.kolmogorov_smirnov_test(real_pos, synthetic_pos)
    print(f"   Average p-value: {ks_results['avg_p_value']:.4f}")
    print(f"   Features passed (p>0.05): {ks_results['features_passed']}/{ks_results['total_features']}")
    
    # Visualizations
    print("\n[INFO] Generating GAN quality visualizations...")
    MetricsVisualizer.plot_tsne_comparison(real_pos, synthetic_pos)
    MetricsVisualizer.plot_correlation_heatmap(real_pos, synthetic_pos)
    
    # ========== EVALUATE ML CLASSIFIER ==========
    print("\n" + "="*80)
    print("[2] ML CLASSIFIER EVALUATION")
    print("="*80)
    
    # Test set
    X_test = np.vstack([data['pos_test'], data['neg_test']])
    y_test = np.hstack([np.ones(len(data['pos_test'])), 
                        np.zeros(len(data['neg_test']))])
    
    # Predictions
    y_pred_ml = rf_classifier.predict(X_test)
    y_proba_ml = rf_classifier.predict_proba(X_test)[:, 1]
    
    # Metrics
    ml_metrics = FraudDetectionMetrics.compute_all_metrics(y_test, y_pred_ml, y_proba_ml)
    print_metrics_report(ml_metrics, "Random Forest Performance")
    
    # Cost analysis
    print("\n[INFO] Cost-Sensitive Analysis:")
    cost_metrics = FraudDetectionMetrics.cost_sensitive_evaluation(
        y_test, y_pred_ml, fn_cost=500, fp_cost=5
    )
    print_metrics_report(cost_metrics, "Financial Impact")
    
    # Detection at FPR
    print("\n[INFO] Detection Rate @ 2% FPR:")
    detection_metrics = FraudDetectionMetrics.detection_at_fpr(y_test, y_proba_ml, target_fpr=0.02)
    print(f"   Detection Rate: {detection_metrics['detection_rate']:.2%}")
    print(f"   Actual FPR: {detection_metrics['actual_fpr']:.2%}")
    print(f"   Threshold: {detection_metrics['threshold']:.4f}")
    
    # Visualizations
    print("\n[INFO] Generating ML classifier visualizations...")
    MetricsVisualizer.plot_confusion_matrix(y_test, y_pred_ml, 
                                           save_path='outputs/ml_confusion_matrix.png')
    MetricsVisualizer.plot_roc_pr_curves(y_test, y_proba_ml,
                                        save_path='outputs/ml_roc_pr_curves.png')
    
    # ========== EVALUATE HYBRID SYSTEM ==========
    if hybrid_detector is not None:
        print("\n" + "="*80)
        print("[3] HYBRID SYSTEM EVALUATION")
        print("="*80)
        
        # Get hybrid predictions
        y_proba_hybrid = np.array([hybrid_detector.predict(tx) for tx in X_test])
        y_pred_hybrid = (y_proba_hybrid > 0.5).astype(int)
        
        # Get fuzzy-only predictions
        y_proba_fuzzy = np.array([hybrid_detector.predict_fuzzy(tx) for tx in X_test])
        y_pred_fuzzy = (y_proba_fuzzy > 0.5).astype(int)
        
        # Compare all three
        print("\n[INFO] Comparing Pure ML vs Pure Fuzzy vs Hybrid...")
        comparison = HybridSystemMetrics.compare_models(
            y_test, y_pred_ml, y_pred_fuzzy, y_pred_hybrid,
            y_proba_ml, y_proba_fuzzy, y_proba_hybrid
        )
        
        print("\n[INFO] PERFORMANCE COMPARISON:")
        print("-"*70)
        print(f"{'Metric':<20} {'Pure ML':<15} {'Pure Fuzzy':<15} {'Hybrid':<15}")
        print("-"*70)
        
        for metric in ['f1_score', 'recall', 'precision', 'mcc']:
            ml_val = comparison['ml_only'][metric]
            fuzzy_val = comparison['fuzzy_only'][metric]
            hybrid_val = comparison['hybrid'][metric]
            print(f"{metric.replace('_', ' ').title():<20} {ml_val:<15.4f} {fuzzy_val:<15.4f} {hybrid_val:<15.4f}")
        
        print("-"*70)
        print(f"\n[SUCCESS] F1 Improvement over Pure ML: {comparison['improvements']['f1_improvement_over_ml']:+.2f}%")
        print(f"[SUCCESS] F1 Improvement over Pure Fuzzy: {comparison['improvements']['f1_improvement_over_fuzzy']:+.2f}%")
        
        # ML-Fuzzy correlation
        print("\n[INFO] ML-Fuzzy Score Correlation:")
        correlation = HybridSystemMetrics.ml_fuzzy_correlation(y_proba_ml, y_proba_fuzzy)
        print(f"   Pearson: {correlation['pearson_correlation']:.4f}")
        print(f"   Spearman: {correlation['spearman_correlation']:.4f}")
        print(f"   Interpretation: {correlation['interpretation']}")
        
        # Visualizations
        print("\n[INFO] Generating hybrid comparison visualizations...")
        MetricsVisualizer.plot_hybrid_comparison(comparison,
                                                save_path='outputs/hybrid_comparison.png')
        MetricsVisualizer.plot_confusion_matrix(y_test, y_pred_hybrid,
                                               save_path='outputs/hybrid_confusion_matrix.png')
    
    # ========== FINAL SUMMARY ==========
    print("\n" + "="*80)
    print("[INFO] EVALUATION SUMMARY")
    print("="*80)
    
    print("\n[SUCCESS] GAN Quality:")
    print(f"   - MMD: {mmd:.6f}")
    print(f"   - Correlation Distance: {corr_dist:.6f}")
    print(f"   - Statistical Similarity: Mean diff = {stats['mean_difference']:.6f}")
    
    print("\n[SUCCESS] ML Classifier:")
    print(f"   - F1-Score: {ml_metrics['f1_score']:.4f}")
    print(f"   - Recall: {ml_metrics['recall']:.4f}")
    print(f"   - Precision: {ml_metrics['precision']:.4f}")
    print(f"   - MCC: {ml_metrics['mcc']:.4f}")
    print(f"   - PR-AUC: {ml_metrics['pr_auc']:.4f}")
    
    if hybrid_detector:
        print("\n[SUCCESS] Hybrid System:")
        print(f"   - F1-Score: {comparison['hybrid']['f1_score']:.4f}")
        print(f"   - Improvement: {comparison['improvements']['f1_improvement_over_ml']:+.2f}% over ML")
    
    print("\n[INFO] All visualizations saved to: outputs/")
    print("="*80)


if __name__ == "__main__":
    evaluate_complete_system()
