"""
📊 Comprehensive Logging System for Great Convergence
Captures all metrics, battles, and performance data
"""

import json
import logging
import os
from datetime import datetime

class ConvergenceLogger:
    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f'convergence_{timestamp}.json')
        
        self.data = {
            'timestamp': timestamp,
            'rounds': [],
            'battles': [],
            'hall_of_fame': [],
            'final_results': {},
            'convergence_stats': {}
        }
    
    def log_round(self, round_num, metrics):
        """Log metrics for each round"""
        self.data['rounds'].append({
            'round': round_num,
            'f1': metrics.get('f1', 0),
            'precision': metrics.get('precision', 0),
            'recall': metrics.get('recall', 0),
            'pr_auc': metrics.get('pr_auc', 0),
            'optimal_threshold': metrics.get('threshold', 0.5),
            'total_warriors': metrics.get('warriors', 0),
            'timestamp': datetime.now().isoformat()
        })
    
    def log_battle(self, battle_data):
        """Log East vs West battles"""
        self.data['battles'].append({
            'round': battle_data.get('round'),
            'winner': battle_data.get('winner'),
            'arena_f1': battle_data.get('arena_f1', 0),
            'kurukshetra_f1': battle_data.get('kurukshetra_f1', 0),
            'margin': battle_data.get('margin', 0)
        })
    
    def log_final_results(self, results):
        """Log final cascade and test results"""
        self.data['final_results'] = {
            'cascade_f1': results.get('cascade_f1', 0),
            'test_f1': results.get('test_f1', 0),
            'test_precision': results.get('test_precision', 0),
            'test_recall': results.get('test_recall', 0),
            'confusion_matrix': results.get('confusion_matrix', {}),
            'total_time': results.get('total_time', 0)
        }
    
    def save(self):
        """Save all logs to JSON"""
        with open(self.log_file, 'w') as f:
            json.dump(self.data, f, indent=2)
        print(f"📊 Comprehensive logs saved to: {self.log_file}")
        return self.log_file


class TenSeedsAggregator:
    """Aggregate results across 10 seeds"""
    
    def __init__(self):
        self.all_results = []
    
    def add_result(self, seed, f1, precision, recall):
        self.all_results.append({
            'seed': seed,
            'f1': f1,
            'precision': precision,
            'recall': recall
        })
    
    def get_statistics(self):
        import numpy as np
        
        f1s = [r['f1'] for r in self.all_results]
        
        return {
            'mean_f1': float(np.mean(f1s)),
            'std_f1': float(np.std(f1s)),
            'min_f1': float(np.min(f1s)),
            'max_f1': float(np.max(f1s)),
            'all_results': self.all_results
        }
    
    def print_summary(self):
        stats = self.get_statistics()
        print("\n" + "="*70)
        print("📊 10-SEEDS AGGREGATE RESULTS")
        print("="*70)
        print(f"\n🎯 F1 Statistics:")
        print(f"   Mean: {stats['mean_f1']:.4f} ({stats['mean_f1']*100:.2f}%)")
        print(f"   Std:  {stats['std_f1']:.4f}")
        print(f"   Min:  {stats['min_f1']:.4f}")
        print(f"   Max:  {stats['max_f1']:.4f}")
        print(f"   Range: {stats['max_f1'] - stats['min_f1']:.4f}")
        
        print(f"\n🎲 Individual Seeds:")
        for r in stats['all_results']:
            print(f"   Seed {r['seed']}: F1={r['f1']:.4f}, P={r['precision']:.4f}, R={r['recall']:.4f}")
        
        if stats['mean_f1'] >= 0.88:
            print("\n🎉 EXCELLENT! Mean F1 >= 88%")
        elif stats['mean_f1'] >= 0.85:
            print("\n✅ GOOD! Mean F1 >= 85%")
        else:
            print(f"\n⚠️  Mean F1: {stats['mean_f1']:.4f}")


if __name__ == "__main__":
    # Example usage
    logger = ConvergenceLogger()
    aggregator = TenSeedsAggregator()
    
    # After each seed completes:
    # aggregator.add_result(seed=42, f1=0.8840, precision=0.9639, recall=0.8163)
    # 
    # After all 10 seeds:
    # aggregator.print_summary()
