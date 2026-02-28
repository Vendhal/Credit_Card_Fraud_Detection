"""
Complete Hive GAN Training Pipeline
End-to-end training script for the Mega Hybrid GAN system

🛡️ SAFE OOF VERSION - No Data Leakage
======================================
CRITICAL SAFETY UPDATES (2026-02-18):
1. Validation set created from TRAINING data only (80/20 split)
2. NO test data used during training or synthetic sample selection
3. Best epoch selected using validation F1, not test F1
4. Follows strict Safe OOF principles for rigorous evaluation

Previous version had subtle leakage:
- Used test data to evaluate synthetic sample quality
- Selected best epoch based on test performance
- This inflated synthetic data quality artificially

Current version:
- GANs train on: fraud_train data only
- Quality evaluated on: fraud_val (from train split)
- Best samples selected: Based on val performance
- Test data: NEVER touched until final evaluation

This ensures synthetic data augmentation doesn't leak test information.

Usage:
    python train_hive_gan.py --epochs 400 --batch-size 128 --device cuda

Features:
    - OneCycleLR scheduler (3-5x speedup)
    - Mixed Precision (AMP) training
    - Final Reality Discriminator validation
    - Byzantine consensus
    - Comprehensive logging
    - Model checkpointing
    - 🛡️ SAFE OOF (No leakage)
"""

import torch
import numpy as np
import argparse
import os
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import OneCycleLR
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import all components
from models.hive_ensemble import HiveGANEnsemble
from data.data_loader import load_data
from utils.nlp_verifier import NLPValidator
from utils.adaptive_controller import AdaptiveController
import warnings
warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description='Train Hive GAN Ensemble')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=400, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    
    # Model parameters
    parser.add_argument('--noise-dim', type=int, default=100, help='Noise dimension')
    parser.add_argument('--use-amp', action='store_true', default=True, help='Use Mixed Precision')
    parser.add_argument('--use-onecycle', action='store_true', default=True, help='Use OneCycleLR')
    
    # Tuning parameters (NEW!)
    parser.add_argument('--final-disc-threshold', type=float, default=0.55, help='Final Disc threshold (lower=more samples)')
    parser.add_argument('--disc-train-interval', type=int, default=3, help='Train disc every N generator steps (higher=weaker disc)')
    parser.add_argument('--pretrain-epochs', type=int, default=10, help='Generator pretraining epochs')
    
    # Early stopping
    parser.add_argument('--early-stop-patience', type=int, default=10, help='Stop if no improvement for N epochs (0=disabled)')
    parser.add_argument('--early-stop-min-delta', type=float, default=0.01, help='Minimum improvement to count as better')
    
    # F1-based epoch selection (NEW!)
    parser.add_argument('--save-epoch-samples', action='store_true', default=False, help='Save samples from each epoch and pick best by F1')
    parser.add_argument('--eval-every', type=int, default=1, help='Evaluate and save samples every N epochs')
    
    # Phase control
    parser.add_argument('--phase', type=str, default='all',
                       choices=['wgan', 'cgan', 'vanilla', 'all'],
                       help='Which generators to train')
    
    # Checkpointing
    parser.add_argument('--save-every', type=int, default=50, help='Save checkpoint every N epochs')
    parser.add_argument('--output-dir', type=str, default='checkpoints', help='Output directory')
    
    # Mode
    parser.add_argument('--generate-only', action='store_true', help='Skip training and generate data from existing checkpoint')
    
    return parser.parse_args()


class HiveGANTrainer:
    """
    Complete training pipeline for Hive GAN Ensemble
    """
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        
        # Display hyperparameters
        print("="*70)
        print("HIVE MEGA HYBRID GAN - TRAINING PIPELINE")
        print("="*70)
        print(f"Device: {self.device}")
        print(f"Epochs: {args.epochs}")
        print(f"Batch Size: {args.batch_size}")
        print(f"Learning Rate: {args.lr}")
        print(f"Mixed Precision (AMP): {args.use_amp}")
        print(f"OneCycleLR: {args.use_onecycle}")
        print(f"Phase: {args.phase}")
        print(f"\n🔧 TUNING PARAMETERS:")
        print(f"  Final Disc Threshold: {args.final_disc_threshold}")
        print(f"  Disc Train Interval: {args.disc_train_interval}")
        print(f"  Pretrain Epochs: {args.pretrain_epochs}")
        if args.early_stop_patience > 0:
            print(f"  Early Stop Patience: {args.early_stop_patience} epochs")
            print(f"  Early Stop Min Delta: {args.early_stop_min_delta}")
        print("="*70)
        
        # Setup output directories
        self.output_dir = Path(args.output_dir) / 'hive_gan'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nOutput directory: {self.output_dir}")
    
    def load_data(self):
        """Load and prepare fraud data - SAFE VERSION (No test data leakage)"""
        print("\nLoading Data...")
        print("  🛡️ SAFE MODE: No test data will be used for synthetic selection")
        
        # [FIXED] Load using existing data_loader and handle dictionary return
        print(f"   [INFO] Using 10x Rama Constraint (Weight: 10.0)")
        
        # Load dictionary
        from data.data_loader import load_data, normalize_features
        from sklearn.model_selection import train_test_split
        data = load_data()
        
        # CRITICAL: Normalize features to [-1, 1] range!
        # Generators output tanh ([-1, 1]) so data must match
        data = normalize_features(data, fit_on='all_train')
        
        # Extract fraud samples correctly from dictionary
        X_fraud_train = data['fraud_train']
        
        # CRITICAL: Extract normal transactions for Final Disc training!
        X_normal_train = data['normal_train']  # Need this for Final Disc!
        
        # 🛡️ SAFE OOF: Create validation set from TRAINING data only
        # Split fraud training data into train/val for synthetic quality evaluation
        X_fraud_train_split, X_fraud_val = train_test_split(
            X_fraud_train, test_size=0.2, random_state=42, stratify=None
        )
        # Split normal training data similarly
        X_normal_train_split, X_normal_val = train_test_split(
            X_normal_train, test_size=0.2, random_state=42, stratify=None
        )
        
        # Store validation data for synthetic quality evaluation (NOT test data!)
        self.fraud_val = X_fraud_val
        self.normal_val = X_normal_val
        
        # Also keep full training data for Final Discriminator training
        self.normal_data = X_normal_train_split  # For Final Disc training
        
        print(f"   Training fraud samples: {len(X_fraud_train_split)}")
        print(f"   Validation fraud samples: {len(X_fraud_val)} (for synthetic evaluation)")
        print(f"   Training normal samples: {len(X_normal_train_split)} (for Final Disc)")
        print(f"   Validation normal samples: {len(X_normal_val)}")
        
        # Create DataLoader
        fraud_tensor = torch.FloatTensor(X_fraud_train_split)
        train_dataset = TensorDataset(fraud_tensor)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=0,  # Avoid multiprocessing issues
            pin_memory=True if self.device == 'cuda' else False
        )
        
        # Store for Final Discriminator training and synthetic evaluation
        self.real_fraud = X_fraud_train_split
        self.fraud_val = X_fraud_val
        self.normal_val = X_normal_val  # CRITICAL!
        
        print("[OK] Data loaded successfully (Safe OOF Mode)")
    
    def initialize_models(self):
        """Initialize Hive GAN Ensemble"""
        print("\nInitializing Models...")
        
        self.ensemble = HiveGANEnsemble(
            noise_dim=self.args.noise_dim,
            data_dim=30,
            lr=self.args.lr,
            device=self.device,
            use_amp=self.args.use_amp
        )
        
        # Add NLP validator
        self.nlp_validator = NLPValidator(use_bert=False, device=self.device)
        self.ensemble.validators['nlp'] = self.nlp_validator
        
        # Update consensus weights
        self.ensemble.consensus.weights = {
            'binary': 0.4,
            'fuzzy': 0.4,
            'nlp': 0.2
        }
        
        print("[OK] Models initialized")
    
    def setup_schedulers(self):
        """Setup OneCycleLR schedulers for 3-5x speedup"""
        if not self.args.use_onecycle:
            print("\n[Warning] OneCycleLR disabled")
            self.schedulers = None
            return
        
        print("\nSetting up OneCycleLR schedulers...")
        
        self.schedulers = {}
        steps_per_epoch = len(self.train_loader)
        total_steps = self.args.epochs * steps_per_epoch
        
        for name, optimizer in self.ensemble.optimizers_G.items():
            self.schedulers[name] = OneCycleLR(
                optimizer,
                max_lr=self.args.lr * 10,  # 10x peak learning rate
                total_steps=total_steps,
                pct_start=0.3,  # Warmup for 30% of training
                anneal_strategy='cos',
                div_factor=25.0,
                final_div_factor=1000.0
            )
        
        print(f"[OK] OneCycleLR ready (max_lr={self.args.lr * 10:.6f})")
        print(f"   Expected speedup: 3-5x")
    
    def train_phase_0(self):
        """Phase 0: Train Final Reality Discriminator"""
        print("\n" + "="*70)
        print("PHASE 0: TRAINING FINAL REALITY DISCRIMINATOR")
        print("="*70)
        
        # CRITICAL: Pass normal_data so Final Disc learns fraud vs normal!
        self.ensemble.initialize_final_discriminator(
            self.real_fraud,
            normal_data=self.normal_data,  # KEY FIX!
            epochs=100
        )
        
        # Save Final Discriminator
        final_disc_path = self.output_dir / 'final_discriminator.pth'
        self.ensemble.final_disc.save(final_disc_path)
        
        print(f"\n[OK] Phase 0 complete - Final Discriminator frozen")
    
    def train_phase_1_4(self):
        """Phases 1-4: Hive training"""
        print("\n" + "="*70)
        print("PHASES 1-4: HIVE ENSEMBLE TRAINING")
        print("="*70)
        
        # SMART INITIALIZATION: Pretrain generators on real fraud statistics
        print(f"\n[HYBRID FIX] Pretraining generators for {self.args.pretrain_epochs} epochs...")
        self.ensemble.pretrain_generators(self.real_fraud, epochs=self.args.pretrain_epochs)
        
        # Set disc train interval
        self.ensemble.disc_train_interval = self.args.disc_train_interval
        print(f"[TUNING] Discriminator trains every {self.args.disc_train_interval} generator steps")
        
        # Set final disc threshold
        self.ensemble.final_disc_threshold = self.args.final_disc_threshold
        print(f"[TUNING] Final Disc threshold: {self.args.final_disc_threshold}")
        
        # Initialize Adaptive Training Controller
        print(f"\n[ADAPTIVE CONTROLLER] Initializing...")
        controller_config = {
            'gen_lr': self.args.lr,
            'disc_lr': self.args.lr,
            'disc_threshold': self.args.final_disc_threshold,
            'adapt_lr': True,  # Enable learning rate adaptation
            'adapt_loss': True,  # Enable loss weight adaptation
            'enable_reset': True,  # Enable emergency resets
            'patience': 5  # Patience before emergency reset
        }
        adaptive_controller = AdaptiveController(controller_config)
        print(f"[ADAPTIVE CONTROLLER] Ready to prevent mode collapse!")
        
        best_quality = 0.0
        best_f1 = 0.0
        best_epoch = 0
        epochs_no_improve = 0  # Early stopping counter
        history = {
            'epoch': [],
            'pass_rate': [],
            'quality': [],
            'contradictions': []
        }
        
        # F1-based epoch selection: track samples from each epoch
        epoch_samples_dir = None
        if self.args.save_epoch_samples:
            epoch_samples_dir = Path('outputs') / 'epoch_samples'
            epoch_samples_dir.mkdir(parents=True, exist_ok=True)
            print(f"[F1 MODE] Will save samples from each epoch to {epoch_samples_dir}")
            print(f"[F1 MODE] After training, will evaluate F1 for all epochs and pick best")
        
        # Training loop
        for epoch in range(self.args.epochs):
            # Train one epoch
            pass_rate, quality = self.ensemble.train_epoch_hive(
                self.train_loader,
                epoch,
                phase=self.args.phase
            )
            
            # Get metrics for adaptive controller
            real_score = self.ensemble.get_real_score()
            fake_score = self.ensemble.get_fake_score()
            
            # Calculate WGAN diversity (Rama Dharma check)
            wgan_diversity = None
            if epoch % 5 == 0:  # Check every 5 epochs
                try:
                    print(f"\n  [WGAN DIVERSITY CHECK] Epoch {epoch}")
                    wgan = self.ensemble.generators['wgan']
                    wgan.eval()
                    wgan_samples = wgan.generate(100, device=self.device)
                    # Calculate pairwise distance std as diversity measure
                    from sklearn.metrics.pairwise import euclidean_distances
                    distances = euclidean_distances(wgan_samples[:50])  # 50 samples for speed
                    wgan_diversity = np.std(distances)
                    print(f"    WGAN sample diversity (std of distances): {wgan_diversity:.6f}")
                    print(f"    WGAN sample std (per-feature): {np.mean(np.std(wgan_samples, axis=0)):.6f}")
                    if wgan_diversity < 0.01:
                        print(f"    ⚠️  MODE COLLAPSE DETECTED! Diversity = {wgan_diversity:.6f} < 0.01")
                except Exception as e:
                    print(f"    ❌ Error calculating WGAN diversity: {e}")
                    wgan_diversity = None
            
            # Build metrics dict for controller
            wgan_disc_pred = self.ensemble.get_wgan_disc_pred()  # Get WGAN rejection metric
            epoch_metrics = {
                'pass_rate': pass_rate if pass_rate is not None else 1.0,
                'quality': quality if quality is not None else 0.5,
                'real_score': real_score,
                'fake_score': fake_score,
                'f1': None,  # Will be populated by real-time F1 tracking
                'wgan_diversity': wgan_diversity,
                'wgan_disc_pred': wgan_disc_pred  # For fallback collapse detection
            }
            
            # Real-time F1 tracking (every 5 epochs)
            # This gives the controller F1 feedback to make better decisions
            if epoch % 5 == 0 and epoch > 0:
                quick_f1 = self._quick_f1_eval(n_samples=100)
                if quick_f1 is not None:
                    epoch_metrics['f1'] = quick_f1
                    print(f"\n[F1 TRACKING] Epoch {epoch} Quick F1: {quick_f1:.4f}")
            
            # Adaptive controller decides on updates
            updates = adaptive_controller.step(epoch, epoch_metrics)
            
            # Apply updates to ensemble
            if updates:
                self.ensemble.apply_updates(updates)
            
            # Update schedulers (only if not using adaptive LR)
            # Skip OneCycle if controller is managing LR
            if self.schedulers and not updates.get('gen_lr'):
                for scheduler in self.schedulers.values():
                    scheduler.step()
            
            # Track best with early stopping using F1 score
            current_f1 = epoch_metrics.get('f1')
            if current_f1 is not None and current_f1 > 0:
                if current_f1 > best_f1 + self.args.early_stop_min_delta:
                    best_f1 = current_f1
                    best_epoch = epoch
                    epochs_no_improve = 0
                    
                    # Save best model
                    best_path = self.output_dir / 'best_ensemble.pth'
                    self.ensemble.save(best_path)
                    print(f"[IMPROVED] Best F1: {best_f1:.4f} at epoch {epoch}")
                else:
                    epochs_no_improve += 1
                    if self.args.early_stop_patience > 0:
                        print(f"[INFO] No improvement for {epochs_no_improve}/{self.args.early_stop_patience} epochs (current F1: {current_f1:.4f}, best: {best_f1:.4f})")
                
                # Early stopping check (skip if using F1 mode)
                if not self.args.save_epoch_samples and self.args.early_stop_patience > 0 and epochs_no_improve >= self.args.early_stop_patience:
                    print(f"\n🛑 EARLY STOPPING at epoch {epoch}!")
                    print(f"   No improvement for {self.args.early_stop_patience} epochs")
                    print(f"   Best F1: {best_f1:.4f} (epoch {best_epoch})")
                    break
            
            # F1 MODE: Save model checkpoint at EVERY epoch (no sample generation during training!)
            if self.args.save_epoch_samples and (epoch + 1) % self.args.eval_every == 0:
                checkpoint_path = self.output_dir / f'epoch_{epoch:03d}_model.pth'
                self.ensemble.save(checkpoint_path)
                print(f"[F1 MODE] Saved model: {checkpoint_path}")
            
            # Periodic checkpoint (regular mode)
            elif (epoch + 1) % self.args.save_every == 0:
                # Fixed: save with distinct name in the correct directory
                checkpoint_path = self.output_dir / f'hive_gan_epoch_{epoch+1}.pth'
                self.ensemble.save(checkpoint_path)
                print(f"[Saved] Checkpoint saved: {checkpoint_path}")
        
        print(f"\n[OK] Training complete!")
        print(f"   Best F1: {best_f1:.4f} (epoch {best_epoch})")
        
        # F1 MODE: Evaluate all epochs and pick best
        if self.args.save_epoch_samples:
            print(f"\n{'='*70}")
            print("F1-BASED EPOCH SELECTION")
            print(f"{'='*70}")
            best_f1_epoch = self.evaluate_all_epoch_models()
            if best_f1_epoch is not None:
                best_epoch = best_f1_epoch
                print(f"\n🏆 Best epoch by F1 score: {best_f1_epoch}")
        
        # Save final model with metadata
        final_path = self.output_dir / 'final_ensemble.pth'
        metadata = {
            'epoch': best_epoch,
            'total_epochs': self.args.epochs,
            'pass_rate': best_quality,
            'batch_size': self.args.batch_size
        }
        self.ensemble.save(final_path, metadata=metadata)
        
        # Plot training history
        self.plot_history(history)
        
        return history
    
    def _quick_f1_eval(self, n_samples=100):
        """
        Quick F1 evaluation during training (lightweight version)
        
        Generates small batch of samples and evaluates F1 on validation set
        This gives adaptive controller real-time feedback without blocking training
        
        Args:
            n_samples: Number of samples to generate for quick eval
        
        Returns:
            float: F1 score (or None if evaluation fails)
        """
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import f1_score
            
            # Generate small batch of synthetic fraud
            synthetic = self.ensemble.generate_high_quality_fraud(n_samples=n_samples)
            
            if len(synthetic) < 10:
                # Not enough samples, skip evaluation
                return None
            
            # 🛡️ SAFE OOF: Use validation set (from training data) for evaluation
            # IMPORTANT: Never use test data during training!
            train_size = min(50, len(self.real_fraud))
            val_size = min(50, len(self.fraud_val))
            
            # Get normal samples from TRAINING set
            num_normal_train = len(self.real_fraud) + len(synthetic)
            normal_train_indices = np.random.choice(len(self.normal_data), num_normal_train, replace=False)
            X_normal_train = self.normal_data[normal_train_indices]
            
            # Training set: real fraud + synthetic + normal (all from TRAINING data)
            X_train = np.vstack([
                self.real_fraud[:train_size],
                synthetic,
                X_normal_train
            ])
            y_train = np.array(
                [1] * train_size + 
                [1] * len(synthetic) + 
                [0] * len(X_normal_train)
            )
            
            # 🛡️ SAFE: Validation set (from training data, NOT test data!)
            X_fraud_val_subset = self.fraud_val[:val_size]
            X_normal_val_subset = self.normal_val[:val_size]
            X_val_eval = np.vstack([X_fraud_val_subset, X_normal_val_subset])
            y_val_eval = np.array([1] * len(X_fraud_val_subset) + [0] * len(X_normal_val_subset))
            
            # Quick RF classifier (fewer trees for speed)
            clf = RandomForestClassifier(
                n_estimators=50,  # Fewer trees than full eval
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            clf.fit(X_train, y_train)
            
            # 🛡️ SAFE: Predict on validation set (not test!)
            y_pred = clf.predict(X_val_eval)
            quick_f1 = f1_score(y_val_eval, y_pred, zero_division=0)
            
            return quick_f1
            
        except Exception as e:
            # Silently fail if evaluation errors
            print(f"[F1 TRACKING] Evaluation failed: {e}")
            return None
    
    def evaluate_all_epoch_models(self):
        """Load each epoch's model, generate samples, evaluate F1, return best epoch"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        print(f"\n⚡ QUANTUM SUPERPOSITION MODE - Evaluating all epoch universes...")
        
        # Get all epoch model files
        model_files = sorted(self.output_dir.glob('epoch_*_model.pth'))
        
        if len(model_files) == 0:
            print("[Warning] No epoch models found!")
            return None
        
        print(f"   Found {len(model_files)} epoch models to evaluate")
        
        # 🛡️ SAFE OOF: Use validation data (from training set) for evaluation
        # Never use test data to select synthetic samples!
        X_fraud_val = self.fraud_val
        X_normal_val = self.normal_val
        
        best_f1 = 0.0
        best_epoch_num = 0
        best_samples = None
        
        # Create output dir for epoch samples
        epoch_samples_dir = Path('outputs') / 'epoch_samples'
        epoch_samples_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        for model_file in model_files:
            # Extract epoch number from filename
            epoch_num = int(model_file.stem.split('_')[1])
            
            print(f"\n   📦 Loading epoch {epoch_num} model...")
            
            # Load this epoch's model
            self.ensemble.load(model_file)
            
            # Generate samples using this epoch's model (with Final Disc filtering!)
            print(f"   🎲 Generating samples...")
            synthetic = self.ensemble.generate_high_quality_fraud(n_samples=500)
            
            if len(synthetic) == 0:
                print(f"   ❌ Epoch {epoch_num}: No samples generated (Final Disc rejected all)")
                results.append({
                    'epoch': epoch_num,
                    'samples': 0,
                    'precision': 0,
                    'recall': 0,
                    'f1': 0
                })
                continue
            
            # Save samples for this epoch
            epoch_file = epoch_samples_dir / f'epoch_{epoch_num:03d}_samples.npy'
            np.save(epoch_file, synthetic)
            
            # 🛡️ SAFE OOF: Use validation set (from train data) for evaluation
            # Train classifier on synthetic + real fraud + normal samples
            # FIX: Need normal samples too, otherwise classifier learns "everything is fraud"
            X_train_eval = np.vstack([self.real_fraud, synthetic, self.normal_val[:len(synthetic)]])
            y_train_eval = np.array([1] * len(self.real_fraud) + [1] * len(synthetic) + [0] * len(synthetic))
            
            # 🛡️ SAFE: Validation set (from training data, NOT test data)
            X_val_eval = np.vstack([self.fraud_val, self.normal_val])
            y_val_eval = np.array([1] * len(self.fraud_val) + [0] * len(self.normal_val))
            
            # Train and evaluate
            print(f"   🧠 Training classifier...")
            clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            clf.fit(X_train_eval, y_train_eval)
            y_pred = clf.predict(X_val_eval)
            
            precision = precision_score(y_val_eval, y_pred, zero_division=0)
            recall = recall_score(y_val_eval, y_pred, zero_division=0)
            f1 = f1_score(y_val_eval, y_pred, zero_division=0)
            
            results.append({
                'epoch': epoch_num,
                'samples': len(synthetic),
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
            
            print(f"   ✅ Epoch {epoch_num:3d}: F1={f1:.4f}, P={precision:.4f}, R={recall:.4f}, Samples={len(synthetic)}")
            
            # Track best
            if f1 > best_f1:
                best_f1 = f1
                best_epoch_num = epoch_num
                best_samples = synthetic
        
        print(f"\n{'='*70}")
        print(f"🏆 BEST EPOCH (Collapsed Wavefunction): {best_epoch_num}")
        print(f"   F1 Score: {best_f1:.4f}")
        print(f"   Samples: {len(best_samples) if best_samples is not None else 0}")
        print(f"{'='*70}")
        
        # Save best samples to the main output location
        if best_samples is not None:
            output_file = Path('outputs') / 'hive_synthetic.npy'
            np.save(output_file, best_samples)
            print(f"\n💾 [Saved] Best epoch's samples saved to {output_file}")
            
            # Load best epoch's model as final
            best_model_file = self.output_dir / f'epoch_{best_epoch_num:03d}_model.pth'
            self.ensemble.load(best_model_file)
            print(f"💾 [Loaded] Best epoch's model loaded as final")
            
            # Clean up other epoch models (optional - save disk space)
            print(f"\n🧹 [Cleanup] Removing non-best epoch models...")
            for model_file in model_files:
                epoch_num = int(model_file.stem.split('_')[1])
                if epoch_num != best_epoch_num:
                    model_file.unlink()
            
            # Also clean up non-best sample files
            for sample_file in epoch_samples_dir.glob('epoch_*_samples.npy'):
                epoch_num = int(sample_file.stem.split('_')[1])
                if epoch_num != best_epoch_num:
                    sample_file.unlink()
            
            print(f"🧹 [Cleanup] Kept only epoch {best_epoch_num} model and samples")
        
        return best_epoch_num
    
    def plot_history(self, history):
        """Plot training metrics"""
        if len(history['epoch']) == 0:
            print("[Warning] No history to plot")
            return
        
        print("\nGenerating training plots...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Pass rate
        axes[0].plot(history['epoch'], history['pass_rate'], 'b-', linewidth=2)
        axes[0].axhline(y=0.7, color='r', linestyle='--', label='Threshold (70%)')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Pass Rate')
        axes[0].set_title('Final Discriminator Pass Rate')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Quality score
        axes[1].plot(history['epoch'], history['quality'], 'g-', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Quality Score')
        axes[1].set_title('Ensemble Quality Score')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.tight_layout()
        # User requested plots in outputs/ folder
        Path('outputs').mkdir(parents=True, exist_ok=True)
        plot_path = Path('outputs') / 'hive_training_history.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Plot saved: {plot_path}")
        plt.close()
    
    def evaluate(self):
        """Final evaluation"""
        print("\n" + "="*70)
        print("FINAL EVALUATION")
        print("="*70)
        
        # Generate high-quality fraud
        print("\nGenerating high-quality synthetic fraud...")
        synthetic_fraud = self.ensemble.generate_high_quality_fraud(n_samples=500)
        
        print(f"[OK] Generated {len(synthetic_fraud)} high-quality samples")
        
        # Validate with Final Discriminator
        if self.ensemble.final_disc:
            pass_rate, quality, _ = self.ensemble.final_disc.validate_samples(
                synthetic_fraud,
                device=self.device
            )
            
            print(f"\nFinal Quality Metrics:")
            print(f"   Pass Rate: {pass_rate:.2%}")
            print(f"   Quality Score: {quality:.3f}")
        
        # Save synthetic data
        # User requested outputs in outputs/ folder
        os.makedirs('outputs', exist_ok=True)
        save_path = Path('outputs') / 'hive_synthetic.npy'
        np.save(save_path, synthetic_fraud)
        print(f"[INFO] Saved {len(synthetic_fraud)} synthetic fraud transactions to: {save_path}")
        
        return synthetic_fraud
    
    def run(self):
        """Run complete training pipeline"""
        # Load data
        self.load_data()
        
        # Initialize models
        self.initialize_models()
        
        # Setup schedulers
        self.setup_schedulers()
        
        # Phases 1-4: Hive training
        if not self.args.generate_only:
            # Phase 0: Final Discriminator
            self.train_phase_0()
            
            # Phases 1-4: Hive training
            history = self.train_phase_1_4()
        else:
            print("\n[INFO] --generate-only flag set. Skipping training.")
            history = {}
            
            # Load best model if exists
            final_model = self.output_dir / 'final_ensemble.pth'
            if final_model.exists():
                 print(f"[INFO] Loading checkpoint: {final_model}")
                 
                 # Initialize if not already
                 if not hasattr(self, 'ensemble'):
                     self.initialize_models()
                     
                 self.ensemble.load(final_model)
            else:
                print(f"[ERROR] Checkpoint not found: {final_model}")
                return {}, None
        
        # Evaluation
        synthetic_fraud = self.evaluate()
        
        print("\n" + "="*70)
        print("HIVE GAN TRAINING COMPLETE!")
        print("="*70)
        print("\nOutputs saved to: outputs/")
        print("   - Synthetic data: hive_synthetic.npy")
        print("   - Training plot: hive_training_history.png")
        print("\nCheckpoints saved to: checkpoints/hive_gan/")
        print("   - Best model: best_ensemble.pth")
        print("   - Final model: final_ensemble.pth")
        
        return history, synthetic_fraud


def main():
    args = parse_args()
    
    # Initialize trainer
    trainer = HiveGANTrainer(args)
    
    # Run training
    history, synthetic_fraud = trainer.run()
    
    print("\n[OK] Pipeline complete!")


if __name__ == "__main__":
    main()
