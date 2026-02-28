"""
Adaptive Training Controller for Hive GAN

Monitors training metrics in real-time and adjusts hyperparameters
to prevent mode collapse, maintain quality, and optimize performance.

Key Features:
- Automatic learning rate adjustment
- Dynamic loss weight tuning
- Discriminator-generator balance monitoring
- Quality collapse detection and recovery
- Emergency reset mechanism
- WGAN Dharma Protection (Rama rule) - prevents WGAN from collapsing under harsh feedback
"""

import numpy as np
import torch


class AdaptiveController:
    """
    Real-time training optimization controller.
    
    Monitors key metrics and adjusts hyperparameters dynamically to:
    1. Prevent discriminator dominance (score gap monitoring)
    2. Prevent quality collapse (F1 tracking)
    3. Maintain healthy pass rates
    4. Balance generator-discriminator training
    
    All adjustments use standard ML optimization techniques.
    """
    
    def __init__(self, config):
        """
        Initialize controller with base configuration.
        
        Args:
            config: Training configuration with initial hyperparameters
        """
        # Base hyperparameters (will be adjusted dynamically)
        self.gen_lr = config.get('gen_lr', 0.0002)
        self.disc_lr = config.get('disc_lr', 0.0002)
        self.disc_threshold = config.get('disc_threshold', 0.55)
        
        # Loss weight configuration - UPDATED for high contradictions strategy
        self.loss_weights = {
            'wasserstein': 0.90,
            'hinge': 0.03,
            'lsgan': 0.03,
            'r1': 0.01,
            'adversarial': 1.0,  # Generator adversarial weight (adaptive controller will boost this)
            'diversity': 0.3,      # 30x stronger! Force within-GAN diversity
            'specialization': 0.5  # 50x stronger! Force between-GAN differences
        }
        
        # Tracking metrics history
        self.f1_history = []
        self.pass_rate_history = []
        self.disc_gap_history = []
        self.wgan_diversity_history = []  # Track WGAN sample diversity
        self.wgan_disc_pred_history = []  # Track WGAN discriminator predictions
        
        # State tracking
        self.current_strategy = "OBSERVING"  # Current optimization strategy
        self.epochs_since_improvement = 0
        self.best_f1 = 0.0
        self.wgan_collapse_detected = False  # Rama Dharma protection flag
        self.wgan_rejection_count = 0  # Count consecutive epochs where WGAN is rejected
        
        # Configuration
        self.enable_lr_adaptation = config.get('adapt_lr', True)
        self.enable_loss_adaptation = config.get('adapt_loss', True)
        self.enable_emergency_reset = config.get('enable_reset', True)
        
        # Thresholds for intervention
        self.disc_gap_critical = 0.3  # Discriminator too strong
        self.f1_collapse_threshold = 0.3  # Quality collapse
        self.pass_rate_low = 50  # Percent
        self.improvement_threshold = 0.01  # Minimum F1 improvement
        
        print("Adaptive Controller initialized")
        print(f"  Learning Rate Adaptation: {self.enable_lr_adaptation}")
        print(f"  Loss Weight Adaptation: {self.enable_loss_adaptation}")
        print(f"  Emergency Reset: {self.enable_emergency_reset}")
    
    def step(self, epoch, metrics):
        """
        Perform one optimization step based on current metrics.
        
        Args:
            epoch: Current training epoch
            metrics: Dict with keys:
                - 'f1': F1 score (if available)
                - 'pass_rate': Percentage of samples passing final validation
                - 'real_score': Discriminator score on real samples
                - 'fake_score': Discriminator score on fake samples
                - 'quality': Overall quality metric
                - 'wgan_diversity': WGAN sample diversity metric (optional)
                - 'wgan_disc_pred': WGAN discriminator prediction (optional, for fallback detection)
        
        Returns:
            Dict with updated hyperparameters to apply
        """
        # Extract metrics
        f1 = metrics.get('f1', None)
        pass_rate = metrics.get('pass_rate', 100)
        real_score = metrics.get('real_score', 0)
        fake_score = metrics.get('fake_score', 0)
        wgan_diversity = metrics.get('wgan_diversity', None)
        wgan_disc_pred = metrics.get('wgan_disc_pred', None)
        
        # Calculate discriminator gap
        disc_gap = abs(real_score - fake_score)
        
        # Update history
        if f1 is not None:
            self.f1_history.append(f1)
            if f1 > self.best_f1:
                self.best_f1 = f1
                self.epochs_since_improvement = 0
            else:
                self.epochs_since_improvement += 1
        
        self.pass_rate_history.append(pass_rate)
        self.disc_gap_history.append(disc_gap)
        
        if wgan_diversity is not None:
            self.wgan_diversity_history.append(wgan_diversity)
        
        if wgan_disc_pred is not None:
            self.wgan_disc_pred_history.append(wgan_disc_pred)
        
        # Check for WGAN mode collapse (Rama Dharma broken)
        # Method 1: Direct diversity check
        if wgan_diversity is not None and wgan_diversity < 0.01:
            self.wgan_collapse_detected = True
            print(f"  ⚠️  [RAMA DHARMA] WGAN mode collapse detected via diversity!")
            print(f"      Diversity = {wgan_diversity:.6f} < 0.01")
        
        # Method 2: FALLBACK - Check if WGAN is getting rejected by discriminator
        # If disc pred is negative for 2+ consecutive epochs, WGAN is collapsing (was 3, now 2 for earlier detection)
        if wgan_disc_pred is not None:
            if wgan_disc_pred < 0:
                self.wgan_rejection_count += 1
                if self.wgan_rejection_count >= 2 and not self.wgan_collapse_detected:
                    self.wgan_collapse_detected = True
                    print(f"  ⚠️  [RAMA DHARMA] WGAN mode collapse detected via discriminator rejection!")
                    print(f"      WGAN disc pred = {wgan_disc_pred:.4f} (negative for {self.wgan_rejection_count} epochs)")
            else:
                # Reset counter if WGAN is getting accepted
                self.wgan_rejection_count = 0
        
        # Determine optimization strategy
        strategy = self._assess_training_health(f1, disc_gap, pass_rate)
        self.current_strategy = strategy
        
        # Apply appropriate intervention
        updates = {}
        
        # Priority: Handle WGAN collapse first (Rama protection)
        if self.wgan_collapse_detected and strategy == "LOW_PASS_RATE":
            updates = self._handle_wgan_dharma_protection()
            
        elif strategy == "DISCRIMINATOR_DOMINANCE":
            updates = self._handle_discriminator_dominance()
            
        elif strategy == "QUALITY_COLLAPSE":
            updates = self._handle_quality_collapse()
            
        elif strategy == "LOW_PASS_RATE":
            updates = self._handle_low_pass_rate()
            
        elif strategy == "EMERGENCY_RESET":
            updates = self._handle_emergency_reset()
            
        else:  # HEALTHY or OBSERVING
            updates = {}  # No intervention needed
        
        # Log current state
        if epoch % 10 == 0:
            self._log_status(epoch, metrics, strategy)
        
        return updates
    
    def _assess_training_health(self, f1, disc_gap, pass_rate):
        """
        Assess current training health and determine strategy.
        
        Returns:
            Strategy string: HEALTHY, DISCRIMINATOR_DOMINANCE, QUALITY_COLLAPSE, etc.
        """
        # Emergency: Quality collapse
        if f1 is not None and f1 < self.f1_collapse_threshold:
            if self.epochs_since_improvement > 5:
                return "EMERGENCY_RESET"
            return "QUALITY_COLLAPSE"
        
        # Critical: Discriminator too strong
        if disc_gap > self.disc_gap_critical:
            return "DISCRIMINATOR_DOMINANCE"
        
        # Warning: Low pass rate
        if pass_rate < self.pass_rate_low:
            return "LOW_PASS_RATE"
        
        # Good: Everything healthy
        if f1 is not None and f1 > 0.7 and pass_rate > 70:
            return "HEALTHY"
        
        # Default: Observe
        return "OBSERVING"
    
    def _handle_discriminator_dominance(self):
        """
        Handle case where discriminator is crushing generators.
        
        Symptoms: Large gap between real_score and fake_score
        Solution: Boost generator learning, reduce discriminator strength
        """
        updates = {}
        
        if self.enable_lr_adaptation:
            # Increase generator learning rate
            self.gen_lr = min(self.gen_lr * 1.3, 0.01)  # Cap at 0.01
            # Decrease discriminator learning rate
            self.disc_lr = max(self.disc_lr * 0.7, 0.00005)  # Floor at 0.00005
            
            updates['gen_lr'] = self.gen_lr
            updates['disc_lr'] = self.disc_lr
        
        if self.enable_loss_adaptation:
            # Reduce Wasserstein weight (less adversarial pressure)
            self.loss_weights['wasserstein'] = max(0.70, self.loss_weights['wasserstein'] * 0.95)
            # Increase hinge weight (more stability)
            self.loss_weights['hinge'] = min(0.15, self.loss_weights['hinge'] * 1.5)
            # Boost generator adversarial weight
            self.loss_weights['adversarial'] = min(20.0, self.loss_weights['adversarial'] * 1.5)
            
            updates['loss_weights'] = self.loss_weights.copy()
        
        print(f"  [Strategy] Discriminator Dominance Detected")
        print(f"    Action: Boosting generator LR to {self.gen_lr:.6f}, reducing disc LR to {self.disc_lr:.6f}")
        
        return updates
    
    def _handle_quality_collapse(self):
        """
        Handle quality collapse (F1 dropping critically low).
        
        Symptoms: F1 < 0.3
        Solution: Relax validation, boost generator, add diversity
        """
        updates = {}
        
        # Relax final discriminator threshold (be more lenient)
        self.disc_threshold = max(0.2, self.disc_threshold * 0.9)
        updates['disc_threshold'] = self.disc_threshold
        
        if self.enable_lr_adaptation:
            # Boost generator learning
            self.gen_lr = min(self.gen_lr * 1.5, 0.01)
            updates['gen_lr'] = self.gen_lr
        
        if self.enable_loss_adaptation:
            # Add diversity pressure to prevent mode collapse
            self.loss_weights['diversity'] = min(0.1, self.loss_weights['diversity'] * 2.0)
            updates['loss_weights'] = self.loss_weights.copy()
        
        print(f"  [Strategy] Quality Collapse Detected")
        print(f"    Action: Relaxing threshold to {self.disc_threshold:.2f}, boosting diversity")
        
        return updates
    
    def _handle_low_pass_rate(self):
        """
        Handle low sample pass rate.
        
        Symptoms: < 50% of generated samples passing validation
        Solution: Strengthen generator adversarial learning
        """
        updates = {}
        
        if self.enable_loss_adaptation:
            # Significantly boost adversarial weight (but cap lower if WGAN collapsed)
            max_adv_weight = 20.0 if self.wgan_collapse_detected else 50.0
            self.loss_weights['adversarial'] = min(max_adv_weight, self.loss_weights['adversarial'] * 3.0)
            updates['loss_weights'] = self.loss_weights.copy()
        
        print(f"  [Strategy] Low Pass Rate Detected")
        print(f"    Action: Boosting adversarial weight to {self.loss_weights['adversarial']:.1f}x")
        
        return updates
    
    def _handle_wgan_dharma_protection(self):
        """
        RAMA DHARMA PROTECTION
        
        WGAN = Rama - follows discriminator feedback with unbreakable discipline
        Problem: When discriminator is harsh, WGAN collapses to single safe point
        Solution: Ease discriminator pressure specifically for WGAN
        
        Symptoms: WGAN sample diversity < 0.01 (mode collapse) + low pass rate
        Solution: Reduce adversarial pressure, boost diversity, gentle discriminator
        
        Philosophy:
        - Rama never breaks dharma (rules), but rules must not be impossible
        - Krishna adapts creatively, but Rama needs clear, fair guidelines
        - If discriminator is too harsh, Rama gives up → needs protection
        """
        updates = {}
        
        print(f"  ⚡ [RAMA DHARMA PROTECTION] Protecting WGAN from harsh feedback")
        
        if self.enable_loss_adaptation:
            # AGGRESSIVELY REDUCE adversarial weight (more than before!)
            # Cap at 5x instead of letting it stay at 25x
            self.loss_weights['adversarial'] = max(5.0, self.loss_weights['adversarial'] * 0.3)
            
            # STRONGLY BOOST diversity weight to force exploration
            self.loss_weights['diversity'] = min(0.5, self.loss_weights['diversity'] * 10.0)
            
            # Reduce Wasserstein weight (less harsh gradient)
            self.loss_weights['wasserstein'] = max(0.60, self.loss_weights['wasserstein'] * 0.85)
            
            updates['loss_weights'] = self.loss_weights.copy()
        
        if self.enable_lr_adaptation:
            # AGGRESSIVELY lower discriminator learning rate
            self.disc_lr = max(self.disc_lr * 0.3, 0.00001)  # Lower floor, bigger reduction
            updates['disc_lr'] = self.disc_lr
        
        print(f"    Adversarial weight: {self.loss_weights['adversarial']:.1f}x (REDUCED - was crushing Rama)")
        print(f"    Diversity weight: {self.loss_weights['diversity']:.3f} (BOOSTED - force exploration)")
        print(f"    Disc LR: {self.disc_lr:.6f} (GENTLER - give Rama breathing room)")
        print(f"    Philosophy: Rama follows dharma, but dharma must be achievable!")
        
        # Reset collapse flag and rejection counter after intervention
        self.wgan_collapse_detected = False
        self.wgan_rejection_count = 0
        
        return updates
    
    def _handle_emergency_reset(self):
        """
        Emergency reset when training is stuck.
        
        Symptoms: No improvement for many epochs + critically low F1
        Solution: Reset to best known configuration + increase exploration
        """
        if not self.enable_emergency_reset:
            return {}
        
        print(f"  [EMERGENCY] Training Reset Triggered")
        print(f"    Reason: No improvement for {self.epochs_since_improvement} epochs, F1={self.f1_history[-1]:.3f}")
        
        # Reset to initial conservative values
        self.gen_lr = 0.0001  # Lower LR for stability
        self.disc_lr = 0.0001
        self.disc_threshold = 0.4  # More lenient
        
        # Reset loss weights to balanced
        self.loss_weights = {
            'wasserstein': 0.80,
            'hinge': 0.10,
            'lsgan': 0.05,
            'r1': 0.01,
            'adversarial': 5.0,  # Moderate boost
            'diversity': 0.05,  # Increase diversity
            'specialization': 0.01
        }
        
        updates = {
            'gen_lr': self.gen_lr,
            'disc_lr': self.disc_lr,
            'disc_threshold': self.disc_threshold,
            'loss_weights': self.loss_weights.copy(),
            'reset_flag': True  # Signal to trainer to reinitialize optimizers
        }
        
        self.epochs_since_improvement = 0
        
        return updates
    
    def _log_status(self, epoch, metrics, strategy):
        """Log current controller status."""
        print(f"\n  [Adaptive Controller] Epoch {epoch}")
        print(f"    Strategy: {strategy}")
        print(f"    Gen LR: {self.gen_lr:.6f} | Disc LR: {self.disc_lr:.6f}")
        print(f"    Disc Threshold: {self.disc_threshold:.2f}")
        print(f"    Loss Weights: Wass={self.loss_weights['wasserstein']:.2f}, "
              f"Adv={self.loss_weights['adversarial']:.1f}x, "
              f"Div={self.loss_weights['diversity']:.3f}")
        
        if len(self.f1_history) > 0:
            print(f"    F1 History (last 5): {self.f1_history[-5:]}")
            print(f"    Best F1: {self.best_f1:.3f} | Epochs since improvement: {self.epochs_since_improvement}")
    
    def get_state(self):
        """Get current controller state for checkpointing."""
        return {
            'gen_lr': self.gen_lr,
            'disc_lr': self.disc_lr,
            'disc_threshold': self.disc_threshold,
            'loss_weights': self.loss_weights.copy(),
            'f1_history': self.f1_history.copy(),
            'pass_rate_history': self.pass_rate_history.copy(),
            'disc_gap_history': self.disc_gap_history.copy(),
            'best_f1': self.best_f1,
            'current_strategy': self.current_strategy,
            'epochs_since_improvement': self.epochs_since_improvement
        }
    
    def load_state(self, state):
        """Load controller state from checkpoint."""
        self.gen_lr = state['gen_lr']
        self.disc_lr = state['disc_lr']
        self.disc_threshold = state['disc_threshold']
        self.loss_weights = state['loss_weights']
        self.f1_history = state['f1_history']
        self.pass_rate_history = state['pass_rate_history']
        self.disc_gap_history = state['disc_gap_history']
        self.best_f1 = state['best_f1']
        self.current_strategy = state['current_strategy']
        self.epochs_since_improvement = state['epochs_since_improvement']
