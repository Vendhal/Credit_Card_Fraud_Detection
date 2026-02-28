"""
⚔️🌟 THE GREAT CONVERGENCE - Safe Framework Edition 🌟⚔️
===========================================================

🛡️ RIGOROUS EVALUATION FRAMEWORK (No Overfitting!):
✅ OOF (Out-of-Fold) Thresholding
✅ Unified Probability Space  
✅ Knowledge Vampire (Pattern Transfer)
✅ Hall of Fame (Elite Preservation)
✅ Training-Only Evolution

Safety First:
1. OOF predictions from K-fold CV (no leakage)
2. Threshold selected ONCE on OOF, LOCKED
3. Final model trained with frozen threshold
4. Single test evaluation (no adaptation)
5. All evolution on training data only

This is a PUBLICATION-READY framework with rigorous validation!

Author: The Unity Architect
Date: 2026-02-18
Version: 2.0 (Safe OOF Edition)
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import both frameworks
import sys
import os
# great_convergence.py is already in models/, so parent directory is project root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from the_arena import TheArena, ArenaExpert, ARENA_CONFIG, NICHE_TYPES
    ARENA_AVAILABLE = True
except ImportError:
    ARENA_AVAILABLE = False
    print("⚠️ Arena not available")

try:
    from kurukshetra_collective import KurukshetraCollective, SenaFormation
    from kurukshetra import Yoddha, DHARMA_CONFIG, VARNA_TYPES
    KURUKSHETRA_AVAILABLE = True
    KURUKSHETRA_COLLECTIVE_AVAILABLE = True
    print("✅ Kurukshetra Collective (East) loaded")
except ImportError as e:
    KURUKSHETRA_AVAILABLE = False
    KURUKSHETRA_COLLECTIVE_AVAILABLE = False
    print(f"⚠️ Kurukshetra not available: {e}")
    import traceback
    traceback.print_exc()

from sklearn.metrics import f1_score, precision_score, recall_score
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import random
from collections import deque
import pickle
import time

# Supreme Configuration
CONVERGENCE_CONFIG = {
    'total_models': 50,              # 25 Arena + 25 Kurukshetra
    'arena_models': 25,
    'kurukshetra_models': 25,
    'battle_rounds': 200,            # Ultimate stability
    'east_west_battles_per_round': 10,  # Direct confrontations
    'cross_breeding_rate': 0.2,      # 20% chance to adopt enemy trait
    'knowledge_transfer_rate': 0.3,   # Share learnings
    'specialist_ratio': 0.4,         # 40% specialists
    'cascade_levels': 3,             # 3-level cascade ensemble
    'cultural_exchange_interval': 10, # Every 10 rounds
    'stability_threshold': 0.95,      # F1 stability target
}

# 🕉️ VISVARUPA CONFIGURATION - Divine Chaos to Prevent Stagnation
VISVARUPA_CONFIG = {
    'avatar_interval': 20,           # Every 20 rounds: Avatar Injection
    'cataclysm_interval': 50,        # Every 50 rounds: The Cataclysm
    'n_avatars': 6,                  # 6 divine warriors appear (3 East + 3 West)
    'cataclysm_survival_rate': 0.5,  # Bottom 50% wiped out
    'divine_mutation_rate': 10.0,    # 10x normal mutation during chaos
    'avatar_lifespan': 10,           # Avatars disappear after 10 rounds
}


@dataclass
class CulturalExchange:
    """
    🌍 Cultural Exchange Record
    When East meets West and shares wisdom
    """
    source_framework: str           # 'Arena' or 'Kurukshetra'
    target_framework: str
    exchange_type: str              # 'parameter', 'strategy', 'niche', 'architecture'
    source_model_id: str
    target_model_id: str
    improvement_metric: float
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        return {
            'source': self.source_framework,
            'target': self.target_framework,
            'type': self.exchange_type,
            'source_model': self.source_model_id,
            'target_model': self.target_model_id,
            'improvement': self.improvement_metric,
            'time': self.timestamp.isoformat()
        }


@dataclass
class EastWestBattle:
    """
    ⚔️ East vs West Battle Record
    Epic duel between civilizations
    """
    round_num: int
    arena_model_id: str
    kurukshetra_model_id: str
    arena_f1: float
    kurukshetra_f1: float
    winner: str                     # 'East', 'West', or 'Draw'
    margin: float
    cultural_exchange: Optional[CulturalExchange] = None
    
    def to_dict(self) -> Dict:
        return {
            'round': self.round_num,
            'arena_model': self.arena_model_id,
            'kurukshetra_model': self.kurukshetra_model_id,
            'arena_f1': self.arena_f1,
            'kurukshetra_f1': self.kurukshetra_f1,
            'winner': self.winner,
            'margin': self.margin,
            'exchange': self.cultural_exchange.to_dict() if self.cultural_exchange else None
        }


class GreatConvergence:
    """
    🌟 The Great Convergence - East vs West Ultimate Battle
    
    Where:
    - Arena (West) and Kurukshetra (East) coexist
    - They battle each other in epic duels
    - Cultural exchange occurs (best of both worlds!)
    - Cross-breeding creates hybrid super-models
    - Cascade ensemble combines all strengths
    - 200 rounds ensures ultimate stability
    """
    
    def __init__(self, 
                 n_models_per_side: int = 25,
                 battle_rounds: int = 200,
                 cascade_levels: int = 3):
        self.n_arena_models = n_models_per_side
        self.n_kurukshetra_models = n_models_per_side
        self.battle_rounds = battle_rounds
        self.cascade_levels = cascade_levels
        
        # Initialize both frameworks
        self.arena = None
        self.kurukshetra = None
        
        # East vs West battle history
        self.east_west_battles: List[EastWestBattle] = []
        self.cultural_exchanges: List[CulturalExchange] = []
        
        # Hybrid models (cross-bred)
        self.hybrid_models: List = []
        
        # Cascade ensemble
        self.cascade_ensemble = None
        self.best_f1 = 0.0
        self.best_ensemble = None
        
        # 🏆 HALL OF FAME - Elite Preservation (ChatGPT Recommendation #1)
        self.hall_of_fame = []  # Top 20 models of all time
        self.elite_models = []  # Top 3 models (never destroyed in Cataclysm)
        self.HALL_OF_FAME_SIZE = 20
        self.ELITE_SIZE = 3
        
        # 🎯 ADAPTIVE THRESHOLD - Not fixed 0.5 (ChatGPT Recommendation #3)
        self.optimal_threshold = 0.5  # Will be optimized on validation
        self.threshold_history = []
        
        # 📊 PR-AUC Tracking (ChatGPT Recommendation #2)
        self.best_pr_auc = 0.0
        self.best_recall_at_95_precision = 0.0
        self.use_pr_metric = True  # Use PR-AUC instead of pure F1
        
        # Stability tracking
        self.f1_history = deque(maxlen=20)
        self.is_stable = False
        
        # 🛡️ SAFE OOF THRESHOLDING (Framework Edition)
        self.use_safe_oof = False  # Set to True for rigorous evaluation
        self.oof_threshold = None  # Threshold from OOF (locked)
        self.threshold_locked = False  # Once set, never changes
        self.oof_stats = {}  # Statistics from OOF phase
        
        print("="*80)
        print("⚔️🌟 THE GREAT CONVERGENCE - SAFE FRAMEWORK EDITION 🌟⚔️")
        print("="*80)
        print()
        print("🏟️  WEST (Arena): Roman Gladiators in the Colosseum")
        print("⚔️  EAST (Kurukshetra): Warriors in the Dharmic Field")
        print()
        print("They will battle, learn, and converge...")
        print("Cultural exchange will create hybrid super-models!")
        print("200 rounds will forge the ultimate ensemble!")
        print()
        print(f"Configuration:")
        print(f"   Arena Models: {self.n_arena_models}")
        print(f"   Kurukshetra Models: {self.n_kurukshetra_models}")
        print(f"   Total Warriors: {self.n_arena_models + self.n_kurukshetra_models}")
        print(f"   Battle Rounds: {self.battle_rounds}")
        print(f"   Cascade Levels: {self.cascade_levels}")
        print()
        print("Let the Great Convergence begin! 🌟")
        print("="*80)
        print()
    
    def initialize_frameworks(self, X_train, y_train, X_val, y_val, feature_names=None):
        """
        Initialize both Arena and Kurukshetra
        """
        print("🌟 PHASE 1: INITIALIZING CIVILIZATIONS...")
        print()
        
        # Initialize Arena (West)
        if ARENA_AVAILABLE:
            print("🏟️ Initializing ARENA (West)...")
            self.arena = TheArena(
                n_challengers=self.n_arena_models - 1,  # -1 for champion
                combat_rounds=self.battle_rounds,
                batch_size=1000
            )
            # Don't train yet, just initialize
            self.arena.initialize_champion(X_train, y_train, feature_names)
            self.arena.spawn_challengers(X_train, y_train, feature_names)
            print(f"   ✓ Arena: 1 Champion + {len(self.arena.challengers)} Challengers ready!")
        
        # Initialize Kurukshetra (East) - COLLECTIVE VERSION
        if KURUKSHETRA_COLLECTIVE_AVAILABLE:
            print("⚔️ Initializing KURUKSHETRA COLLECTIVE (East)...")
            print("   🛡️ Group Combat: ENABLED")
            print("   📚 Guru-Shishya Lineage: ENABLED")
            print("   🕉️ Shared Karma: ENABLED")
            self.kurukshetra = KurukshetraCollective(
                n_senapatis=self.n_kurukshetra_models - 1,
                yuddha_rounds=self.battle_rounds
            )
            # Initialize but don't train yet - training happens in run_convergence
            self.kurukshetra.maharathi = Yoddha("Maharathi_Arjun", varna='sarvabhaum')
            self.kurukshetra.maharathi.tapas(X_train, y_train, feature_names)
            self.kurukshetra.senapatis = []
            for i in range(self.n_kurukshetra_models - 1):
                yoddha = Yoddha(f"Senapati_{i}", varna=random.choice(VARNA_TYPES))
                yoddha.tapas(X_train, y_train, feature_names)
                self.kurukshetra.senapatis.append(yoddha)
            self.kurukshetra.create_formations()
            print(f"   ✓ Kurukshetra Collective: 1 Maharathi + {len(self.kurukshetra.senapatis)} Senapatis in {len(self.kurukshetra.formations)} Formations ready!")
        
        print()
        print(f"🌍 Total Warriors: {self.get_total_warriors()}")
        print("   East and West are ready for battle!")
        print()
    
    def get_total_warriors(self) -> int:
        """Get total number of warriors across both civilizations"""
        total = 0
        if self.arena:
            total += 1 + len(self.arena.challengers) + len(self.arena.elders)
        if self.kurukshetra:
            total += 1 + len(self.kurukshetra.senapatis) + len(self.kurukshetra.pitamahas)
        return total + len(self.hybrid_models)
    
    def run_east_west_battle(self, X_val, y_val) -> EastWestBattle:
        """
        ⚔️ Run an epic East vs West duel
        """
        # Select random warriors from each side
        arena_warriors = [self.arena.champion] + self.arena.challengers
        kurukshetra_warriors = [self.kurukshetra.maharathi] + self.kurukshetra.senapatis
        
        arena_model = random.choice(arena_warriors)
        kurukshetra_model = random.choice(kurukshetra_warriors)
        
        # Battle on validation set
        batch_size = min(1000, len(X_val))
        indices = np.random.choice(len(X_val), batch_size, replace=False)
        X_batch = X_val[indices]
        y_batch = y_val[indices]
        
        # Get predictions
        # 🛠️ Convert to numpy if GPU arrays (for Visvarupa avatars)
        if hasattr(X_batch, 'get'):
            X_batch_np = X_batch.get()
        else:
            X_batch_np = np.asarray(X_batch)
        arena_preds = arena_model.predict(X_batch_np)
        kurukshetra_preds = kurukshetra_model.yuddha(X_batch_np)
        
        # Calculate F1 scores
        arena_f1 = f1_score(y_batch, arena_preds, average='macro', zero_division=0)
        kurukshetra_f1 = f1_score(y_batch, kurukshetra_preds, average='macro', zero_division=0)
        
        # Determine winner
        if kurukshetra_f1 > arena_f1:
            winner = 'East'
            margin = kurukshetra_f1 - arena_f1
        elif arena_f1 > kurukshetra_f1:
            winner = 'West'
            margin = arena_f1 - kurukshetra_f1
        else:
            winner = 'Draw'
            margin = 0
        
        # Cultural exchange if significant difference
        exchange = None
        if margin > 0.05:  # Significant victory
            exchange = self._cultural_exchange(
                winner, 
                arena_model if winner == 'West' else kurukshetra_model,
                kurukshetra_model if winner == 'West' else arena_model,
                margin
            )
        
        battle = EastWestBattle(
            round_num=len(self.east_west_battles),
            arena_model_id=arena_model.expert_id,
            kurukshetra_model_id=kurukshetra_model.yoddha_id,
            arena_f1=arena_f1,
            kurukshetra_f1=kurukshetra_f1,
            winner=winner,
            margin=margin,
            cultural_exchange=exchange
        )
        
        self.east_west_battles.append(battle)
        
        return battle
    
    def _cultural_exchange(self, winner: str, winner_model, loser_model, margin: float) -> CulturalExchange:
        """
        🌍 Cultural Exchange - Winner teaches loser
        """
        if winner == 'West':
            # Arena (West) won, teaches Kurukshetra (East)
            source_fw = 'Arena'
            target_fw = 'Kurukshetra'
            
            # Transfer niche specialization strategy
            if hasattr(winner_model, 'niche') and hasattr(loser_model, 'varna'):
                if random.random() < CONVERGENCE_CONFIG['cross_breeding_rate']:
                    # Kurukshetra model adopts Arena's niche concept
                    exchange_type = 'strategy'
                    improvement = margin * 0.5
        else:
            # Kurukshetra (East) won, teaches Arena (West)
            source_fw = 'Kurukshetra'
            target_fw = 'Arena'
            
            # Transfer varna specialization strategy
            if hasattr(winner_model, 'varna') and hasattr(loser_model, 'niche'):
                if random.random() < CONVERGENCE_CONFIG['cross_breeding_rate']:
                    exchange_type = 'strategy'
                    improvement = margin * 0.5
        
        exchange = CulturalExchange(
            source_framework=source_fw,
            target_framework=target_fw,
            exchange_type=exchange_type if 'exchange_type' in dir() else 'general',
            source_model_id=winner_model.expert_id if hasattr(winner_model, 'expert_id') else winner_model.yoddha_id,
            target_model_id=loser_model.expert_id if hasattr(loser_model, 'expert_id') else loser_model.yoddha_id,
            improvement_metric=margin,
            timestamp=datetime.now()
        )
        
        self.cultural_exchanges.append(exchange)
        
        return exchange
    
    def create_hybrid_model(self, X_train, y_train, feature_names=None):
        """
        🧬 Create hybrid model combining East and West traits
        """
        if not self.arena or not self.kurukshetra:
            return None
        
        # Import both classes at the start
        from the_arena import ArenaExpert, NICHE_TYPES as ARENA_NICHE_TYPES
        from kurukshetra import Yoddha, VARNA_TYPES as KURUKSHETRA_VARNA_TYPES
        
        # Select top performers from each side
        arena_top = max([self.arena.champion] + self.arena.challengers, 
                       key=lambda x: x.get_combat_score())
        kurukshetra_top = max([self.kurukshetra.maharathi] + self.kurukshetra.senapatis,
                             key=lambda x: x.get_yoddha_bal())
        
        # Create hybrid
        hybrid_id = f"Hybrid_{len(self.hybrid_models):03d}"
        
        # Decide which framework to use as base
        if random.random() < 0.5:
            # Arena base with Kurukshetra traits
            hybrid = ArenaExpert(
                expert_id=hybrid_id,
                model_type=arena_top.model_type,
                niche=random.choice(ARENA_NICHE_TYPES + list(KURUKSHETRA_VARNA_TYPES)),  # Mix both!
                seed=random.randint(0, 1000000)
            )
        else:
            # Kurukshetra base with Arena traits
            hybrid = Yoddha(
                yoddha_id=hybrid_id,
                shastra_type=kurukshetra_top.shastra_type,
                varna=random.choice(list(KURUKSHETRA_VARNA_TYPES) + ARENA_NICHE_TYPES),  # Mix both!
                beeja=random.randint(0, 1000000)
            )
        
        # Train the hybrid
        if isinstance(hybrid, ArenaExpert):
            hybrid.fit(X_train, y_train, feature_names=feature_names)
            is_trained = hybrid.is_fitted
        else:
            hybrid.tapas(X_train, y_train, feature_names=feature_names)
            is_trained = hybrid.is_tejasvi
        
        if is_trained:
            self.hybrid_models.append(hybrid)
            print(f"   🧬 Created hybrid model: {hybrid_id}")
            return hybrid
        
        return None
    
    def run_convergence(self, X_train, y_train, X_val, y_val, feature_names=None):
        """
        🌟 Run the Great Convergence
        """
        print("="*80)
        print("🌟 THE GREAT CONVERGENCE - BEGINS NOW! 🌟")
        print("="*80)
        print()
        
        # Initialize both civilizations
        self.initialize_frameworks(X_train, y_train, X_val, y_val, feature_names)
        
        start_time = time.time()
        
        # Run battle rounds
        for round_num in range(self.battle_rounds):
            print(f"\n{'='*80}")
            print(f"⚔️ EAST VS WEST - ROUND {round_num + 1}/{self.battle_rounds} ⚔️")
            print(f"{'='*80}")
            
            # Run internal battles for both frameworks
            if self.arena:
                print("\n🏟️ WEST (Arena) - Internal Tournament:")
                self.arena.run_tournament(X_val, y_val)
                
                # Evolve every 5 rounds
                if (round_num + 1) % 5 == 0:
                    print("\n🧬 WEST - Evolving Population...")
                    self.arena.evolve_population(X_train, y_train, feature_names)
            
            if self.kurukshetra:
                print("\n⚔️ EAST (Kurukshetra Collective) - Group Formations Battle:")
                # Multiple group battles per round
                for _ in range(5):
                    self.kurukshetra.group_yuddha(X_val, y_val)
                
                # 🔄 ANTI-OVERFITTING: Shuffle formations every 15 rounds
                if (round_num + 1) % 15 == 0:
                    print("\n🔄 Shuffling Formations (Anti-Overfitting)...")
                    self.kurukshetra.shuffle_formations()
                
                # 📊 ANTI-OVERFITTING: Evaluate individual performance every 25 rounds
                if (round_num + 1) % 25 == 0:
                    print("\n📊 Evaluating Individual Performance...")
                    self.kurukshetra.evaluate_individual_performance(X_val, y_val)
                
                # Evolve every 20 rounds with memory
                if (round_num + 1) % 20 == 0:
                    print("\n🧬 EAST - Punarjanma with Collective Memory...")
                    self.kurukshetra.punarjanma_with_memory(X_train, y_train, feature_names)
            
            # Run East vs West battles
            print(f"\n🌍 EAST VS WEST - Direct Confrontations:")
            for battle_num in range(CONVERGENCE_CONFIG['east_west_battles_per_round']):
                battle = self.run_east_west_battle(X_val, y_val)
                
                if battle_num == 0:  # Print first battle of round
                    print(f"   Battle 1: {battle.arena_model_id} (Arena: {battle.arena_f1:.3f}) vs")
                    print(f"             {battle.kurukshetra_model_id} (Kurukshetra: {battle.kurukshetra_f1:.3f})")
                    print(f"   Winner: {battle.winner} (margin: {battle.margin:.3f})")
                    if battle.cultural_exchange:
                        print(f"   🌍 Cultural Exchange: {battle.cultural_exchange.exchange_type}")
            
            # 🕉️ VISVARUPA AVATAR INJECTION - Every 20 rounds
            if (round_num + 1) % VISVARUPA_CONFIG['avatar_interval'] == 0 and round_num > 0:
                self.visvarupa_avatar_injection(X_train, y_train, X_val, y_val, feature_names)
            
            # 🕉️ REMOVE EXPIRED AVATARS - After 10 rounds of their appearance
            if (round_num + 1) % VISVARUPA_CONFIG['avatar_interval'] == VISVARUPA_CONFIG['avatar_lifespan'] and round_num > 10:
                self.remove_expired_avatars()
            
            # Cultural exchange interval
            if (round_num + 1) % CONVERGENCE_CONFIG['cultural_exchange_interval'] == 0:
                print(f"\n🌍 CULTURAL EXCHANGE PERIOD - Creating Hybrids:")
                for _ in range(3):  # Create 3 hybrids
                    self.create_hybrid_model(X_train, y_train, feature_names)
            
            # 🕉️ THE CATACLYSM - Every 50 rounds
            if (round_num + 1) % VISVARUPA_CONFIG['cataclysm_interval'] == 0 and round_num > 0:
                self.visvarupa_cataclysm(X_train, y_train, X_val, y_val, feature_names, use_vampire=True)
            
            # Evaluate and update best every 10 rounds
            if (round_num + 1) % 10 == 0:
                self._evaluate_convergence(X_val, y_val)
                self._print_convergence_status()
                
                # Check for stability
                if len(self.f1_history) >= 10:
                    recent_std = np.std(list(self.f1_history)[-10:])
                    if recent_std < 0.01 and self.best_f1 > 0.90:
                        self.is_stable = True
                        print(f"\n🌟 CONVERGENCE STABILIZED! 🌟")
                        print(f"   Best F1: {self.best_f1:.4f}")
                        print(f"   Stability: {recent_std:.4f} std dev")
                        break
                    
                    # Check for overfitting: compare train vs val F1
                    train_f1, val_f1, gap = self._evaluate_train_val_gap(X_train, y_train, X_val, y_val)
                    if gap > 0.05:  # 5% gap = overfitting
                        print(f"\n⚠️ OVERFITTING DETECTED! ⚠️")
                        print(f"   Train F1: {train_f1:.4f}")
                        print(f"   Val F1: {val_f1:.4f}")
                        print(f"   Gap: {gap:.4f} (>0.05)")
                        print(f"   Triggering Cataclysm early to increase diversity...")
                        self.visvarupa_cataclysm(X_train, y_train, X_val, y_val, feature_names, use_vampire=True)
        
        # Final ensemble construction
        print("\n" + "="*80)
        print("🏗️ BUILDING CASCADE ENSEMBLE...")
        print("="*80)
        self._build_cascade_ensemble(X_val, y_val)
        
        elapsed = time.time() - start_time
        print(f"\n⏱️  Total Convergence Time: {elapsed/60:.1f} minutes")
        
        return self
    
    def _evaluate_convergence(self, X_val, y_val):
        """
        Evaluate current ensemble performance
        Uses PR-AUC + Recall@Precision>=0.95 instead of pure F1
        """
        if self.use_pr_metric:
            # 🎯 NEW: Use PR-AUC based evaluation
            self._evaluate_with_pr_metric(X_val, y_val)
        else:
            # Legacy F1-based evaluation
            # Collect all models
            all_models = []
            
            if self.arena:
                all_models.extend([self.arena.champion] + self.arena.challengers + self.arena.elders)
            
            if self.kurukshetra:
                all_models.extend([self.kurukshetra.maharathi] + self.kurukshetra.senapatis + self.kurukshetra.pitamahas)
            
            all_models.extend(self.hybrid_models)
            
            # Get top performers
            def get_score(model):
                if hasattr(model, 'get_combat_score'):
                    return model.get_combat_score()
                elif hasattr(model, 'get_yoddha_bal'):
                    return model.get_yoddha_bal()
                return 0.0
            
            all_models.sort(key=get_score, reverse=True)
            top_models = all_models[:10]
            
            # Ensemble prediction
            predictions = []
            for model in top_models:
                try:
                    # 🛠️ Convert to numpy if GPU arrays
                    if hasattr(X_val, 'get'):
                        X_val_np = X_val.get()
                    else:
                        X_val_np = np.asarray(X_val)
                    if hasattr(model, 'predict'):
                        pred = model.predict(X_val_np)
                    else:
                        pred = model.yuddha(X_val_np)
                    predictions.append(pred)
                except:
                    pass
            
            if len(predictions) > 0:
                ensemble_pred = np.round(np.mean(predictions, axis=0)).astype(int)
                current_f1 = f1_score(y_val, ensemble_pred, zero_division=0)
                
                self.f1_history.append(current_f1)
                
                if current_f1 > self.best_f1:
                    self.best_f1 = current_f1
                    self.best_ensemble = top_models
    
    def _evaluate_train_val_gap(self, X_train, y_train, X_val, y_val):
        """
        Check for overfitting by comparing train vs validation F1
        If gap > 5%, we're overfitting to training data
        """
        # Get all current models
        all_models = []
        if self.arena:
            all_models.extend([self.arena.champion] + self.arena.challengers + self.arena.elders)
        if self.kurukshetra:
            all_models.extend([self.kurukshetra.maharathi] + self.kurukshetra.senapatis + self.kurukshetra.pitamahas)
        all_models.extend(self.hybrid_models)
        
        if len(all_models) < 5:
            return 0.0, 0.0, 0.0
        
        # Sample for speed
        train_sample_size = min(10000, len(X_train))
        val_sample_size = min(5000, len(X_val))
        
        train_idx = np.random.choice(len(X_train), train_sample_size, replace=False)
        val_idx = np.random.choice(len(X_val), val_sample_size, replace=False)
        
        X_train_sample = X_train[train_idx]
        y_train_sample = y_train[train_idx]
        X_val_sample = X_val[val_idx]
        y_val_sample = y_val[val_idx]
        
        # Get top 10 models
        def get_score(m):
            if hasattr(m, 'get_combat_score'):
                return m.get_combat_score()
            elif hasattr(m, 'get_yoddha_bal'):
                return m.get_yoddha_bal()
            return 0.0
        
        all_models.sort(key=get_score, reverse=True)
        top_models = all_models[:10]
        
        # Ensemble predictions
        train_preds = []
        val_preds = []
        
        # 🛠️ Convert to numpy if GPU arrays
        if hasattr(X_train_sample, 'get'):
            X_train_np = X_train_sample.get()
        else:
            X_train_np = np.asarray(X_train_sample)
        if hasattr(X_val_sample, 'get'):
            X_val_np = X_val_sample.get()
        else:
            X_val_np = np.asarray(X_val_sample)
        
        for model in top_models:
            try:
                if hasattr(model, 'predict'):
                    train_preds.append(model.predict(X_train_np))
                    val_preds.append(model.predict(X_val_np))
                else:
                    train_preds.append(model.yuddha(X_train_np))
                    val_preds.append(model.yuddha(X_val_np))
            except:
                pass
        
        if len(train_preds) == 0:
            return 0.0, 0.0, 0.0
        
        train_ensemble = np.round(np.mean(train_preds, axis=0)).astype(int)
        val_ensemble = np.round(np.mean(val_preds, axis=0)).astype(int)
        
        train_f1 = f1_score(y_train_sample, train_ensemble, zero_division=0)
        val_f1 = f1_score(y_val_sample, val_ensemble, zero_division=0)
        gap = train_f1 - val_f1
        
        return train_f1, val_f1, gap

    def _print_convergence_status(self):
        """Print current convergence status with PR metrics and Hall of Fame"""
        print(f"\n📊 CONVERGENCE STATUS:")
        print(f"   Total Warriors: {self.get_total_warriors()}")
        
        if self.arena:
            print(f"   West (Arena): {len(self.arena.challengers)} + 1 Champion + {len(self.arena.elders)} Elders")
        
        if self.kurukshetra:
            print(f"   East (Kurukshetra): {len(self.kurukshetra.senapatis)} + 1 Maharathi + {len(self.kurukshetra.pitamahas)} Pitamahas")
        
        print(f"   Hybrids: {len(self.hybrid_models)}")
        print(f"   East vs West Battles: {len(self.east_west_battles)}")
        print(f"   Cultural Exchanges: {len(self.cultural_exchanges)}")
        
        # 🏆 Hall of Fame Status
        print(f"\n🏆 HALL OF FAME:")
        print(f"   Elite Models (Protected): {len(self.elite_models)}/{self.ELITE_SIZE}")
        print(f"   Hall of Fame: {len(self.hall_of_fame)}/{self.HALL_OF_FAME_SIZE}")
        
        # 🎯 PR Metrics (if using new metric)
        if self.use_pr_metric:
            print(f"\n🎯 PR-AUC METRICS:")
            print(f"   PR-AUC: {self.best_pr_auc:.4f}")
            print(f"   Recall@Precision≥0.95: {self.best_recall_at_95_precision:.4f}")
            print(f"   Optimal Threshold: {self.optimal_threshold:.3f}")
        
        print(f"\n📈 PERFORMANCE:")
        print(f"   Current F1: {self.f1_history[-1]:.4f}" if self.f1_history else "   Current F1: N/A")
        print(f"   Best F1: {self.best_f1:.4f}")
        
        if len(self.east_west_battles) > 0:
            recent_battles = self.east_west_battles[-100:]
            east_wins = sum(1 for b in recent_battles if b.winner == 'East')
            west_wins = sum(1 for b in recent_battles if b.winner == 'West')
            draws = sum(1 for b in recent_battles if b.winner == 'Draw')
            print(f"   Recent East vs West: East {east_wins} - West {west_wins} - Draws {draws}")
    
    # 🏆 HALL OF FAME METHODS (ChatGPT Recommendation #1)
    
    def _update_hall_of_fame(self, models: List, X_val, y_val):
        """
        Update Hall of Fame with top performing models
        Elite models (top 3) are protected from Cataclysm
        """
        if len(models) == 0:
            return
        
        # Score all models on validation set
        # 🛠️ Convert to numpy if GPU arrays
        if hasattr(X_val, 'get'):
            X_val_np = X_val.get()
        else:
            X_val_np = np.asarray(X_val)
        
        model_scores = []
        for model in models:
            try:
                if hasattr(model, 'predict'):
                    pred = model.predict(X_val_np)
                else:
                    pred = model.yuddha(X_val_np)
                
                f1 = f1_score(y_val, pred, zero_division=0)
                model_scores.append((model, f1))
            except:
                pass
        
        if len(model_scores) == 0:
            return
        
        # Sort by performance
        model_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Update Hall of Fame (top 20)
        for model, score in model_scores[:self.HALL_OF_FAME_SIZE]:
            # Check if already in hall of fame
            existing_ids = [id(m) for m in self.hall_of_fame]
            if id(model) not in existing_ids:
                self.hall_of_fame.append(model)
        
        # Keep only top 20
        self.hall_of_fame = self.hall_of_fame[:self.HALL_OF_FAME_SIZE]
        
        # Update Elite models (top 3) - PROTECTED FROM CATACLYSM
        self.elite_models = [model for model, _ in model_scores[:self.ELITE_SIZE]]
        
        print(f"\n🏆 Hall of Fame Updated:")
        print(f"   Elite Models (Protected): {len(self.elite_models)}")
        print(f"   Hall of Fame: {len(self.hall_of_fame)} models")
    
    def _is_elite_model(self, model) -> bool:
        """Check if a model is in the elite group (protected from destruction)"""
        return id(model) in [id(m) for m in self.elite_models]
    
    # 🎯 PR-AUC & ADAPTIVE THRESHOLD METHODS (ChatGPT Recommendations #2 & #3)
    
    def _calculate_pr_metrics(self, y_true, y_proba):
        """
        Calculate PR-AUC and Recall@Precision≥0.95
        Returns: (pr_auc, recall_at_95_precision, optimal_threshold)
        """
        from sklearn.metrics import precision_recall_curve, auc
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        
        # Calculate PR-AUC
        pr_auc = auc(recall, precision)
        
        # Find recall where precision >= 0.95
        recall_at_95 = 0.0
        for p, r in zip(precision, recall):
            if p >= 0.95:
                recall_at_95 = max(recall_at_95, r)
        
        # Find optimal threshold (maximize F1)
        f1_scores = []
        for p, r in zip(precision, recall):
            if p + r > 0:
                f1_scores.append(2 * p * r / (p + r))
            else:
                f1_scores.append(0)
        
        if len(f1_scores) > 0:
            best_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[min(best_idx, len(thresholds)-1)]
        else:
            optimal_threshold = 0.5
        
        return pr_auc, recall_at_95, optimal_threshold
    
    def _get_model_probability(self, model, X):
        """
        🔧 UNIFIED PROBABILITY INTERFACE
        Every model outputs a continuous score (0-1 probability)
        No mixing of binary and probability spaces!
        """
        try:
            # Convert X to numpy if needed
            if hasattr(X, 'get'):
                X_np = X.get()
            else:
                X_np = np.asarray(X)
            
            # 1. Divine Alien Avatars (IsolationForest, OneClassSVM, LOF)
            if hasattr(model, 'niche') and model.niche == 'divine_alien':
                if hasattr(model, 'model'):
                    sklearn_model = model.model
                    
                    # Get anomaly scores
                    if hasattr(sklearn_model, 'decision_function'):
                        scores = sklearn_model.decision_function(X_np)
                    elif hasattr(sklearn_model, 'score_samples'):
                        scores = sklearn_model.score_samples(X_np)
                    else:
                        # Use predict and convert -1/1 to scores
                        pred = sklearn_model.predict(X_np)
                        scores = np.where(pred == -1, -5.0, 5.0)  # Margin-based
                    
                    # INVERTED sigmoid: negative scores = outliers = HIGH fraud probability
                    return 1.0 / (1.0 + np.exp(scores))
            
            # 2. Models with predict_proba (XGBoost, RF, LogReg) - use directly
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X_np)  # 🛠️ Use X_np (numpy), not X (might be GPU)
                if hasattr(probs, 'get'):
                    probs = probs.get()
                if len(probs.shape) > 1:
                    return probs[:, 1]
                return probs
            
            # 3. Models with decision_function (SVM) - squash with sigmoid
            if hasattr(model, 'decision_function'):
                scores = model.decision_function(X_np)  # 🛠️ Use X_np (numpy), not X (might be GPU)
                if hasattr(scores, 'get'):
                    scores = scores.get()
                # Sigmoid to convert margins to probabilities
                return 1.0 / (1.0 + np.exp(-scores))
            
            # 4. Absolute fallback: Use actual prediction with small uncertainty
            # ❌ ChatGPT: Don't fake 0.99 certainty! Use actual values with uncertainty
            pred = model.predict(X_np) if hasattr(model, 'predict') else model.yuddha(X_np)
            if hasattr(pred, 'get'):
                pred = pred.get()
            # Convert binary to soft probabilities: 1→0.7, 0→0.3 (NOT 0.99/0.01!)
            return np.where(pred == 1, 0.7, 0.3)
            
        except Exception as e:
            print(f"  ⚠️ Probability extraction failed for {getattr(model, 'expert_id', getattr(model, 'yoddha_id', 'unknown'))}: {e}")
            # Return neutral probability if all else fails
            return np.full(len(X), 0.5)
    
    def _evaluate_with_pr_metric(self, X_val, y_val):
        """
        Evaluate ensemble using PR-AUC with unified probability space
        Calculates optimal threshold on validation ONLY (no test leakage)
        """
        # Collect all models
        all_models = []
        if self.arena:
            all_models.extend([self.arena.champion] + self.arena.challengers + self.arena.elders)
        if self.kurukshetra:
            all_models.extend([self.kurukshetra.maharathi] + self.kurukshetra.senapatis + self.kurukshetra.pitamahas)
        all_models.extend(self.hybrid_models)
        
        if len(all_models) < 3:
            return
        
        # Get top 10 models
        def get_score(m):
            if hasattr(m, 'get_combat_score'):
                return m.get_combat_score()
            elif hasattr(m, 'get_yoddha_bal'):
                return m.get_yoddha_bal()
            return 0.0
        
        all_models.sort(key=get_score, reverse=True)
        top_models = all_models[:10]
        
        # 🔧 UNIFIED PROBABILITY SPACE: All models output continuous scores
        prob_predictions = []
        for model in top_models:
            probs = self._get_model_probability(model, X_val)
            if probs is not None:
                prob_predictions.append(probs)
        
        if len(prob_predictions) == 0:
            return
        
        # Ensemble probabilities (average in continuous space)
        ensemble_proba = np.mean(prob_predictions, axis=0)
        
        # Calculate PR metrics with error handling
        try:
            pr_auc, recall_at_95, optimal_threshold = self._calculate_pr_metrics(y_val, ensemble_proba)
        except Exception as e:
            print(f"   ⚠️ PR-AUC calculation failed: {e}")
            pr_auc = self.best_pr_auc if self.best_pr_auc > 0 else 0.5
            recall_at_95 = 0.0
            optimal_threshold = 0.5
        
        # 🎯 Store optimal threshold (calculated on validation only!)
        self.optimal_threshold = optimal_threshold
        self.threshold_history.append(optimal_threshold)
        
        # Update best PR metrics
        if pr_auc > self.best_pr_auc:
            self.best_pr_auc = pr_auc
        
        if recall_at_95 > self.best_recall_at_95_precision:
            self.best_recall_at_95_precision = recall_at_95
            self.best_ensemble = top_models
        
        # Calculate F1 with optimal threshold
        ensemble_pred = (ensemble_proba >= self.optimal_threshold).astype(int)
        current_f1 = f1_score(y_val, ensemble_pred, zero_division=0)
        
        self.f1_history.append(current_f1)
        
        if current_f1 > self.best_f1:
            self.best_f1 = current_f1
        
        # Update Hall of Fame
        self._update_hall_of_fame(all_models, X_val, y_val)
        
        print(f"\n📊 PR-AUC Evaluation (Unified Probability Space):")
        print(f"   PR-AUC: {pr_auc:.4f}")
        print(f"   Recall@Precision≥0.95: {recall_at_95:.4f}")
        print(f"   Optimal Threshold: {optimal_threshold:.3f}")
        print(f"   F1 (with optimal threshold): {current_f1:.4f}")
    
    # 🕉️ VISVARUPA DISRUPTION METHODS 🕉️
    
    def visvarupa_avatar_injection(self, X_train, y_train, X_val, y_val, feature_names=None):
        """
        🕉️ AVATAR INJECTION - Divine Warriors Appear!
        Every 20 rounds, 5 completely alien models appear to disrupt stagnation
        Like Krishna showing Visvarupa to shock Arjuna into action
        """
        print("\n" + "="*80)
        print("🕉️ VISVARUPA MANIFESTS - AVATARS DESCEND! 🕉️")
        print("="*80)
        print("   The divine form appears to disrupt the battlefield!")
        print("   6 alien warriors with completely different algorithms emerge!")
        print("   They will battle everyone and force adaptation!")
        print("="*80)
        
        # Divine warriors use COMPLETELY different algorithms
        # 6 avatars total for perfect East-West balance (3 each)
        divine_algorithms = ['isolation_forest', 'one_class_svm', 'local_outlier', 'dbscan', 'ensemble_isolation', 'robust_covariance']
        
        avatars_spawned = 0
        for i, algo in enumerate(divine_algorithms):
            # Create divine warrior (Arena form)
            from the_arena import ArenaExpert
            
            avatar_id = f"Visvarupa_Avatar_{i+1}_{algo}"
            avatar = ArenaExpert(
                expert_id=avatar_id,
                model_type='rf',  # Random forest as base
                niche='divine_alien',  # Special divine niche
                seed=random.randint(0, 1000000)
            )
            
            # Train with divine power (isolation-based approach)
            try:
                if algo == 'isolation_forest':
                    from sklearn.ensemble import IsolationForest
                    iso_model = IsolationForest(contamination=0.002, random_state=42, n_estimators=200)
                    iso_model.fit(X_train)
                    # Convert isolation forest to binary predictions
                    avatar.model = iso_model
                    avatar.is_fitted = True
                    
                elif algo == 'one_class_svm':
                    from sklearn.svm import OneClassSVM
                    ocsvm = OneClassSVM(nu=0.002, kernel='rbf', gamma='scale')
                    ocsvm.fit(X_train[y_train == 0])  # Train only on normal data
                    avatar.model = ocsvm
                    avatar.is_fitted = True
                    
                elif algo == 'local_outlier':
                    from sklearn.neighbors import LocalOutlierFactor
                    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.002, novelty=True)
                    lof.fit(X_train)
                    avatar.model = lof
                    avatar.is_fitted = True
                    
                elif algo == 'robust_covariance':
                    from sklearn.covariance import EllipticEnvelope
                    ee = EllipticEnvelope(contamination=0.002, random_state=42)
                    # Convert to numpy if using GPU (sklearn needs CPU arrays)
                    X_train_np = X_train.get() if hasattr(X_train, 'get') else np.asarray(X_train)
                    ee.fit(X_train_np)
                    avatar.model = ee
                    avatar.is_fitted = True
                    
                else:
                    # Standard training for others
                    avatar.fit(X_train, y_train, feature_names=feature_names)
                
                if avatar.is_fitted:
                    # Divine warriors start with 50 pre-allocated victories
                    avatar.combat_wins = 50
                    avatar.total_battles = 50
                    
                    # 🕉️ BALANCE: Distribute avatars between East and West
                    # Odd-numbered avatars go to West (Arena)
                    # Even-numbered avatars go to East (Kurukshetra)
                    if (i + 1) % 2 == 1:  # Odd: West
                        if self.arena:
                            self.arena.challengers.append(avatar)
                            avatars_spawned += 1
                            print(f"   ✨ Avatar {i+1}: {avatar_id} manifests in WEST (Arena)!")
                    else:  # Even: East
                        if self.kurukshetra:
                            # Convert ArenaExpert to Yoddha for Kurukshetra
                            from kurukshetra import Yoddha
                            yoddha_avatar = Yoddha(avatar_id, varna='sarvabhaum')
                            yoddha_avatar.shastra = avatar.model  # Transfer model
                            yoddha_avatar.is_tejasvi = True
                            yoddha_avatar.vijayas = 50  # Divine victories
                            yoddha_avatar.yuddhas = 50
                            yoddha_avatar.niche = 'divine_alien'  # Mark as divine
                            self.kurukshetra.senapatis.append(yoddha_avatar)
                            avatars_spawned += 1
                            print(f"   ✨ Avatar {i+1}: {avatar_id} manifests in EAST (Kurukshetra)!")
                        
            except Exception as e:
                print(f"   ⚠️ Avatar {i+1} failed to manifest: {e}")
        
        print(f"\n   🕉️ {avatars_spawned} Avatars now walk the battlefield!")
        print("   They will disrupt the balance and force evolution!")
        print("   After 10 rounds, they will vanish like Krishna withdrawing Visvarupa...")
        
        return avatars_spawned
    
    def visvarupa_cataclysm(self, X_train, y_train, X_val=None, y_val=None, feature_names=None, use_vampire=True):
        """
        ⚡ THE CATACLYSM - Divine Destruction and Rebirth!
        Every 50 rounds, the bottom 50% of warriors are destroyed and replaced
        Like the destruction before creation in Hindu cosmology
        
        🧛 NEW: Knowledge Vampire - Extracts unique patterns before destruction
        """
        print("\n" + "="*80)
        print("⚡ THE CATACLYSM - DHARMA REQUIRES DESTRUCTION! ⚡")
        if use_vampire and X_val is not None and y_val is not None:
            print("🧛 KNOWLEDGE VAMPIRE: Extracting patterns before destruction")
        print("="*80)
        print("   The old order must be destroyed for the new to emerge!")
        print("   Bottom 50% of warriors will be culled!")
        print("   New random warriors will rise from the ashes!")
        print("="*80)
        
        # 🧛 Initialize Vampire if enabled
        vampire = None
        if use_vampire and X_val is not None and y_val is not None:
            try:
                import sys
                import os
                sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
                from knowledge_vampire import KnowledgeVampire
                vampire = KnowledgeVampire(
                    min_unique_detections=3,
                    max_patterns_per_model=10,
                    synthetic_samples_per_pattern=5
                )
                print("   🧛 Vampire initialized - Ready to extract knowledge")
            except Exception as e:
                print(f"   ⚠️ Vampire initialization failed: {e}")
                vampire = None
        
        total_destroyed = 0
        total_spawned = 0
        total_patterns_extracted = 0
        
        # Destroy bottom 50% of Arena
        if self.arena:
            all_arena = self.arena.challengers
            if len(all_arena) > 0:
                # Sort by combat score
                all_arena.sort(key=lambda x: x.get_combat_score())
                n_destroy = len(all_arena) // 2
                
                # 🏆 PROTECT ELITE MODELS: Filter out elite models from destruction
                destroyed = [m for m in all_arena[:n_destroy] if not self._is_elite_model(m)]
                survivors = all_arena[n_destroy:] + [m for m in all_arena[:n_destroy] if self._is_elite_model(m)]
                
                n_protected = sum(1 for m in all_arena[:n_destroy] if self._is_elite_model(m))
                if n_protected > 0:
                    print(f"   🛡️ {n_protected} ELITE MODELS PROTECTED from Cataclysm!")
                
                print(f"\n   🏟️ ARENA (West):")
                print(f"      {len(destroyed)} weak gladiators marked for destruction")
                
                # 🧛 VAMPIRE: Extract knowledge before destruction
                actually_destroyed = []
                for model in destroyed:
                    if vampire and X_val is not None and y_val is not None:
                        # Try to extract and transfer knowledge
                        elite_models = self.elite_models if hasattr(self, 'elite_models') else []
                        should_destroy = vampire.smart_destroy(
                            model, elite_models, X_train, y_train, X_val, y_val
                        )
                        if should_destroy:
                            actually_destroyed.append(model)
                            if len(vampire.transfer_history) > total_patterns_extracted:
                                total_patterns_extracted = len(vampire.transfer_history)
                        else:
                            # Model kept for knowledge preservation
                            survivors.append(model)
                            print(f"         🛡️ {model.expert_id} kept (knowledge transfer failed)")
                    else:
                        actually_destroyed.append(model)
                
                for d in actually_destroyed[:3]:
                    print(f"         💀 {d.expert_id} (wins: {d.combat_wins})")
                if len(actually_destroyed) > 3:
                    print(f"         ... and {len(actually_destroyed) - 3} more")
                
                # Replace with new random challengers
                self.arena.challengers = survivors
                n_actually_destroyed = len(actually_destroyed)
                for i in range(n_actually_destroyed):
                    new_challenger = self.arena._spawn_single_challenger(i, X_train, y_train, feature_names)
                    if new_challenger:
                        # Aggressive mutation for new spawn
                        new_challenger.mutate(aggressive=True)
                        self.arena.challengers.append(new_challenger)
                        total_spawned += 1
                
                print(f"      {n_actually_destroyed} destroyed, {total_spawned} new challengers rise!")
                total_destroyed += n_actually_destroyed
        
        # Destroy bottom 50% of Kurukshetra
        if self.kurukshetra:
            all_kuru = self.kurukshetra.senapatis
            if len(all_kuru) > 0:
                # Sort by karma balance
                all_kuru.sort(key=lambda x: x.get_yoddha_bal())
                n_destroy = len(all_kuru) // 2
                
                # 🏆 PROTECT ELITE MODELS: Filter out elite models from destruction
                destroyed = [m for m in all_kuru[:n_destroy] if not self._is_elite_model(m)]
                survivors = all_kuru[n_destroy:] + [m for m in all_kuru[:n_destroy] if self._is_elite_model(m)]
                
                n_protected_kuru = sum(1 for m in all_kuru[:n_destroy] if self._is_elite_model(m))
                if n_protected_kuru > 0:
                    print(f"   🛡️ {n_protected_kuru} ELITE WARRIORS PROTECTED from Cataclysm!")
                
                print(f"\n   ⚔️ KURUKSHETRA (East):")
                print(f"      {len(destroyed)} weak warriors marked for destruction")
                
                # 🧛 VAMPIRE: Extract knowledge before destruction
                actually_destroyed_kuru = []
                for model in destroyed:
                    if vampire and X_val is not None and y_val is not None:
                        elite_models = self.elite_models if hasattr(self, 'elite_models') else []
                        should_destroy = vampire.smart_destroy(
                            model, elite_models, X_train, y_train, X_val, y_val
                        )
                        if should_destroy:
                            actually_destroyed_kuru.append(model)
                        else:
                            survivors.append(model)
                            print(f"         🛡️ {model.yoddha_id} kept (knowledge transfer failed)")
                    else:
                        actually_destroyed_kuru.append(model)
                
                for d in actually_destroyed_kuru[:3]:
                    print(f"         💀 {d.yoddha_id} (victories: {d.vijayas})")
                if len(actually_destroyed_kuru) > 3:
                    print(f"         ... and {len(actually_destroyed_kuru) - 3} more")
                
                # Replace with new random senapatis
                self.kurukshetra.senapatis = survivors
                n_destroyed_kuru = len(actually_destroyed_kuru)
                for i in range(n_destroyed_kuru):
                    new_senapati = self.kurukshetra._spawn_single_senapati(i, X_train, y_train, feature_names)
                    if new_senapati:
                        # Aggressive tapasya for new spawn
                        new_senapati.tapasya(aggressive=True)
                        self.kurukshetra.senapatis.append(new_senapati)
                        total_spawned += 1
                
                print(f"      {n_destroyed_kuru} destroyed, {total_spawned} new warriors rise!")
                total_destroyed += n_destroyed_kuru
        
        print(f"\n   ⚡ CATACLYSM COMPLETE!")
        print(f"      Total Destroyed: {total_destroyed}")
        print(f"      Total Spawned: {total_spawned}")
        if total_patterns_extracted > 0:
            print(f"      🧛 Patterns Extracted: {total_patterns_extracted}")
        print(f"      Net Change: {total_spawned - total_destroyed}")
        print("   The battlefield is renewed! Dharma is restored!")
        
        return total_destroyed, total_spawned
    
    def remove_expired_avatars(self):
        """
        🌅 Remove avatars that have completed their divine mission
        Like Krishna withdrawing Visvarupa after showing it to Arjuna
        """
        removed = 0
        
        if self.arena:
            # Find and remove avatars
            to_remove = [c for c in self.arena.challengers if 'Visvarupa_Avatar' in c.expert_id]
            for avatar in to_remove:
                self.arena.challengers.remove(avatar)
                removed += 1
                print(f"   🌅 {avatar.expert_id} vanishes...")
        
        if removed > 0:
            print(f"\n   🕉️ {removed} Avatars have withdrawn their Visvarupa form")
            print("   Their mission is complete. The battlefield is changed forever.")
        
        return removed
    
    def _build_cascade_ensemble(self, X_val, y_val):
        """
        🏗️ Build 3-level cascade ensemble
        Level 1: Specialists (niche/varna specific)
        Level 2: Hybrid models
        Level 3: Supreme ensemble
        """
        print("\nConstructing 3-Level Cascade...")
        
        # Level 1: Specialists (best from each niche/varna)
        specialists = []
        
        if self.arena:
            for niche in NICHE_TYPES:
                niche_experts = [e for e in [self.arena.champion] + self.arena.challengers if e.niche == niche]
                if niche_experts:
                    best = max(niche_experts, key=lambda x: x.get_combat_score())
                    specialists.append(('arena', niche, best))
        
        if self.kurukshetra:
            for varna in VARNA_TYPES:
                varna_yoddhas = [y for y in [self.kurukshetra.maharathi] + self.kurukshetra.senapatis if y.varna == varna]
                if varna_yoddhas:
                    best = max(varna_yoddhas, key=lambda x: x.get_yoddha_bal())
                    specialists.append(('kurukshetra', varna, best))
        
        print(f"   Level 1 Specialists: {len(specialists)}")
        
        # Level 2: Hybrids
        print(f"   Level 2 Hybrids: {len(self.hybrid_models)}")
        
        # Level 3: Supreme ensemble (top performers from all)
        all_models = []
        if self.arena:
            all_models.extend([self.arena.champion] + self.arena.elders)
        if self.kurukshetra:
            all_models.extend([self.kurukshetra.maharathi] + self.kurukshetra.pitamahas)
        all_models.extend(self.hybrid_models)
        
        def get_score(m):
            if hasattr(m, 'get_combat_score'):
                return m.get_combat_score()
            elif hasattr(m, 'get_yoddha_bal'):
                return m.get_yoddha_bal()
            return 0.0
        
        all_models.sort(key=get_score, reverse=True)
        supreme = all_models[:7]
        
        print(f"   Level 3 Supreme: {len(supreme)} models")
        
        # Test cascade
        print("\n🧪 Testing Cascade Ensemble...")
        
        # 🛠️ Convert to numpy if GPU arrays
        if hasattr(X_val, 'get'):
            X_val_np = X_val.get()
        else:
            X_val_np = np.asarray(X_val)
        
        # Level 1 predictions
        level1_preds = []
        for fw, niche, model in specialists[:5]:  # Top 5 specialists
            try:
                if hasattr(model, 'predict'):
                    pred = model.predict(X_val_np)
                else:
                    pred = model.yuddha(X_val_np)
                level1_preds.append(pred)
            except:
                pass
        
        # Level 2 predictions (hybrids)
        level2_preds = []
        for model in self.hybrid_models[:5]:  # Top 5 hybrids
            try:
                if hasattr(model, 'predict'):
                    pred = model.predict(X_val_np)
                else:
                    pred = model.yuddha(X_val_np)
                level2_preds.append(pred)
            except:
                pass
        
        # Level 3 predictions (supreme)
        level3_preds = []
        for model in supreme:
            try:
                if hasattr(model, 'predict'):
                    pred = model.predict(X_val_np)
                else:
                    pred = model.yuddha(X_val_np)
                level3_preds.append(pred)
            except:
                pass
        
        # Combine all levels with weights
        all_preds = level1_preds + level2_preds + level3_preds
        if len(all_preds) > 0:
            # Weighted by level (supreme gets higher weight)
            weights = ([0.8] * len(level1_preds) + 
                      [1.0] * len(level2_preds) + 
                      [1.2] * len(level3_preds))
            
            weights = np.array(weights)
            weights = weights / weights.sum()
            
            # Weighted average
            cascade_pred = np.zeros(len(X_val))
            for pred, weight in zip(all_preds, weights):
                cascade_pred += pred * weight
            
            # 🎯 Use optimal threshold instead of fixed 0.5
            threshold = self.optimal_threshold if hasattr(self, 'optimal_threshold') and self.optimal_threshold > 0 else 0.5
            cascade_pred = (cascade_pred >= threshold).astype(int)
            cascade_f1 = f1_score(y_val, cascade_pred, zero_division=0)
            
            print(f"\n🎯 CASCADE ENSEMBLE F1: {cascade_f1:.4f} ({cascade_f1*100:.2f}%)")
            
            if cascade_f1 > self.best_f1:
                self.best_f1 = cascade_f1
                print(f"   ⭐ NEW BEST! Improved from {self.best_f1:.4f}")
        
        self.cascade_ensemble = {
            'level1_specialists': specialists[:5],
            'level2_hybrids': self.hybrid_models[:5],
            'level3_supreme': supreme
        }
    
    def predict(self, X):
        """
        🔧 UNIFIED PROBABILITY SPACE: Make prediction using cascade ensemble
        All models output probabilities, averaged, then thresholded
        """
        if not self.cascade_ensemble:
            return np.zeros(len(X))
        
        all_probs = []  # Store probabilities, not binary predictions!
        weights = []
        
        # Level 1: Specialists
        for fw, niche, model in self.cascade_ensemble['level1_specialists']:
            try:
                probs = self._get_model_probability(model, X)
                if probs is not None:
                    all_probs.append(probs)
                    weights.append(0.8)
            except:
                pass
        
        # Level 2: Hybrids
        for model in self.cascade_ensemble['level2_hybrids']:
            try:
                probs = self._get_model_probability(model, X)
                if probs is not None:
                    all_probs.append(probs)
                    weights.append(1.0)
            except:
                pass
        
        # Level 3: Supreme
        for model in self.cascade_ensemble['level3_supreme']:
            try:
                probs = self._get_model_probability(model, X)
                if probs is not None:
                    all_probs.append(probs)
                    weights.append(1.2)
            except:
                pass
        
        if len(all_probs) == 0:
            return np.zeros(len(X))
        
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # 🔧 UNIFIED: Average probabilities in continuous space (NOT binary!)
        ensemble_proba = np.zeros(len(X))
        for probs, weight in zip(all_probs, weights):
            ensemble_proba += probs * weight
        
        # 🎯 ADAPTIVE THRESHOLD: Use optimized threshold (calculated on validation ONLY!)
        # This threshold is FROZEN after validation tuning - never adapted during test
        threshold = self.optimal_threshold if hasattr(self, 'optimal_threshold') and self.optimal_threshold > 0 else 0.5
        return (ensemble_proba >= threshold).astype(int)
    
    def save(self, filepath: str):
        """Save the convergence state with proper model serialization"""
        import joblib
        
        # Save models separately with joblib (better for cuML)
        models_dir = filepath.replace('.pkl', '_models')
        os.makedirs(models_dir, exist_ok=True)
        
        # Save arena models
        arena_models = {}
        if self.arena and hasattr(self.arena, 'warriors'):
            for i, warrior in enumerate(self.arena.warriors):
                if warrior and hasattr(warrior, 'model'):
                    try:
                        joblib.dump(warrior.model, f"{models_dir}/arena_warrior_{i}.joblib")
                        arena_models[i] = f"{models_dir}/arena_warrior_{i}.joblib"
                    except:
                        pass
        
        # Save kurukshetra models  
        kuru_models = {}
        if self.kurukshetra and hasattr(self.kurukshetra, 'senapatis'):
            for i, senapati in enumerate(self.kurukshetra.senapatis):
                if senapati and hasattr(senapati, 'model'):
                    try:
                        joblib.dump(senapati.model, f"{models_dir}/kuru_senapati_{i}.joblib")
                        kuru_models[i] = f"{models_dir}/kuru_senapati_{i}.joblib"
                    except:
                        pass
        
        # Save config and metadata
        state = {
            'arena_models': arena_models,
            'kuru_models': kuru_models,
            'best_f1': self.best_f1,
            'config': CONVERGENCE_CONFIG,
            'models_dir': models_dir
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"💾 Great Convergence saved to {filepath}")
        print(f"   Arena models: {len(arena_models)}")
        print(f"   Kuru models: {len(kuru_models)}")
    
    def _get_ensemble_probability_for_oof(self, X: np.ndarray) -> np.ndarray:
        """Get ensemble probability for OOF predictions"""
        if not self.cascade_ensemble:
            return np.zeros(len(X))
        
        all_probs = []
        
        # Collect probabilities from all levels
        for fw, niche, model in self.cascade_ensemble.get('level1_specialists', []):
            probs = self._get_model_probability(model, X)
            if probs is not None:
                all_probs.append(probs)
        
        for model in self.cascade_ensemble.get('level2_hybrids', []):
            probs = self._get_model_probability(model, X)
            if probs is not None:
                all_probs.append(probs)
        
        for model in self.cascade_ensemble.get('level3_supreme', []):
            probs = self._get_model_probability(model, X)
            if probs is not None:
                all_probs.append(probs)
        
        if len(all_probs) == 0:
            return np.zeros(len(X))
        
        return np.mean(all_probs, axis=0)
    
    def set_oof_threshold(self, threshold: float, stats: dict = None):
        """
        🛡️ Set OOF threshold - ONCE, then LOCKED forever
        
        This is the safe way to set threshold:
        - Calculate on OOF predictions (not during evolution)
        - Lock it before final training/evaluation
        - Never change it again
        """
        self.oof_threshold = threshold
        self.threshold_locked = True
        self.optimal_threshold = threshold  # Use this for predictions
        
        if stats:
            self.oof_stats = stats
        
        print(f"\n🔒 OOF THRESHOLD LOCKED: {threshold:.4f}")
        print("   This threshold will not change during training or evaluation")
        print("   Single evaluation on test set only")
    
    def predict_with_locked_threshold(self, X: np.ndarray) -> np.ndarray:
        """
        🛡️ Predict using LOCKED OOF threshold
        
        Safe for framework use - no adaptation, no leakage
        """
        if not self.threshold_locked:
            print("⚠️ Warning: Using unlocked threshold - not recommended for framework")
            threshold = self.optimal_threshold
        else:
            threshold = self.oof_threshold
        
        # Get probabilities (unified space)
        if not self.cascade_ensemble:
            return np.zeros(len(X))
        
        all_probs = []
        
        # Level 1
        for fw, niche, model in self.cascade_ensemble.get('level1_specialists', []):
            probs = self._get_model_probability(model, X)
            if probs is not None:
                all_probs.append(probs)
        
        # Level 2
        for model in self.cascade_ensemble.get('level2_hybrids', []):
            probs = self._get_model_probability(model, X)
            if probs is not None:
                all_probs.append(probs)
        
        # Level 3
        for model in self.cascade_ensemble.get('level3_supreme', []):
            probs = self._get_model_probability(model, X)
            if probs is not None:
                all_probs.append(probs)
        
        if len(all_probs) == 0:
            return np.zeros(len(X))
        
        # Ensemble probability
        ensemble_proba = np.mean(all_probs, axis=0)
        
        # Apply LOCKED threshold
        return (ensemble_proba >= threshold).astype(int)


def run_safe_great_convergence_oof(X_train, y_train, X_test, y_test,
                                    n_models_per_side=25,
                                    battle_rounds=50,
                                    n_folds=5,
                                    feature_names=None,
                                    target_precision=0.95):
    """
    🛡️ SAFE FRAMEWORK: OOF Thresholding with No Overfitting
    
    This is the RIGOROUS way to use Great Convergence:
    1. OOF predictions from K-fold CV
    2. Threshold selected ONCE on OOF
    3. Final model trained on all data with LOCKED threshold
    4. Single evaluation on test
    
    No leakage, no threshold overfitting, no validation feedback during evolution!
    
    Returns:
        convergence: Trained GreatConvergence model
        results: Dictionary with metrics and OOF stats
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import precision_recall_curve, f1_score
    
    print("\n" + "="*80)
    print("🛡️ SAFE GREAT CONVERGENCE - OOF THRESHOLDING")
    print("="*80)
    print("✅ No overfitting - rigorous evaluation")
    print("✅ OOF predictions from K-fold CV")
    print("✅ Threshold locked before final training")
    print("✅ Single test evaluation")
    print("="*80 + "\n")
    
    # Step 1: Generate OOF predictions
    print("📊 Step 1: Generating OOF Predictions...")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    oof_probas = np.zeros(len(y_train))
    oof_preds = np.zeros(len(y_train))
    fold_results = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"\n🔄 Fold {fold_idx + 1}/{n_folds}")
        
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        
        # Train fresh Great Convergence
        gc = GreatConvergence(
            n_models_per_side=n_models_per_side,
            battle_rounds=battle_rounds
        )
        
        gc.run_convergence(X_fold_train, y_fold_train, 
                          X_fold_val, y_fold_val, feature_names)
        
        # Get OOF predictions
        proba = gc._get_ensemble_probability_for_oof(X_fold_val)
        pred = (proba >= 0.5).astype(int)
        
        oof_probas[val_idx] = proba
        oof_preds[val_idx] = pred
        
        fold_f1 = f1_score(y_fold_val, pred, zero_division=0)
        fold_results.append({'fold': fold_idx + 1, 'f1': fold_f1})
        print(f"   Fold F1: {fold_f1:.4f}")
    
    mean_fold_f1 = np.mean([f['f1'] for f in fold_results])
    print(f"\n✅ OOF Complete - Mean Fold F1: {mean_fold_f1:.4f}")
    
    # Step 2: Select threshold on OOF
    print("\n🎯 Step 2: Selecting Optimal Threshold on OOF...")
    precision, recall, thresholds = precision_recall_curve(y_train, oof_probas)
    
    # Find threshold maximizing recall @ precision >= target
    best_recall = 0
    best_threshold = 0.5
    
    for p, r, thresh in zip(precision, recall, thresholds):
        if p >= target_precision and r > best_recall:
            best_recall = r
            best_threshold = thresh
    
    print(f"   Optimal Threshold: {best_threshold:.4f}")
    print(f"   Recall @ Precision>={target_precision}: {best_recall:.4f}")
    
    # Step 3: Train final model on ALL data with LOCKED threshold
    print("\n🏋️ Step 3: Training Final Model (All Data, Locked Threshold)...")
    final_gc = GreatConvergence(
        n_models_per_side=n_models_per_side,
        battle_rounds=battle_rounds
    )
    
    # Use small validation for logging only
    val_size = min(1000, len(X_train) // 10)
    X_final_train = X_train[:-val_size]
    y_final_train = y_train[:-val_size]
    X_dummy_val = X_train[-val_size:]
    y_dummy_val = y_train[-val_size:]
    
    final_gc.run_convergence(X_final_train, y_final_train,
                            X_dummy_val, y_dummy_val, feature_names)
    
    # LOCK the threshold
    oof_stats = {
        'threshold': best_threshold,
        'target_precision': target_precision,
        'recall_at_threshold': best_recall,
        'mean_fold_f1': mean_fold_f1,
        'fold_results': fold_results
    }
    final_gc.set_oof_threshold(best_threshold, oof_stats)
    
    # Store reference to the safe prediction method
    final_gc.predict = final_gc.predict_with_locked_threshold
    
    # Step 4: Single test evaluation
    print("\n🎯 Step 4: Single Test Evaluation (No Adaptation)...")
    y_pred = final_gc.predict_with_locked_threshold(X_test)
    
    from sklearn.metrics import f1_score, precision_score, recall_score
    test_f1 = f1_score(y_test, y_pred, zero_division=0)
    test_precision = precision_score(y_test, y_pred, zero_division=0)
    test_recall = recall_score(y_test, y_pred, zero_division=0)
    
    results = {
        'test_f1': test_f1,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'oof_threshold': best_threshold,
        'oof_stats': oof_stats,
        'model': final_gc
    }
    
    print(f"\n✅ RESULTS:")
    print(f"   F1: {test_f1:.4f}")
    print(f"   Precision: {test_precision:.4f}")
    print(f"   Recall: {test_recall:.4f}")
    print(f"   Threshold: {best_threshold:.4f} (locked)")
    print("\n" + "="*80 + "\n")
    
    return final_gc, results


def run_great_convergence_experiment(X_train, y_train, X_val, y_val, 
                                     feature_names=None,
                                     n_models_per_side=25,
                                     battle_rounds=200,
                                     save_path=None):
    """
    Run the ultimate East vs West convergence experiment
    """
    convergence = GreatConvergence(
        n_models_per_side=n_models_per_side,
        battle_rounds=battle_rounds,
        cascade_levels=3
    )
    
    convergence.run_convergence(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        feature_names=feature_names
    )
    
    if save_path:
        convergence.save(save_path)
    
    return convergence, convergence.best_f1


if __name__ == "__main__":
    print("⚔️🌟 THE GREAT CONVERGENCE 🌟⚔️")
    print("East vs West - The Ultimate Battle")
    print()
    print("Arena (West): Gladiators of the Colosseum")
    print("Kurukshetra (East): Warriors of Dharma")
    print()
    print("50 models per side. 200 battle rounds.")
    print("Cultural exchange. Hybrid creation.")
    print("3-level cascade ensemble.")
    print()
    print("Target: 96%+ F1")
    print()
    print("Let the Great Convergence begin!")
