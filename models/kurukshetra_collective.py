"""
⚔️ KURUKSHETRA V2 - TRUE COLLECTIVE WARFARE
===========================================
MAJOR ARCHITECTURAL CHANGES:
- Group combat instead of individual duels
- Shared karma (team performance)
- Guru-Shishya lineage (master-student)
- Collective decision making
- Persistent memory across reincarnation
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import random
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

# Import base Yoddha
try:
    # Try relative import first (when imported as part of package)
    from .kurukshetra import Yoddha, Shastrartha, VARNA_TYPES, get_yoddha_title, DHARMA_CONFIG
except ImportError:
    # Fall back to absolute import (when run directly)
    from kurukshetra import Yoddha, Shastrartha, VARNA_TYPES, get_yoddha_title, DHARMA_CONFIG


@dataclass 
class SenaFormation:
    """
    🛡️ सेना रचना (Sena Formation)
    A group of 3-5 warriors fighting together as a unit
    """
    formation_id: str
    warriors: List[Yoddha]
    formation_type: str  # 'chakra', 'garuda', 'makara', etc.
    created_round: int
    
    # Collective karma
    collective_vijayas: int = 0
    collective_parajayas: int = 0
    shared_karma_score: float = 0.0
    collective_f1: float = 0.0  # Track F1 for overfitting detection
    
    # Guru-Shishya lineage
    guru: Optional[Yoddha] = None  # Master
    shishyas: List[Yoddha] = field(default_factory=list)  # Students
    
    def __post_init__(self):
        if len(self.warriors) > 0:
            # Senior most warrior becomes guru
            self.guru = max(self.warriors, key=lambda w: w.vijayas)
            self.shishyas = [w for w in self.warriors if w != self.guru]
    
    def collective_predict(self, X: np.ndarray) -> np.ndarray:
        """
        🗳️ Collective Decision Making
        All warriors vote, guru has 2x weight
        """
        # 🛠️ Convert GPU arrays to numpy for sklearn models
        if hasattr(X, 'get'):
            X = X.get()
        else:
            X = np.asarray(X)
        
        votes = []
        weights = []
        
        for warrior in self.warriors:
            pred = warrior.yuddha(X)
            votes.append(pred)
            
            # Guru has double weight
            if warrior == self.guru:
                weights.append(2.0)
            else:
                weights.append(1.0)
        
        # Weighted voting
        votes = np.array(votes)
        weights = np.array(weights)
        weighted_pred = np.average(votes, axis=0, weights=weights)
        
        return np.round(weighted_pred).astype(int)
    
    def share_karma(self, victory: bool, f1_score: float):
        """
        🕉️ Shared Karma Distribution
        All warriors share the result
        """
        if victory:
            self.collective_vijayas += 1
            karma_boost = f1_score * 0.1
        else:
            self.collective_parajayas += 1
            karma_boost = -0.05
        
        self.shared_karma_score += karma_boost
        
        # Distribute to individual warriors
        for warrior in self.warriors:
            # Guru gets 50% more karma
            if warrior == self.guru:
                warrior.vijayas += 1 if victory else 0
            else:
                # Shishyas learn from guru
                warrior.vijayas += 0.5 if victory else 0
    
    def share_weighted_karma(self, victory: bool, f1_score: float, X_batch: np.ndarray, y_batch: np.ndarray):
        """
        🎯 WEIGHTED Karma Distribution (Anti-Carry Mechanism)
        Warriors get karma based on INDIVIDUAL contribution, not just group win
        Prevents weak warriors from being carried
        """
        # 🛠️ Convert GPU arrays to numpy for sklearn models
        if hasattr(X_batch, 'get'):
            X_batch = X_batch.get()
        else:
            X_batch = np.asarray(X_batch)
        
        from sklearn.metrics import f1_score as sk_f1
        
        # Calculate individual F1 scores
        individual_f1s = {}
        for warrior in self.warriors:
            try:
                pred = warrior.yuddha(X_batch)
                individual_f1s[warrior.yoddha_id] = sk_f1(y_batch, pred, zero_division=0)
            except:
                individual_f1s[warrior.yoddha_id] = 0.0
        
        # Calculate contribution ratio
        avg_individual_f1 = np.mean(list(individual_f1s.values()))
        collective_f1 = f1_score
        
        if victory:
            self.collective_vijayas += 1
            # Warriors who helped the group win get more karma
            for warrior in self.warriors:
                individual_f1 = individual_f1s[warrior.yoddha_id]
                
                # Contribution ratio: how much did they help?
                if collective_f1 > avg_individual_f1:
                    contribution = individual_f1 / collective_f1
                else:
                    contribution = 0.5  # Neutral
                
                # Guru bonus + contribution bonus
                if warrior == self.guru:
                    karma = int(1.5 + contribution)  # Guru gets base 1.5
                else:
                    karma = int(0.5 + contribution)  # Shishya gets base 0.5
                
                warrior.vijayas += max(karma, 0)  # No negative karma
        else:
            self.collective_parajayas += 1
            # Even in defeat, good individual performance gets some karma
            for warrior in self.warriors:
                individual_f1 = individual_f1s[warrior.yoddha_id]
                if individual_f1 > 0.7:  # Good individual performance despite group loss
                    warrior.vijayas += 0.5  # Consolation karma
                    print(f"    💪 {warrior.yoddha_id} fought well despite loss (F1: {individual_f1:.3f})")
    
    def transfer_gyan(self):
        """
        📚 Guru-Shishya Knowledge Transfer
        Master teaches students
        """
        if not self.guru or not self.shishyas:
            return
        
        # Guru transfers shastras to shishyas
        for shishya in self.shishyas:
            for shastra in self.guru.shastras_spoken[-5:]:  # Last 5 teachings
                shishya.shun_shastra(shastra)
                print(f"    📚 {self.guru.yoddha_id} teaches {shishya.yoddha_id}: {shastra.upadesha[:50]}...")


class KurukshetraCollective:
    """
    ⚔️ कुरुक्षेत्र - TRUE COLLECTIVE BATTLEFIELD
    
    Unlike Arena's individual duels, Kurukshetra uses:
    - Group formations (3-5 warriors)
    - Collective decision making
    - Shared karma
    - Guru-Shishya lineages
    - Reincarnation with memory
    """
    
    def __init__(self, n_senapatis: int = 20, yuddha_rounds: int = 50):
        self.n_senapatis = n_senapatis
        self.yuddha_rounds = yuddha_rounds
        
        self.maharathi: Optional[Yoddha] = None
        self.senapatis: List[Yoddha] = []
        self.pitamahas: List[Yoddha] = []
        
        # NEW: Formations (groups)
        self.formations: List[SenaFormation] = []
        self.formation_size = 3  # 3 warriors per formation
        
        # Collective memory (persists across reincarnation)
        self.kshetra_smriti = {
            'successful_varnas': {},  # Which varnas worked well
            'failed_strategies': [],   # What didn't work
            'guru_lineages': {}       # Master-student chains
        }
        
        print("⚔️ KURUKSHETRA COLLECTIVE INITIALIZED")
        print("   Group Combat: ENABLED")
        print("   Shared Karma: ENABLED")
        print("   Guru-Shishya Lineage: ENABLED")
        print("   Collective Memory: ENABLED")
    
    def create_formations(self):
        """Group warriors into formations of 3"""
        all_warriors = [self.maharathi] + self.senapatis
        random.shuffle(all_warriors)
        
        self.formations = []
        for i in range(0, len(all_warriors), self.formation_size):
            group = all_warriors[i:i+self.formation_size]
            if len(group) >= 2:  # Need at least 2 for a formation
                formation = SenaFormation(
                    formation_id=f"Sena_{i//self.formation_size}_{random.randint(1000,9999)}",
                    warriors=group,
                    formation_type=random.choice(['chakra', 'garuda', 'makara']),
                    created_round=0
                )
                self.formations.append(formation)
                print(f"   🛡️ Formation {formation.formation_id}: {len(group)} warriors")
    
    def shuffle_formations(self):
        """
        🔄 Shuffle warriors into new formations
        Prevents static overfitting, maintains diversity
        """
        print("\n🔄 Shuffling Formations (Anti-Overfitting)...")
        
        # Collect all warriors
        all_warriors = []
        for formation in self.formations:
            all_warriors.extend(formation.warriors)
        
        # Sort by individual performance (not collective!)
        all_warriors.sort(key=lambda w: w.vijayas / max(w.total_yuddhas, 1), reverse=True)
        
        # Shuffle to create diversity
        random.shuffle(all_warriors)
        
        # Recreate formations with mixed warriors
        self.formations = []
        for i in range(0, len(all_warriors), self.formation_size):
            group = all_warriors[i:i+self.formation_size]
            if len(group) >= 2:
                formation = SenaFormation(
                    formation_id=f"Sena_{i//self.formation_size}_shuffled_{random.randint(1000,9999)}",
                    warriors=group,
                    formation_type=random.choice(['chakra', 'garuda', 'makara']),
                    created_round=0
                )
                self.formations.append(formation)
        
        print(f"   ✓ {len(self.formations)} new formations created")
        print("   Warriors shuffled to prevent group overfitting!")
    
    def evaluate_individual_performance(self, X_val, y_val):
        """
        📊 Track individual performance (not just collective)
        Ensures weak warriors don't get carried
        """
        # 🛠️ Convert GPU arrays to numpy for sklearn models
        if hasattr(X_val, 'get'):
            X_val = X_val.get()
        else:
            X_val = np.asarray(X_val)
        
        print("\n📊 Evaluating Individual Performance...")
        
        individual_scores = {}
        
        for formation in self.formations:
            for warrior in formation.warriors:
                # Test warrior ALONE (not in formation)
                try:
                    pred = warrior.yuddha(X_val)
                    from sklearn.metrics import f1_score
                    f1 = f1_score(y_val, pred, zero_division=0)
                    individual_scores[warrior.yoddha_id] = f1
                    
                    # Penalize warriors carried by formation
                    collective_f1 = formation.collective_f1 if hasattr(formation, 'collective_f1') else f1
                    if collective_f1 > f1 + 0.15:  # Big gap = being carried!
                        print(f"   ⚠️ {warrior.yoddha_id} is being carried (individual: {f1:.3f}, collective: {collective_f1:.3f})")
                        warrior.vijayas *= 0.8  # Penalty!
                        
                except:
                    individual_scores[warrior.yoddha_id] = 0.0
        
        return individual_scores
    
    def group_yuddha(self, X_val, y_val):
        """
        ⚔️ GROUP BATTLE (not individual duels)
        Formations battle each other
        """
        if len(self.formations) < 2:
            return
        
        # Random formations battle
        formation_a, formation_b = random.sample(self.formations, 2)
        
        batch_size = min(500, len(X_val))
        indices = np.random.choice(len(X_val), batch_size, replace=False)
        X_batch = X_val[indices]
        y_batch = y_val[indices]
        
        # Collective predictions
        pred_a = formation_a.collective_predict(X_batch)
        pred_b = formation_b.collective_predict(X_batch)
        
        from sklearn.metrics import f1_score
        f1_a = f1_score(y_batch, pred_a, zero_division=0)
        f1_b = f1_score(y_batch, pred_b, zero_division=0)
        
        # Store for tracking
        formation_a.collective_f1 = f1_a
        formation_b.collective_f1 = f1_b
        
        # Weighted karma distribution (not equal!)
        if f1_a > f1_b:
            # Winner: Guru gets most, individual contribution matters
            formation_a.share_weighted_karma(True, f1_a, X_batch, y_batch)
            formation_b.share_weighted_karma(False, f1_b, X_batch, y_batch)
            formation_a.transfer_gyan()
        else:
            formation_b.share_weighted_karma(True, f1_b, X_batch, y_batch)
            formation_a.share_weighted_karma(False, f1_a, X_batch, y_batch)
            formation_b.transfer_gyan()
    
    def punarjanma_with_memory(self, X_train, y_train, feature_names=None):
        """
        🔄 Reincarnation with Memory
        Warriors return with knowledge of past lives
        """
        print("\n🔄 Punarjanma with Collective Memory...")
        
        # Sort by karma (collective + individual)
        all_warriors = [self.maharathi] + self.senapatis
        all_warriors.sort(key=lambda w: w.vijayas + self.kshetra_smriti['successful_varnas'].get(w.varna, 0), reverse=True)
        
        # Top 50% survive and mentor
        survivors = all_warriors[:len(all_warriors)//2]
        reborn = []
        
        for i, warrior in enumerate(survivors):
            # Create reincarnated version with memory
            new_warrior = Yoddha(
                yoddha_id=f"{warrior.yoddha_id}_reborn_{random.randint(1000,9999)}",
                shastra_type=warrior.shastra_type,
                varna=warrior.varna,
                beeja=warrior.beeja
            )
            
            # Transfer memory!
            new_warrior.shastras_heard = warrior.shastras_heard.copy()
            new_warrior.varna_vijayas = warrior.varna_vijayas.copy()
            
            # Train with inherited wisdom
            new_warrior.tapas(X_train, y_train, feature_names)
            reborn.append(new_warrior)
            
            print(f"   🕉️ {warrior.yoddha_id} → {new_warrior.yoddha_id} (memory preserved)")
        
        # Update population
        self.senapatis = reborn[1:] if len(reborn) > 0 else []
        self.maharathi = reborn[0] if len(reborn) > 0 else None
        
        # Recreate formations with new warriors
        self.create_formations()
    
    def run_collective_tapasya(self, X_train, y_train, X_val, y_val, feature_names=None):
        """Main training with collective warfare"""
        print("\n⚔️ COLLECTIVE TAPASYA BEGINS")
        print("="*70)
        
        # Initialize
        if not self.maharathi:
            self.maharathi = Yoddha("Maharathi_Arjun", varna='sarvabhaum')
            self.maharathi.tapas(X_train, y_train, feature_names)
        
        # Spawn senapatis
        self.senapatis = []
        for i in range(self.n_senapatis):
            yoddha = Yoddha(f"Senapati_{i}", varna=random.choice(VARNA_TYPES))
            yoddha.tapas(X_train, y_train, feature_names)
            self.senapatis.append(yoddha)
        
        # Create formations
        self.create_formations()
        
        # Run collective battles
        for round_num in range(self.yuddha_rounds):
            if round_num % 10 == 0:
                print(f"\n⚔️ Round {round_num}/{self.yuddha_rounds} - {len(self.formations)} formations battling")
            
            # Multiple group battles per round
            for _ in range(5):
                self.group_yuddha(X_val, y_val)
            
            # 🔄 ANTI-OVERFITTING: Shuffle formations every 15 rounds
            if (round_num + 1) % 15 == 0:
                self.shuffle_formations()
            
            # 📊 ANTI-OVERFITTING: Evaluate individual performance every 25 rounds
            if (round_num + 1) % 25 == 0:
                self.evaluate_individual_performance(X_val, y_val)
            
            # Reincarnation with memory every 20 rounds
            if (round_num + 1) % 20 == 0:
                self.punarjanma_with_memory(X_train, y_train, feature_names)
        
        print("\n✅ Collective Tapasya Complete!")
        return self
    
    def _spawn_single_senapati(self, idx: int, X_train, y_train, feature_names=None):
        """
        Spawn a single senapati (used by Cataclysm for replacement)
        """
        yoddha = Yoddha(f"Senapati_{idx}_reborn_{random.randint(1000,9999)}", 
                       varna=random.choice(VARNA_TYPES))
        yoddha.tapas(X_train, y_train, feature_names)
        
        if yoddha.is_tejasvi:
            return yoddha
        return None


# Export
__all__ = ['KurukshetraCollective', 'SenaFormation']
