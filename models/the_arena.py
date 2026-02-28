"""
🏟️ THE ARENA - Evolutionary Adversarial Ensemble
================================================
A revolutionary ensemble learning system where experts:
- Battle each other in combative prediction tournaments
- Share explainable stories to teach each other
- Evolve and mutate to discover new niches
- Learn through positive-sum competition (no punishment!)


Author: Sai Sandeep
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

# GPU/CUDA Setup
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

# Try cuML (GPU), fallback to sklearn (CPU)
try:
    from cuml.ensemble import RandomForestClassifier as cuRF
    from cuml.linear_model import LogisticRegression as cuLR
    from cuml.svm import SVC as cuSVC
    GPU_AVAILABLE = True
    print("✅ GPU acceleration enabled (cuML)")
except ImportError:
    from sklearn.ensemble import RandomForestClassifier as cuRF
    from sklearn.linear_model import LogisticRegression as cuLR
    from sklearn.svm import SVC as cuSVC
    GPU_AVAILABLE = False
    print("⚠️ CPU mode (cuML not available)")

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGBClassifier = None
    XGB_AVAILABLE = False

from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import random
from collections import deque
import pickle

# Arena Configuration
ARENA_CONFIG = {
    'batch_size': 1000,
    'combat_rounds': 50,
    'challengers_per_round': 5,
    'mutation_rate': 0.3,
    'niche_depth_threshold': 5,
    'exploration_bonus': 0.1,
    'story_memory_size': 1000,
    'gpu_batch_size': 10000,
    'min_wins_for_champion': 10
}

NICHE_TYPES = [
    'high_amount',      # Focus on high transaction amounts
    'time_based',       # Focus on unusual times (3 AM, etc.)
    'v14_extreme',      # Focus on V14 extreme values
    'v17_extreme',      # Focus on V17 extreme values
    'amount_v2_combo',  # Focus on Amount * V2 combinations
    'outlier_detector', # Focus on statistical outliers
    'velocity_based',   # Focus on rapid transactions
    'merchant_pattern', # Focus on merchant patterns
    'hybrid_combo',     # Focus on complex interactions
    'divine_alien'      # Special niche for Visvarupa avatars
]


@dataclass
class CombatStory:
    """An explainable story from a battle victory"""
    expert_id: str
    timestamp: datetime
    fraud_sample_idx: int
    features_that_mattered: List[str]
    feature_values: Dict[str, float]
    explanation: str
    confidence: float
    niche_focus: str
    pattern_signature: np.ndarray = field(repr=False)
    
    def to_dict(self) -> Dict:
        return {
            'expert_id': self.expert_id,
            'timestamp': self.timestamp.isoformat(),
            'fraud_sample_idx': self.fraud_sample_idx,
            'features_that_mattered': self.features_that_mattered,
            'feature_values': self.feature_values,
            'explanation': self.explanation,
            'confidence': self.confidence,
            'niche_focus': self.niche_focus
        }


class ArenaExpert:
    """
    🥊 Individual expert that fights in The Arena
    - Has a niche specialization
    - Can tell explainable stories
    - Can hear and adapt to stories
    - Can mutate and evolve
    """
    
    def __init__(self, 
                 expert_id: str,
                 model_type: str = 'rf',
                 niche: str = None,
                 seed: int = None,
                 generation: int = 0):
        self.expert_id = expert_id
        self.model_type = model_type
        self.niche = niche or random.choice(NICHE_TYPES)
        self.seed = seed or random.randint(0, 1000000)
        self.generation = generation
        
        # Combat statistics
        self.combat_wins = 0
        self.combat_losses = 0
        self.combat_draws = 0
        self.total_battles = 0
        self.f1_history = deque(maxlen=50)
        self.recent_wins = deque(maxlen=10)
        
        # Niche expertise
        self.niche_wins = {niche: 0 for niche in NICHE_TYPES}
        self.niche_f1_scores = {niche: [] for niche in NICHE_TYPES}
        self.niche_depth = 0  # How specialized in current niche
        
        # Story memory
        self.stories_heard = deque(maxlen=ARENA_CONFIG['story_memory_size'])
        self.stories_told = []
        
        # Mutation attributes
        self.mutation_rate = ARENA_CONFIG['mutation_rate']
        self.exploration_score = 0
        self.architecture_params = self._generate_random_params()
        
        # The actual model
        self.model = None
        self.is_fitted = False
        
        # Feature names (set during fit)
        self.feature_names = None
        self.feature_importance_history = deque(maxlen=100)
        
    def _generate_random_params(self) -> Dict:
        """Generate random architecture parameters for mutation"""
        params = {
            'rf': {
                'n_estimators': random.randint(50, 300),
                'max_depth': random.randint(3, 15),
                'min_samples_split': random.randint(2, 20),
                'max_features': random.choice(['sqrt', 'log2', None])
            },
            'xgb': {
                'n_estimators': random.randint(50, 300),
                'max_depth': random.randint(3, 10),
                'learning_rate': random.uniform(0.01, 0.3),
                'subsample': random.uniform(0.6, 1.0),
                'colsample_bytree': random.uniform(0.6, 1.0)
            },
            'lr': {
                'C': random.uniform(0.1, 10.0),
                'penalty': random.choice(['l1', 'l2']),
                'max_iter': 5000  # Increased from 1000 to fix convergence warnings
            },
            'svm': {
                'C': random.uniform(0.1, 10.0),
                'kernel': random.choice(['rbf', 'poly', 'sigmoid']),
                'gamma': random.choice(['scale', 'auto'])
            }
        }
        return params.get(self.model_type, params['rf'])
    
    def _get_model(self, n_features: int):
        """Create the actual ML model with current params"""
        params = self.architecture_params.copy()
        
        if self.model_type == 'rf':
            return cuRF(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                min_samples_split=params['min_samples_split'],
                max_features=params['max_features'],
                random_state=self.seed
            )
        elif self.model_type == 'xgb':
            return XGBClassifier(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                learning_rate=params['learning_rate'],
                subsample=params['subsample'],
                colsample_bytree=params['colsample_bytree'],
                random_state=self.seed,
                use_label_encoder=False,
                eval_metric='logloss',
                tree_method='hist',
                device='cuda'
            )
        elif self.model_type == 'lr':
            return cuLR(
                C=params['C'],
                penalty=params['penalty'],
                max_iter=params['max_iter']
            )
        elif self.model_type == 'svm':
            return cuSVC(
                C=params['C'],
                kernel=params['kernel'],
                gamma=params['gamma'],
                probability=True
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            sample_weight: Optional[np.ndarray] = None,
            feature_names: List[str] = None):
        """Train the expert on data"""
        self.feature_names = feature_names or [f'V{i}' for i in range(X.shape[1])]
        
        # Ensure labels are integers (not continuous/float)
        y = np.asarray(y, dtype=np.int32)
        
        # Apply niche focus (sample weighting based on niche)
        if sample_weight is None and self.niche != 'hybrid_combo':
            sample_weight = self._apply_niche_weighting(X, y)
        
        # Create and fit model
        self.model = self._get_model(X.shape[1])
        
        try:
            if sample_weight is not None:
                try:
                    self.model.fit(X, y, sample_weight=sample_weight)
                except TypeError:
                    # cuML doesn't support sample_weight, fit without it
                    self.model.fit(X, y)
            else:
                self.model.fit(X, y)
            self.is_fitted = True
        except Exception as e:
            print(f"  ⚠️ {self.expert_id} failed to fit: {e}")
            self.is_fitted = False
    
    def _apply_niche_weighting(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Apply sample weights based on niche specialization"""
        weights = np.ones(len(y))
        fraud_mask = y == 1
        
        if self.niche == 'high_amount':
            # Weight high amount transactions more
            amount_idx = -3  # Assuming Amount is 3rd from last
            amounts = X[:, amount_idx]
            high_amount_mask = amounts > np.percentile(amounts, 90)
            weights[high_amount_mask & fraud_mask] *= 3.0
            
        elif self.niche == 'time_based':
            # Weight unusual times (3 AM, etc.)
            time_idx = -1  # Assuming Time is last feature
            times = X[:, time_idx]
            unusual_time_mask = (times % 86400 < 14400) | (times % 86400 > 79200)  # 3 AM or late night
            weights[unusual_time_mask & fraud_mask] *= 3.0
            
        elif self.niche in ['v14_extreme', 'v17_extreme']:
            # Weight extreme V14/V17 values
            v_idx = 14 if self.niche == 'v14_extreme' else 17
            v_values = X[:, v_idx]
            extreme_mask = np.abs(v_values) > 2.0
            weights[extreme_mask & fraud_mask] *= 3.0
            
        elif self.niche == 'outlier_detector':
            # Weight statistical outliers
            z_scores = np.abs((X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8))
            outlier_mask = (z_scores > 2.5).any(axis=1)
            weights[outlier_mask & fraud_mask] *= 2.5
        
        return weights
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            return np.zeros(len(X))
        
        # 🛠️ Convert GPU arrays to numpy first
        if hasattr(X, 'get'):
            X = X.get()
        else:
            X = np.asarray(X)
        
        try:
            # 🛠️ Convert GPU arrays to numpy first
            if hasattr(X, 'get'):
                X = X.get()
            else:
                X = np.asarray(X)
            
            # Check if model is sklearn-based (Visvarupa avatars, etc.)
            # sklearn models CANNOT run on GPU - use CPU only!
            model_type = type(self.model).__name__
            is_sklearn = model_type in ['IsolationForest', 'OneClassSVM', 'LocalOutlierFactor', 
                                        'EllipticEnvelope', 'DBSCAN', 'RandomForestClassifier',
                                        'GradientBoostingClassifier', 'ExtraTreesClassifier',
                                        'DecisionTreeClassifier', 'LogisticRegression']
            
            if is_sklearn:
                # Always use CPU for sklearn models
                preds = self.model.predict(X)
                # Convert -1 (outlier) to 1 (fraud), 1 (inlier) to 0 (normal)
                if model_type in ['IsolationForest', 'LocalOutlierFactor', 'OneClassSVM', 'EllipticEnvelope']:
                    preds = (preds == -1).astype(int)
                return np.asarray(preds)
            
            # Try GPU first for GPU-compatible models, fallback to CPU
            if CUPY_AVAILABLE:
                try:
                    X_gpu = cp.asarray(X)
                    preds = self.model.predict(X_gpu)
                    return cp.asnumpy(preds)
                except Exception as e:
                    print(f"  ⚠️ GPU failed for {self.expert_id}, using CPU: {e}")
                    preds = self.model.predict(X)
                    return np.asarray(preds)
            else:
                preds = self.model.predict(X)
                return np.asarray(preds)
        except Exception as e:
            print(f"  ⚠️ {self.expert_id} prediction failed: {e}")
            return np.zeros(len(X))
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        if not self.is_fitted:
            return np.zeros((len(X), 2))
        
        # 🛠️ Convert GPU arrays to numpy first
        if hasattr(X, 'get'):
            X = X.get()
        else:
            X = np.asarray(X)
        
        try:
            # Try GPU first, fallback to CPU
            if CUPY_AVAILABLE:
                try:
                    X_gpu = cp.asarray(X)
                    if hasattr(self.model, 'predict_proba'):
                        probs = self.model.predict_proba(X_gpu)
                    else:
                        # Fallback for models without predict_proba
                        preds = self.model.predict(X_gpu)
                        probs = np.zeros((len(preds), 2))
                        probs[:, 0] = 1 - preds
                        probs[:, 1] = preds
                        return probs
                    return cp.asnumpy(probs)
                except RuntimeError as e:
                    if "libnvrtc" in str(e):
                        print(f"  ⚠️ CUDA runtime missing, using CPU for {self.expert_id}")
                        if hasattr(self.model, 'predict_proba'):
                            probs = self.model.predict_proba(X)
                        else:
                            preds = self.model.predict(X)
                            probs = np.zeros((len(preds), 2))
                            probs[:, 0] = 1 - preds
                            probs[:, 1] = preds
                        return np.asarray(probs)
                    raise
            else:
                if hasattr(self.model, 'predict_proba'):
                    probs = self.model.predict_proba(X)
                else:
                    preds = self.model.predict(X)
                    probs = np.zeros((len(preds), 2))
                    probs[:, 0] = 1 - preds
                    probs[:, 1] = preds
                return np.asarray(probs)
        except Exception as e:
            print(f"  ⚠️ {self.expert_id} predict_proba failed: {e}")
            return np.zeros((len(X), 2))
    
    def tell_story(self, X: np.ndarray, y: np.ndarray, 
                   fraud_indices: List[int]) -> List[CombatStory]:
        """
        Generate explainable stories for frauds this expert caught
        """
        stories = []
        
        if not self.is_fitted or len(fraud_indices) == 0:
            return stories
        
        # Skip story generation for divine alien avatars (isolation-based models)
        if self.niche == 'divine_alien':
            return stories
        
        try:
            # Ensure feature_names exists
            if self.feature_names is None:
                self.feature_names = [f'V{i}' for i in range(X.shape[1])]
            
            # Get feature importances if available
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
            else:
                importances = np.ones(len(self.feature_names)) / len(self.feature_names)
            
            # Get top features
            top_feature_indices = np.argsort(importances)[-5:][::-1]
            top_features = [self.feature_names[i] for i in top_feature_indices]
            
            for idx in fraud_indices[:5]:  # Max 5 stories per battle
                feature_values = {feat: float(X[idx, i]) 
                                for i, feat in enumerate(self.feature_names)}
                
                # Create explanation based on top features
                explanations = []
                for feat in top_features[:3]:
                    val = feature_values[feat]
                    feat_idx = self.feature_names.index(feat)
                    if abs(val) > 2.0:
                        explanations.append(f"{feat} extreme ({val:.2f})")
                    elif val > np.percentile(X[:, feat_idx], 95):
                        explanations.append(f"{feat} very high ({val:.2f})")
                    elif val < np.percentile(X[:, feat_idx], 5):
                        explanations.append(f"{feat} very low ({val:.2f})")
                
                # Get confidence from probability
                proba = self.predict_proba(X[idx:idx+1])
                if proba is not None and len(proba) > 0:
                    confidence = proba[0][1] if y[idx] == 1 else proba[0][0]
                else:
                    confidence = 0.5
                
                story = CombatStory(
                    expert_id=self.expert_id,
                    timestamp=datetime.now(),
                    fraud_sample_idx=idx,
                    features_that_mattered=top_features[:3],
                    feature_values={k: feature_values[k] for k in top_features[:3]},
                    explanation="; ".join(explanations) if explanations else f"Complex pattern in {self.niche}",
                    confidence=float(confidence),
                    niche_focus=self.niche,
                    pattern_signature=X[idx].copy()
                )
                
                stories.append(story)
                self.stories_told.append(story)
        
        except Exception as e:
            print(f"  ⚠️ {self.expert_id} story telling failed: {e}")
        
        return stories
    
    def hear_story(self, story: CombatStory):
        """
        Listen to a story and adapt strategy
        """
        self.stories_heard.append(story)
        
        # Increase weight for features mentioned in story
        if story.features_that_mattered:
            for feat in story.features_that_mattered:
                if feat in self.feature_importance_history:
                    # This feature is important - note it
                    pass
        
        # If story is from different niche, consider exploring it
        if story.niche_focus != self.niche:
            self.exploration_score += 0.05
            
            # If we keep hearing about different niche being successful,
            # maybe we should try it!
            niche_stories = [s for s in self.stories_heard if s.niche_focus == story.niche_focus]
            if len(niche_stories) > 5:
                avg_confidence = np.mean([s.confidence for s in niche_stories])
                if avg_confidence > 0.85 and random.random() < 0.1:
                    # 10% chance to explore new niche
                    print(f"    🔄 {self.expert_id} considering niche switch from {self.niche} to {story.niche_focus}")
    
    def spawn_child(self) -> 'ArenaExpert':
        """
        Spawn a mutated child expert
        """
        child_id = f"{self.expert_id}_child_{self.generation + 1}_{random.randint(1000, 9999)}"
        
        # Mutate niche (small chance)
        child_niche = self.niche
        if random.random() < 0.3:  # 30% chance to mutate niche
            child_niche = random.choice([n for n in NICHE_TYPES if n != self.niche])
        
        # Mutate model type (rare)
        child_model_type = self.model_type
        if random.random() < 0.1:  # 10% chance
            child_model_type = random.choice(['rf', 'xgb', 'lr', 'svm'])
        
        # Mutate seed
        child_seed = self.seed + random.randint(-1000, 1000)
        
        # Create child
        child = ArenaExpert(
            expert_id=child_id,
            model_type=child_model_type,
            niche=child_niche,
            seed=child_seed,
            generation=self.generation + 1
        )
        
        # If model type changed, generate fresh params for that model type
        # Otherwise mutate parent's params
        if child_model_type != self.model_type:
            child.architecture_params = child._generate_random_params()
        else:
            child.architecture_params = self._mutate_params(self.architecture_params)
        
        child.mutation_rate = self.mutation_rate * random.uniform(0.9, 1.1)
        
        # Inherit some combat wisdom
        child.stories_heard = deque(list(self.stories_heard)[-50:], maxlen=ARENA_CONFIG['story_memory_size'])
        
        return child
    
    def _mutate_params(self, params: Dict) -> Dict:
        """Mutate architecture parameters"""
        mutated = params.copy()
        
        for key in mutated:
            if isinstance(mutated[key], (int, np.integer)):
                if random.random() < 0.3:
                    # min_samples_split must be >= 2 for cuML
                    min_val = 2 if key == 'min_samples_split' else 1
                    mutated[key] = max(min_val, int(mutated[key] * random.uniform(0.8, 1.2)))
            elif isinstance(mutated[key], float):
                if random.random() < 0.3:
                    new_val = mutated[key] * random.uniform(0.8, 1.2)
                    # Clamp subsample and colsample to [0, 1] for XGBoost
                    if key in ['subsample', 'colsample_bytree', 'colsample_bylevel']:
                        new_val = max(0.1, min(1.0, new_val))
                    mutated[key] = new_val
            elif isinstance(mutated[key], str):
                if random.random() < 0.1:
                    if key == 'kernel':
                        mutated[key] = random.choice(['rbf', 'poly', 'sigmoid'])
                    elif key == 'penalty':
                        mutated[key] = random.choice(['l1', 'l2'])
                    elif key == 'max_features':
                        mutated[key] = random.choice(['sqrt', 'log2', None])
        
        return mutated
    
    def mutate(self, aggressive: bool = False):
        """
        Mutate this expert's parameters and reinitialize model
        Used by Cataclysm to create aggressive variants
        """
        # Mutate architecture params
        mutation_rate = 0.5 if aggressive else 0.3
        
        for key in self.architecture_params:
            if random.random() < mutation_rate:
                val = self.architecture_params[key]
                if isinstance(val, (int, np.integer)):
                    self.architecture_params[key] = max(1, int(val * random.uniform(0.7, 1.3)))
                elif isinstance(val, float):
                    new_val = val * random.uniform(0.7, 1.3)
                    if key in ['subsample', 'colsample_bytree', 'colsample_bylevel']:
                        new_val = max(0.1, min(1.0, new_val))
                    self.architecture_params[key] = new_val
                elif isinstance(val, str):
                    if key == 'kernel':
                        self.architecture_params[key] = random.choice(['rbf', 'poly', 'sigmoid'])
                    elif key == 'penalty':
                        self.architecture_params[key] = random.choice(['l1', 'l2'])
        
        # Maybe change niche
        if aggressive and random.random() < 0.3:
            self.niche = random.choice(NICHE_TYPES)
        
        # Reset some stats but keep experience
        self.exploration_score += 0.1 if aggressive else 0.05
    
    def get_combat_score(self) -> float:
        """Calculate overall combat score"""
        if self.total_battles == 0:
            return 0.0
        
        base_score = self.combat_wins / max(self.total_battles, 1)
        
        # Bonus for recent performance
        if self.recent_wins:
            recent_win_rate = sum(self.recent_wins) / len(self.recent_wins)
            base_score += recent_win_rate * 0.1
        
        # Bonus for niche diversity (exploration)
        unique_niches = sum(1 for v in self.niche_wins.values() if v > 0)
        base_score += unique_niches * 0.02
        
        # Exploration bonus
        base_score += self.exploration_score * ARENA_CONFIG['exploration_bonus']
        
        return min(base_score, 1.0)
    
    def is_dominant_in_niche(self, niche: str) -> bool:
        """Check if this expert dominates a particular niche"""
        return self.niche == niche and self.niche_wins[niche] > ARENA_CONFIG['niche_depth_threshold']
    
    def __repr__(self):
        score = self.get_combat_score()
        return f"ArenaExpert({self.expert_id}, {self.model_type}, {self.niche}, wins={self.combat_wins}, score={score:.3f})"


class TheArena:
    """
    🏟️ The Arena - Where experts battle for supremacy!
    
    The main orchestrator that:
    - Manages champion and challengers
    - Runs combat tournaments
    - Handles story sharing
    - Evolves the population
    - Returns the ultimate ensemble
    """
    
    def __init__(self, 
                 n_challengers: int = 20,
                 combat_rounds: int = 50,
                 batch_size: int = 1000):
        self.n_challengers = n_challengers
        self.combat_rounds = combat_rounds
        self.batch_size = batch_size
        
        # The champion - starts as current best model
        self.champion: Optional[ArenaExpert] = None
        
        # Pool of challengers
        self.challengers: List[ArenaExpert] = []
        
        # Former champions (elders - still valuable!)
        self.elders: List[ArenaExpert] = []
        
        # All stories shared in arena
        self.arena_stories: deque = deque(maxlen=ARENA_CONFIG['story_memory_size'])
        
        # Combat history
        self.combat_history: List[Dict] = []
        
        # Best ensemble combination found
        self.best_ensemble: Optional[List[ArenaExpert]] = None
        self.best_f1 = 0.0
        
        print("🏟️ THE ARENA INITIALIZED")
        print(f"   Combat Rounds: {combat_rounds}")
        print(f"   Challengers: {n_challengers}")
        print(f"   Batch Size: {batch_size}")
        print("   Ready for battle! ⚔️\n")
    
    def initialize_champion(self, X_train: np.ndarray, y_train: np.ndarray,
                           feature_names: List[str] = None,
                           existing_model: Optional = None):
        """
        Initialize the champion with current best model
        """
        print("👑 CROWNING THE CHAMPION...")
        
        if existing_model is not None:
            # Use provided model as champion
            self.champion = ArenaExpert(
                expert_id='Champion_Ultimate',
                model_type='xgb',
                niche='hybrid_combo',
                seed=42
            )
            self.champion.model = existing_model
            self.champion.is_fitted = True
            self.champion.feature_names = feature_names
        else:
            # Create new strong champion
            self.champion = ArenaExpert(
                expert_id='Champion_Alpha',
                model_type='xgb',
                niche='hybrid_combo',
                seed=42
            )
            self.champion.fit(X_train, y_train, feature_names=feature_names)
        
        # Champion starts with some wins
        self.champion.combat_wins = ARENA_CONFIG['min_wins_for_champion']
        self.champion.total_battles = ARENA_CONFIG['min_wins_for_champion']
        
        print(f"   Champion: {self.champion.expert_id}")
        print(f"   Model: {self.champion.model_type}")
        print(f"   Niche: {self.champion.niche}")
        print(f"   Status: Ready to defend title! 🛡️\n")
    
    def spawn_challengers(self, X_train: np.ndarray, y_train: np.ndarray,
                         feature_names: List[str] = None):
        """
        Spawn initial pool of challengers
        """
        print(f"🌱 SPAWNING {self.n_challengers} CHALLENGERS...")
        
        model_types = ['rf', 'xgb', 'lr', 'svm']
        
        for i in range(self.n_challengers):
            # Random configuration
            model_type = random.choice(model_types)
            niche = random.choice(NICHE_TYPES)
            seed = random.randint(0, 1000000)
            
            challenger = ArenaExpert(
                expert_id=f"Challenger_{i:03d}",
                model_type=model_type,
                niche=niche,
                seed=seed,
                generation=0
            )
            
            # Train on data with niche weighting
            challenger.fit(X_train, y_train, feature_names=feature_names)
            
            if challenger.is_fitted:
                self.challengers.append(challenger)
                print(f"   ✓ {challenger.expert_id} ({model_type}, {niche}) ready!")
            else:
                print(f"   ✗ Challenger {i} failed to spawn")
        
        print(f"   {len(self.challengers)} challengers ready for battle!\n")
    
    def _spawn_single_challenger(self, idx: int, X_train: np.ndarray, y_train: np.ndarray,
                                  feature_names: List[str] = None) -> Optional['ArenaExpert']:
        """
        Spawn a single challenger (used by Cataclysm for replacement)
        """
        model_types = ['rf', 'xgb', 'lr', 'svm']
        
        model_type = random.choice(model_types)
        niche = random.choice(NICHE_TYPES)
        seed = random.randint(0, 1000000)
        
        challenger = ArenaExpert(
            expert_id=f"Challenger_{idx:03d}_reborn_{random.randint(1000,9999)}",
            model_type=model_type,
            niche=niche,
            seed=seed,
            generation=0
        )
        
        challenger.fit(X_train, y_train, feature_names=feature_names)
        
        if challenger.is_fitted:
            return challenger
        return None
    
    def run_battle(self, challenger: ArenaExpert, 
                   X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """
        Run a single battle between champion and challenger
        """
        # Sample a batch
        batch_size = min(self.batch_size, len(X_val))
        indices = np.random.choice(len(X_val), batch_size, replace=False)
        X_batch = X_val[indices]
        y_batch = y_val[indices]
        
        # 🛠️ Convert to numpy if GPU arrays (for Visvarupa avatars with sklearn models)
        if hasattr(X_batch, 'get'):
            X_batch = X_batch.get()
        else:
            X_batch = np.asarray(X_batch)
        
        # Get predictions
        champion_preds = self.champion.predict(X_batch)
        challenger_preds = challenger.predict(X_batch)
        
        # Calculate F1 scores
        champion_f1 = f1_score(y_batch, champion_preds, zero_division=0)
        challenger_f1 = f1_score(y_batch, challenger_preds, zero_division=0)
        
        # Determine winner
        if challenger_f1 > champion_f1:
            winner = challenger
            loser = self.champion
            champion_won = False
            margin = challenger_f1 - champion_f1
        elif champion_f1 > challenger_f1:
            winner = self.champion
            loser = challenger
            champion_won = True
            margin = champion_f1 - challenger_f1
        else:
            # Draw
            winner = None
            champion_won = None
            margin = 0
        
        # Update combat stats
        challenger.total_battles += 1
        self.champion.total_battles += 1
        
        if winner is None:
            # Draw
            challenger.combat_draws += 1
            self.champion.combat_draws += 1
            challenger.recent_wins.append(0.5)
            self.champion.recent_wins.append(0.5)
        elif winner == challenger:
            # Challenger wins!
            challenger.combat_wins += 1
            challenger.recent_wins.append(1)
            self.champion.recent_wins.append(0)
            challenger.niche_wins[challenger.niche] += 1
        else:
            # Champion wins!
            self.champion.combat_wins += 1
            self.champion.recent_wins.append(1)
            challenger.recent_wins.append(0)
            self.champion.niche_wins[self.champion.niche] += 1
        
        # Store F1 history
        challenger.f1_history.append(challenger_f1)
        self.champion.f1_history.append(champion_f1)
        
        # Generate stories from victories
        stories = []
        if winner is not None:
            # Get indices of correctly detected frauds
            fraud_mask = y_batch == 1
            if winner == challenger:
                correct_frauds = np.where((challenger_preds == 1) & fraud_mask)[0]
            else:
                correct_frauds = np.where((champion_preds == 1) & fraud_mask)[0]
            
            if len(correct_frauds) > 0:
                # Map back to original indices
                original_indices = indices[correct_frauds]
                stories = winner.tell_story(X_val, y_val, original_indices.tolist())
                
                # Share stories with arena
                for story in stories:
                    self.arena_stories.append(story)
                    # Loser hears the story
                    loser.hear_story(story)
        
        return {
            'challenger': challenger.expert_id,
            'champion_f1': champion_f1,
            'challenger_f1': challenger_f1,
            'winner': winner.expert_id if winner else 'Draw',
            'champion_won': champion_won,
            'margin': margin,
            'stories_shared': len(stories),
            'challenger_niche': challenger.niche
        }
    
    def dethrone_champion(self, new_champion: ArenaExpert):
        """
        Replace the champion with a new one
        Old champion becomes an elder
        """
        print(f"\n🏆 CHAMPION DETHRONED!")
        print(f"   Old Champion: {self.champion.expert_id} (becomes Elder)")
        print(f"   New Champion: {new_champion.expert_id}")
        print(f"   Combat Wins: {new_champion.combat_wins}")
        print(f"   Victories in niche '{new_champion.niche}': {new_champion.niche_wins[new_champion.niche]}")
        
        # Move old champion to elders
        self.elders.append(self.champion)
        
        # Crown new champion
        self.champion = new_champion
        
        # Remove from challengers pool
        if new_champion in self.challengers:
            self.challengers.remove(new_champion)
        
        print(f"   👑 Long live the new champion!\n")
    
    def evolve_population(self, X_train: np.ndarray, y_train: np.ndarray,
                         feature_names: List[str] = None):
        """
        Evolve the challenger population
        """
        print("🧬 EVOLVING POPULATION...")
        
        # Select top performers to spawn children
        if len(self.challengers) == 0:
            print("   No challengers to evolve!")
            return
        
        # Sort by combat score
        sorted_challengers = sorted(
            self.challengers, 
            key=lambda x: x.get_combat_score(), 
            reverse=True
        )
        
        # Top 30% spawn children
        n_parents = max(1, len(sorted_challengers) // 3)
        parents = sorted_challengers[:n_parents]
        
        new_children = []
        for parent in parents:
            # Spawn 2 children per successful parent
            for _ in range(2):
                child = parent.spawn_child()
                child.fit(X_train, y_train, feature_names=feature_names)
                
                if child.is_fitted:
                    new_children.append(child)
                    print(f"   🌱 {child.expert_id} spawned from {parent.expert_id} ({child.niche})")
        
        # Remove worst performers to maintain population size
        n_to_remove = len(new_children)
        if n_to_remove > 0 and len(self.challengers) > n_to_remove:
            # Remove lowest combat score
            worst = sorted_challengers[-n_to_remove:]
            for w in worst:
                if w in self.challengers:
                    self.challengers.remove(w)
                    print(f"   💀 {w.expert_id} retired (low score: {w.get_combat_score():.3f})")
        
        # Add new children
        self.challengers.extend(new_children)
        
        print(f"   Population: {len(self.challengers)} challengers")
        print(f"   Elders: {len(self.elders)}")
        print(f"   Evolution complete!\n")
    
    def run_tournament(self, X_val: np.ndarray, y_val: np.ndarray):
        """
        Run a full tournament round
        """
        print(f"⚔️ TOURNAMENT ROUND STARTING...")
        print(f"   Battles: {len(self.challengers)}")
        
        round_results = []
        
        for i, challenger in enumerate(self.challengers):
            # Battle champion
            result = self.run_battle(challenger, X_val, y_val)
            round_results.append(result)
            
            # Check if challenger dethrones champion
            if result['winner'] == challenger.expert_id:
                if challenger.combat_wins >= ARENA_CONFIG['min_wins_for_champion']:
                    # Must have minimum wins to challenge for title
                    if result['challenger_f1'] > result['champion_f1'] + 0.01:
                        # Significant margin required
                        self.dethrone_champion(challenger)
            
            if (i + 1) % 5 == 0:
                print(f"   Progress: {i+1}/{len(self.challengers)} battles complete")
        
        # Store combat history
        self.combat_history.extend(round_results)
        
        # Summary
        wins = sum(1 for r in round_results if r['champion_won'] == True)
        losses = sum(1 for r in round_results if r['champion_won'] == False)
        draws = sum(1 for r in round_results if r['champion_won'] is None)
        
        print(f"   Results: Champion {wins}W-{losses}L-{draws}D")
        print(f"   Tournament complete!\n")
        
        return round_results
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              feature_names: List[str] = None,
              existing_model: Optional = None) -> 'TheArena':
        """
        Full training cycle in The Arena
        """
        print("="*70)
        print("🏟️ THE ARENA - EVOLUTIONARY ADVERSARIAL ENSEMBLE 🏟️")
        print("="*70)
        print()
        
        # Initialize
        self.initialize_champion(X_train, y_train, feature_names, existing_model)
        self.spawn_challengers(X_train, y_train, feature_names)
        
        # Run combat rounds
        for round_num in range(self.combat_rounds):
            print(f"\n{'='*70}")
            print(f"🥊 COMBAT ROUND {round_num + 1}/{self.combat_rounds}")
            print(f"{'='*70}")
            
            # Run tournament
            self.run_tournament(X_val, y_val)
            
            # Evolve every 5 rounds
            if (round_num + 1) % 5 == 0:
                self.evolve_population(X_train, y_train, feature_names)
            
            # Evaluate ensemble and update best F1 every 10 rounds
            if (round_num + 1) % 10 == 0:
                self._update_best_f1(X_val, y_val)
                self._print_status()
        
        print("\n" + "="*70)
        print("🏟️ ARENA COMBAT COMPLETE!")
        print("="*70)
        
        # Build final ensemble from top performers
        self._build_final_ensemble(X_val, y_val)
        
        return self
    
    def _print_status(self):
        """Print current arena status"""
        print(f"\n📊 ARENA STATUS:")
        print(f"   Champion: {self.champion.expert_id} (wins: {self.champion.combat_wins})")
        print(f"   Top Challenger: {max(self.challengers, key=lambda x: x.get_combat_score()).expert_id}")
        print(f"   Elders: {len(self.elders)}")
        print(f"   Total Stories Shared: {len(self.arena_stories)}")
        print(f"   Best F1 Achieved: {self.best_f1:.4f}")
    
    def _update_best_f1(self, X_val: np.ndarray, y_val: np.ndarray):
        """Evaluate current ensemble and update best F1"""
        try:
            # 🛠️ Convert GPU arrays to numpy for sklearn models
            if hasattr(X_val, 'get'):
                X_val = X_val.get()
            else:
                X_val = np.asarray(X_val)
            
            # Quick evaluation with current top experts
            all_experts = [self.champion] + self.elders + self.challengers
            all_experts.sort(key=lambda x: x.get_combat_score(), reverse=True)
            top_experts = all_experts[:7]
            
            # Get predictions from top experts
            predictions = []
            for expert in top_experts:
                if expert.is_fitted:
                    try:
                        pred = expert.predict(X_val)
                        if CUPY_AVAILABLE and hasattr(pred, 'get'):
                            pred = pred.get()
                        predictions.append(pred)
                    except:
                        pass
            
            if len(predictions) > 0:
                # Majority vote
                ensemble_preds = np.round(np.mean(predictions, axis=0)).astype(int)
                current_f1 = f1_score(y_val, ensemble_preds)
                
                # Update best F1
                if current_f1 > self.best_f1:
                    self.best_f1 = current_f1
                    
        except Exception as e:
            # Silently ignore evaluation errors during training
            pass
    
    def _build_final_ensemble(self, X_val: np.ndarray, y_val: np.ndarray):
        """
        Build the final ensemble from top performers
        """
        print("\n🏗️ BUILDING FINAL ENSEMBLE...")
        
        # Collect all experts
        all_experts = [self.champion] + self.elders + self.challengers
        
        # Sort by combat score
        all_experts.sort(key=lambda x: x.get_combat_score(), reverse=True)
        
        # Select top 7 experts for final ensemble
        self.best_ensemble = all_experts[:7]
        
        # Evaluate ensemble
        ensemble_preds = self._ensemble_predict(X_val, use_best=True)
        self.best_f1 = f1_score(y_val, ensemble_preds)
        
        print(f"   Selected {len(self.best_ensemble)} experts:")
        for i, expert in enumerate(self.best_ensemble, 1):
            print(f"   {i}. {expert.expert_id} ({expert.model_type}, {expert.niche}) - Score: {expert.get_combat_score():.3f}")
        
        print(f"\n   🎯 Final Ensemble F1: {self.best_f1:.4f}")
    
    def _ensemble_predict(self, X: np.ndarray, 
                         use_best: bool = True) -> np.ndarray:
        """
        Make ensemble prediction
        """
        # 🛠️ Convert GPU arrays to numpy for sklearn models
        if hasattr(X, 'get'):
            X = X.get()
        else:
            X = np.asarray(X)
        
        if use_best and self.best_ensemble:
            experts = self.best_ensemble
        else:
            experts = [self.champion] + self.challengers[:5]
        
        # Get weighted votes
        weights = [e.get_combat_score() for e in experts]
        total_weight = sum(weights)
        
        if total_weight == 0:
            weights = [1.0] * len(experts)
            total_weight = len(experts)
        
        # Normalize weights
        weights = [w / total_weight for w in weights]
        
        # Aggregate predictions
        votes = np.zeros(len(X))
        for expert, weight in zip(experts, weights):
            preds = expert.predict(X)
            votes += preds * weight
        
        # Threshold at 0.5
        return (votes >= 0.5).astype(int)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the trained ensemble"""
        return self._ensemble_predict(X, use_best=True)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions"""
        # 🛠️ Convert GPU arrays to numpy for sklearn models
        if hasattr(X, 'get'):
            X = X.get()
        else:
            X = np.asarray(X)
        
        if not self.best_ensemble:
            return np.zeros((len(X), 2))
        
        weights = [e.get_combat_score() for e in self.best_ensemble]
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Average probabilities
        probs = np.zeros((len(X), 2))
        for expert, weight in zip(self.best_ensemble, weights):
            expert_probs = expert.predict_proba(X)
            probs += expert_probs * weight
        
        return probs
    
    def save(self, filepath: str):
        """Save the arena state"""
        state = {
            'champion': self.champion,
            'challengers': self.challengers,
            'elders': self.elders,
            'best_ensemble': self.best_ensemble,
            'best_f1': self.best_f1,
            'combat_history': self.combat_history,
            'config': ARENA_CONFIG
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        print(f"💾 Arena saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'TheArena':
        """Load a saved arena"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        arena = cls(
            n_challengers=len(state['challengers']),
            combat_rounds=state['config']['combat_rounds'],
            batch_size=state['config']['batch_size']
        )
        
        arena.champion = state['champion']
        arena.challengers = state['challengers']
        arena.elders = state['elders']
        arena.best_ensemble = state['best_ensemble']
        arena.best_f1 = state['best_f1']
        arena.combat_history = state['combat_history']
        
        print(f"💾 Arena loaded from {filepath}")
        return arena


def run_arena_experiment(X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray,
                        feature_names: List[str] = None,
                        existing_model: Optional = None,
                        save_path: str = None) -> Tuple[TheArena, float]:
    """
    Run a full Arena experiment and return the trained arena + best F1
    """
    # Create arena
    arena = TheArena(
        n_challengers=20,
        combat_rounds=50,
        batch_size=1000
    )
    
    # Train
    arena.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        feature_names=feature_names,
        existing_model=existing_model
    )
    
    # Save if path provided
    if save_path:
        arena.save(save_path)
    
    return arena, arena.best_f1


if __name__ == "__main__":
    print("🏟️ THE ARENA - Evolutionary Adversarial Ensemble")
    print("   Run with your data using run_arena_experiment()")
    print("   GPU acceleration: ENABLED")
    print("   Story sharing: ENABLED")
    print("   Evolution: ENABLED")
    print("   Ready for battle! ⚔️")
