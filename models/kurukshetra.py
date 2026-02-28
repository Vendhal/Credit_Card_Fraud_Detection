"""
⚔️ KURUKSHETRA - धर्मक्षेत्र (Field of Righteousness)
=====================================================
A revolutionary ensemble learning system inspired by the Mahabharata:
- Warriors (Yoddhas) battle in the sacred field of Kurukshetra
- Share wisdom through Shastrartha (spiritual discourse)
- Reincarnate and evolve through Punarjanma
- Fight for Dharma (detecting fraud/truth)

Inspired by: Mahabharata War + Sanatana Dharma + Evolutionary Algorithms

Author: Dharma Yoddha
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

# Kurukshetra Configuration
DHARMA_CONFIG = {
    'senas_batch': 1000,           # Batch size (was batch_size)
    'yuddha_rounds': 50,           # Combat rounds (was combat_rounds)
    'senapatis_per_yuddha': 5,     # Challengers per round
    'tapasya_rate': 0.3,           # Mutation rate (was mutation_rate)
    'varna_depth_threshold': 5,    # Niche depth threshold
    'gyan_bonus': 0.1,             # Exploration bonus (was exploration_bonus)
    'shastra_memory_size': 1000,   # Story memory size
    'gpu_batch_size': 10000,
    'min_vijayas_for_maharathi': 10  # Min wins for champion (was min_wins_for_champion)
}

# Varnas (Niches/Specializations) - The 9 forms of warfare
VARNA_TYPES = [
    'dhanurveda',      # High amount (bow warfare)
    'kaal_chakra',     # Time-based (wheel of time)
    'v14_tejas',       # V14 extreme (radiance)
    'v17_shakti',      # V17 extreme (power)
    'arth_v2_yoga',    # Amount * V2 combo (union)
    'anomaly_drishti', # Outlier detection (divine sight)
    'vega_shastra',    # Velocity-based (speed science)
    'vyapar_maya',     # Merchant patterns (illusion of trade)
    'sarvabhaum'       # Hybrid combo (universal emperor)
]

# Warrior titles based on victories
YODDHA_TITLES = {
    0: 'Dharmic_Pratishtha',      # Righteous Establishment
    10: 'Yoddha',                  # Warrior
    25: 'Veera',                   # Hero
    50: 'Mahaveera',               # Great Hero
    100: 'Maharathi',              # Great Charioteer Warrior
    200: 'Atirathi',               # Superior Warrior
    500: 'Rathi',                  # Charioteer
    1000: 'Senapati',              # Commander
    2000: 'Mahanayaka',            # Great Leader
}


def get_yoddha_title(vijayas: int) -> str:
    """Get warrior title based on victories"""
    title = 'Dharmic_Pratishtha'
    for threshold, t in sorted(YODDHA_TITLES.items()):
        if vijayas >= threshold:
            title = t
    return title


@dataclass
class Shastrartha:
    """
    📜 Shastrartha - Spiritual Discourse
    An explainable teaching from a victorious warrior
    """
    yoddha_id: str                      # Warrior ID
    samay: datetime                     # Timestamp
    fraud_sample_idx: int              # Which fraud was caught
    features_that_mattered: List[str]  # Important features (weapons used)
    feature_values: Dict[str, float]   # Feature values
    upadesha: str                       # Teaching/Explanation (was explanation)
    vishwas: float                      # Confidence (was confidence)
    varna_focus: str                   # Varna specialization
    pattern_signature: np.ndarray = field(repr=False)
    
    def to_dict(self) -> Dict:
        return {
            'yoddha_id': self.yoddha_id,
            'samay': self.samay.isoformat(),
            'fraud_sample_idx': self.fraud_sample_idx,
            'features_that_mattered': self.features_that_mattered,
            'feature_values': self.feature_values,
            'upadesha': self.upadesha,
            'vishwas': self.vishwas,
            'varna_focus': self.varna_focus
        }


class Yoddha:
    """
    ⚔️ Yoddha - The Warrior
    
    Individual warrior fighting in Kurukshetra:
    - Has a Varna (specialization like Kshatriya duty)
    - Can share Shastrartha (wisdom)
    - Can hear and adapt to teachings
    - Can undergo Punarjanma (reincarnation/mutation)
    """
    
    def __init__(self, 
                 yoddha_id: str,
                 shastra_type: str = 'rf',    # Weapon type (was model_type)
                 varna: str = None,            # Specialization (was niche)
                 beeja: int = None,           # Seed (was seed)
                 janma: int = 0):             # Generation (was generation)
        self.yoddha_id = yoddha_id
        self.shastra_type = shastra_type
        self.varna = varna or random.choice(VARNA_TYPES)
        self.beeja = beeja or random.randint(0, 1000000)
        self.janma = janma
        
        # Combat statistics (Yuddha statistics)
        self.vijayas = 0               # Victories (was combat_wins)
        self.parajayas = 0             # Defeats (was combat_losses)
        self.samayas = 0               # Draws (was combat_draws)
        self.total_yuddhas = 0         # Total battles (was total_battles)
        self.f1_itihasa = deque(maxlen=50)  # F1 history (was f1_history)
        self.recent_vijayas = deque(maxlen=10)  # Recent wins
        
        # Varna expertise (niche expertise)
        self.varna_vijayas = {varna: 0 for varna in VARNA_TYPES}
        self.varna_f1_scores = {varna: [] for varna in VARNA_TYPES}
        self.varna_gambhira = 0        # Depth in varna (was niche_depth)
        
        # Shastra memory (story memory)
        self.shastras_heard = deque(maxlen=DHARMA_CONFIG['shastra_memory_size'])
        self.shastras_spoken = []
        
        # Punarjanma attributes (mutation)
        self.tapasya_rate = DHARMA_CONFIG['tapasya_rate']
        self.anveshana_score = 0       # Exploration (was exploration_score)
        self.shastra_params = self._generate_random_params()
        
        # The weapon (model)
        self.shastra = None            # Model (was model)
        self.is_tejasvi = False        # Is fitted (was is_fitted)
        
        # Feature names
        self.feature_names = None
        self.feature_importance_itihasa = deque(maxlen=100)
        
    def _generate_random_params(self) -> Dict:
        """Generate random shastra parameters for tapasya"""
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
        return params.get(self.shastra_type, params['rf'])
    
    def _get_shastra(self, n_features: int):
        """Create the actual ML weapon with current params"""
        params = self.shastra_params.copy()
        
        if self.shastra_type == 'rf':
            return cuRF(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                min_samples_split=params['min_samples_split'],
                max_features=params['max_features'],
                random_state=self.beeja
            )
        elif self.shastra_type == 'xgb':
            return XGBClassifier(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                learning_rate=params['learning_rate'],
                subsample=params['subsample'],
                colsample_bytree=params['colsample_bytree'],
                random_state=self.beeja,
                use_label_encoder=False,
                eval_metric='logloss',
                tree_method='hist',
                device='cuda'
            )
        elif self.shastra_type == 'lr':
            return cuLR(
                C=params['C'],
                penalty=params['penalty'],
                max_iter=params['max_iter']
            )
        elif self.shastra_type == 'svm':
            return cuSVC(
                C=params['C'],
                kernel=params['kernel'],
                gamma=params['gamma'],
                probability=True
            )
        else:
            raise ValueError(f"Unknown shastra type: {self.shastra_type}")
    
    def tapas(self, X: np.ndarray, y: np.ndarray, 
              sample_weight: Optional[np.ndarray] = None,
              feature_names: List[str] = None):
        """
        तपस् (Tapas) - Spiritual Training
        Train the warrior on data through spiritual practice
        """
        self.feature_names = feature_names or [f'V{i}' for i in range(X.shape[1])]
        
        # Ensure labels are integers
        y = np.asarray(y, dtype=np.int32)
        
        # Apply varna focus (sample weighting based on varna)
        if sample_weight is None and self.varna != 'sarvabhaum':
            sample_weight = self._apply_varna_weighting(X, y)
        
        # Create and train weapon
        self.shastra = self._get_shastra(X.shape[1])
        
        try:
            if sample_weight is not None:
                try:
                    self.shastra.fit(X, y, sample_weight=sample_weight)
                except TypeError:
                    self.shastra.fit(X, y)
            else:
                self.shastra.fit(X, y)
            self.is_tejasvi = True
        except Exception as e:
            print(f"  ⚠️ {self.yoddha_id} failed tapas: {e}")
            self.is_tejasvi = False
    
    def tapasya(self, aggressive: bool = False):
        """
        तपस्या (Tapasya) - Spiritual practice/mutation
        Used by Cataclysm to create aggressive variants
        """
        # Mutate shastra params
        mutation_rate = 0.5 if aggressive else 0.3
        
        for key in self.shastra_params:
            if np.random.random() < mutation_rate:
                val = self.shastra_params[key]
                if isinstance(val, (int, np.integer)):
                    self.shastra_params[key] = max(1, int(val * np.random.uniform(0.7, 1.3)))
                elif isinstance(val, float):
                    new_val = val * np.random.uniform(0.7, 1.3)
                    if key in ['subsample', 'colsample_bytree', 'colsample_bylevel']:
                        new_val = max(0.1, min(1.0, new_val))
                    self.shastra_params[key] = new_val
                elif isinstance(val, str):
                    if key == 'kernel':
                        self.shastra_params[key] = np.random.choice(['rbf', 'poly', 'sigmoid'])
                    elif key == 'penalty':
                        self.shastra_params[key] = np.random.choice(['l1', 'l2'])
        
        # Maybe change varna
        if aggressive and np.random.random() < 0.3:
            self.varna = np.random.choice(list(VARNA_TYPES))
    
    def _apply_varna_weighting(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Apply sample weights based on varna specialization"""
        weights = np.ones(len(y))
        fraud_mask = y == 1
        
        if self.varna == 'dhanurveda':
            # Weight high amount transactions (bow warfare - heavy strikes)
            amount_idx = -3
            amounts = X[:, amount_idx]
            high_amount_mask = amounts > np.percentile(amounts, 90)
            weights[high_amount_mask & fraud_mask] *= 3.0
            
        elif self.varna == 'kaal_chakra':
            # Weight unusual times (wheel of time)
            time_idx = -1
            times = X[:, time_idx]
            unusual_time_mask = (times % 86400 < 14400) | (times % 86400 > 79200)
            weights[unusual_time_mask & fraud_mask] *= 3.0
            
        elif self.varna in ['v14_tejas', 'v17_shakti']:
            # Weight extreme V14/V17 values (radiance/power)
            v_idx = 14 if self.varna == 'v14_tejas' else 17
            v_values = X[:, v_idx]
            extreme_mask = np.abs(v_values) > 2.0
            weights[extreme_mask & fraud_mask] *= 3.0
            
        elif self.varna == 'anomaly_drishti':
            # Weight statistical outliers (divine sight)
            z_scores = np.abs((X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8))
            outlier_mask = (z_scores > 2.5).any(axis=1)
            weights[outlier_mask & fraud_mask] *= 2.5
        
        return weights
    
    def yuddha(self, X: np.ndarray) -> np.ndarray:
        """
        युद्ध (Yuddha) - Battle/Prediction
        Make predictions (fight the battle)
        """
        if not self.is_tejasvi:
            return np.zeros(len(X))
        
        try:
            # 🛠️ Convert GPU arrays to numpy for sklearn models
            if hasattr(X, 'get'):
                X_np = X.get()
            else:
                X_np = np.asarray(X)
            
            # Check if model is sklearn-based (Visvarupa avatars)
            # sklearn models CANNOT run on GPU - use CPU only!
            model_type = type(self.shastra).__name__
            is_sklearn = model_type in ['IsolationForest', 'OneClassSVM', 'LocalOutlierFactor', 
                                          'EllipticEnvelope', 'DBSCAN', 'RandomForestClassifier',
                                          'GradientBoostingClassifier', 'ExtraTreesClassifier']
            
            if is_sklearn:
                # Always use CPU for sklearn models
                preds = self.shastra.predict(X_np)
                
                # Convert -1 (outlier) to 1 (fraud), 1 (inlier) to 0 (normal)
                if model_type in ['IsolationForest', 'LocalOutlierFactor', 'OneClassSVM', 'EllipticEnvelope']:
                    preds = (preds == -1).astype(int)
                    
                return np.asarray(preds)
            
            # For GPU-enabled models (cuml, etc.), try GPU first
            if CUPY_AVAILABLE:
                try:
                    X_gpu = cp.asarray(X_np)
                    preds = self.shastra.predict(X_gpu)
                    return cp.asnumpy(preds)
                except Exception as e:
                    print(f"  ⚠️ GPU failed for {self.yoddha_id}, using CPU: {e}")
                    preds = self.shastra.predict(X_np)
                    return np.asarray(preds)
            else:
                preds = self.shastra.predict(X_np)
                return np.asarray(preds)
        except Exception as e:
            print(f"  ⚠️ {self.yoddha_id} yuddha failed: {e}")
            return np.zeros(len(X))
    
    def yuddha_sambhavana(self, X: np.ndarray) -> np.ndarray:
        """
        युद्ध संभावना (Yuddha Sambhavana) - Battle Probability
        Get prediction probabilities
        """
        if not self.is_tejasvi:
            return np.zeros((len(X), 2))
        
        try:
            # 🛠️ Convert GPU arrays to numpy for sklearn models
            if hasattr(X, 'get'):
                X_np = X.get()
            else:
                X_np = np.asarray(X)
            
            # Check if model is sklearn-based (Visvarupa avatars)
            # sklearn models CANNOT run on GPU - use CPU only!
            model_type = type(self.shastra).__name__
            is_sklearn = model_type in ['IsolationForest', 'OneClassSVM', 'LocalOutlierFactor', 
                                          'EllipticEnvelope', 'DBSCAN', 'RandomForestClassifier',
                                          'GradientBoostingClassifier', 'ExtraTreesClassifier']
            
            if is_sklearn:
                # Always use CPU for sklearn models
                if hasattr(self.shastra, 'predict_proba'):
                    probs = self.shastra.predict_proba(X_np)
                else:
                    preds = self.shastra.predict(X_np)
                    probs = np.zeros((len(preds), 2))
                    probs[:, 0] = 1 - preds
                    probs[:, 1] = preds
                return np.asarray(probs)
            
            # For GPU-enabled models (cuml, etc.), try GPU first
            if CUPY_AVAILABLE:
                try:
                    X_gpu = cp.asarray(X_np)
                    if hasattr(self.shastra, 'predict_proba'):
                        probs = self.shastra.predict_proba(X_gpu)
                    else:
                        preds = self.shastra.predict(X_gpu)
                        probs = np.zeros((len(preds), 2))
                        probs[:, 0] = 1 - preds
                        probs[:, 1] = preds
                        return probs
                    return cp.asnumpy(probs)
                except Exception as e:
                    print(f"  ⚠️ GPU failed for {self.yoddha_id}, using CPU: {e}")
                    if hasattr(self.shastra, 'predict_proba'):
                        probs = self.shastra.predict_proba(X_np)
                    else:
                        preds = self.shastra.predict(X_np)
                        probs = np.zeros((len(preds), 2))
                        probs[:, 0] = 1 - preds
                        probs[:, 1] = preds
                    return np.asarray(probs)
            else:
                if hasattr(self.shastra, 'predict_proba'):
                    probs = self.shastra.predict_proba(X_np)
                else:
                    preds = self.shastra.predict(X_np)
                    probs = np.zeros((len(preds), 2))
                    probs[:, 0] = 1 - preds
                    probs[:, 1] = preds
                return np.asarray(probs)
        except Exception as e:
            print(f"  ⚠️ {self.yoddha_id} yuddha_sambhavana failed: {e}")
            return np.zeros((len(X), 2))
    
    def sphot_shastra(self, X: np.ndarray, y: np.ndarray, 
                     fraud_indices: List[int]) -> List[Shastrartha]:
        """
        स्फोट शास्त्र (Sphot Shastra) - Weapon Teaching
        Generate explainable teachings for frauds caught
        """
        shastras = []
        
        if not self.is_tejasvi or len(fraud_indices) == 0:
            return shastras
        
        try:
            if hasattr(self.shastra, 'feature_importances_'):
                importances = self.shastra.feature_importances_
            else:
                importances = np.ones(len(self.feature_names)) / len(self.feature_names)
            
            top_feature_indices = np.argsort(importances)[-5:][::-1]
            top_features = [self.feature_names[i] for i in top_feature_indices]
            
            for idx in fraud_indices[:5]:
                feature_values = {feat: float(X[idx, i]) 
                                for i, feat in enumerate(self.feature_names)}
                
                upadeshas = []
                for feat in top_features[:3]:
                    val = feature_values[feat]
                    if abs(val) > 2.0:
                        upadeshas.append(f"{feat} extreme ({val:.2f})")
                    elif val > np.percentile(X[:, self.feature_names.index(feat)], 95):
                        upadeshas.append(f"{feat} very high ({val:.2f})")
                    elif val < np.percentile(X[:, self.feature_names.index(feat)], 5):
                        upadeshas.append(f"{feat} very low ({val:.2f})")
                
                proba = self.yuddha_sambhavana(X[idx:idx+1])[0]
                vishwas = proba[1] if y[idx] == 1 else proba[0]
                
                shastra = Shastrartha(
                    yoddha_id=self.yoddha_id,
                    samay=datetime.now(),
                    fraud_sample_idx=idx,
                    features_that_mattered=top_features[:3],
                    feature_values={k: feature_values[k] for k in top_features[:3]},
                    upadesha="; ".join(upadeshas) if upadeshas else f"Complex pattern in {self.varna}",
                    vishwas=float(vishwas),
                    varna_focus=self.varna,
                    pattern_signature=X[idx].copy()
                )
                
                shastras.append(shastra)
                self.shastras_spoken.append(shastra)
        
        except Exception as e:
            print(f"  ⚠️ {self.yoddha_id} sphot_shastra failed: {e}")
        
        return shastras
    
    def shun_shastra(self, shastra: Shastrartha):
        """
        श्रुणु शास्त्र (Shrunu Shastra) - Hear Teaching
        Listen to a shastra and adapt strategy
        """
        self.shastras_heard.append(shastra)
        
        if shastra.features_that_mattered:
            for feat in shastra.features_that_mattered:
                if feat in self.feature_importance_itihasa:
                    pass
        
        if shastra.varna_focus != self.varna:
            self.anveshana_score += 0.05
            
            varna_shastras = [s for s in self.shastras_heard if s.varna_focus == shastra.varna_focus]
            if len(varna_shastras) > 5:
                avg_vishwas = np.mean([s.vishwas for s in varna_shastras])
                if avg_vishwas > 0.85 and random.random() < 0.1:
                    print(f"    🔄 {self.yoddha_id} considering varna switch from {self.varna} to {shastra.varna_focus}")
    
    def punarjanma(self) -> 'Yoddha':
        """
        पुनर्जन्म (Punarjanma) - Reincarnation
        Spawn a mutated child warrior
        """
        child_id = f"{self.yoddha_id}_putra_{self.janma + 1}_{random.randint(1000, 9999)}"
        
        # Mutate varna (small chance)
        child_varna = self.varna
        if random.random() < 0.3:
            child_varna = random.choice([v for v in VARNA_TYPES if v != self.varna])
        
        # Mutate shastra type (rare)
        child_shastra_type = self.shastra_type
        if random.random() < 0.1:
            child_shastra_type = random.choice(['rf', 'xgb', 'lr', 'svm'])
        
        # Mutate beeja
        child_beeja = self.beeja + random.randint(-1000, 1000)
        
        # Create child
        child = Yoddha(
            yoddha_id=child_id,
            shastra_type=child_shastra_type,
            varna=child_varna,
            beeja=child_beeja,
            janma=self.janma + 1
        )
        
        if child_shastra_type != self.shastra_type:
            child.shastra_params = child._generate_random_params()
        else:
            child.shastra_params = self._mutate_params(self.shastra_params)
        
        child.tapasya_rate = self.tapasya_rate * random.uniform(0.9, 1.1)
        child.shastras_heard = deque(list(self.shastras_heard)[-50:], maxlen=DHARMA_CONFIG['shastra_memory_size'])
        
        return child
    
    def _mutate_params(self, params: Dict) -> Dict:
        """Mutate shastra parameters through tapasya"""
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
    
    def get_yoddha_bal(self) -> float:
        """
        योद्धा बल (Yoddha Bal) - Warrior Strength
        Calculate overall warrior strength score
        """
        if self.total_yuddhas == 0:
            return 0.0
        
        base_score = self.vijayas / max(self.total_yuddhas, 1)
        
        if self.recent_vijayas:
            recent_win_rate = sum(self.recent_vijayas) / len(self.recent_vijayas)
            base_score += recent_win_rate * 0.1
        
        unique_varnas = sum(1 for v in self.varna_vijayas.values() if v > 0)
        base_score += unique_varnas * 0.02
        
        base_score += self.anveshana_score * DHARMA_CONFIG['gyan_bonus']
        
        return min(base_score, 1.0)
    
    def is_varna_dhani(self, varna: str) -> bool:
        """Check if this warrior dominates a particular varna"""
        return self.varna == varna and self.varna_vijayas[varna] > DHARMA_CONFIG['varna_depth_threshold']
    
    def __repr__(self):
        bal = self.get_yoddha_bal()
        title = get_yoddha_title(self.vijayas)
        return f"Yoddha({self.yoddha_id}, {title}, {self.shastra_type}, {self.varna}, vijayas={self.vijayas}, bal={bal:.3f})"


class Kurukshetra:
    """
    ⚔️ कुरुक्षेत्र (Kurukshetra) - The Battlefield of Righteousness
    
    The sacred field where:
    - Warriors battle for Dharma (truth/fraud detection)
    - Maharathi (champion) defends righteousness
    - Senapatis (challengers) seek to prove themselves
    - Shastrartha (wisdom) is shared
    - Punarjanma (evolution) occurs
    """
    
    def __init__(self, 
                 n_senapatis: int = 20,        # Challengers (was n_challengers)
                 yuddha_rounds: int = 50,      # Combat rounds
                 senas_batch: int = 1000):     # Batch size
        self.n_senapatis = n_senapatis
        self.yuddha_rounds = yuddha_rounds
        self.senas_batch = senas_batch
        
        # The Maharathi - supreme warrior (champion)
        self.maharathi: Optional[Yoddha] = None
        
        # Senapatis (challengers)
        self.senapatis: List[Yoddha] = []
        
        # Pitamahas (elders - still valuable!)
        self.pitamahas: List[Yoddha] = []
        
        # All shastras shared in Kurukshetra
        self.kshetra_shastras: deque = deque(maxlen=DHARMA_CONFIG['shastra_memory_size'])
        
        # Yuddha history
        self.yuddha_itihasa: List[Dict] = []
        
        # Best sena (ensemble) found
        self.sreshtha_sena: Optional[List[Yoddha]] = None
        self.sreshtha_f1 = 0.0
        
        print("⚔️ DHARMAKSHETRA KURUKSHETRA INITIALIZED ⚔️")
        print(f"   युद्ध राउंड्स (Yuddha Rounds): {yuddha_rounds}")
        print(f"   सेनापतिस (Senapatis): {n_senapatis}")
        print(f"   सेना बैच (Senas Batch): {senas_batch}")
        print("   धर्म की रक्षा के लिए तैयार! (Ready to defend Dharma!) 🕉️\n")
    
    def abhisheka_maharathi(self, X_train: np.ndarray, y_train: np.ndarray,
                           feature_names: List[str] = None,
                           existing_model: Optional = None):
        """
        अभिषेक महारथी (Abhisheka Maharathi) - Crown the Champion
        Initialize the Maharathi
        """
        print("👑 अभिषेक महारथी (Crowning the Maharathi)...")
        
        if existing_model is not None:
            self.maharathi = Yoddha(
                yoddha_id='Maharathi_Arjun',
                shastra_type='xgb',
                varna='sarvabhaum',
                beeja=42
            )
            self.maharathi.shastra = existing_model
            self.maharathi.is_tejasvi = True
            self.maharathi.feature_names = feature_names
        else:
            self.maharathi = Yoddha(
                yoddha_id='Maharathi_Yudhishthir',
                shastra_type='xgb',
                varna='sarvabhaum',
                beeja=42
            )
            self.maharathi.tapas(X_train, y_train, feature_names=feature_names)
        
        self.maharathi.vijayas = DHARMA_CONFIG['min_vijayas_for_maharathi']
        self.maharathi.total_yuddhas = DHARMA_CONFIG['min_vijayas_for_maharathi']
        
        print(f"   महारथी: {self.maharathi.yoddha_id}")
        print(f"   शास्त्र: {self.maharathi.shastra_type}")
        print(f"   वर्ण: {self.maharathi.varna}")
        print(f"   स्टेटस: धर्म की रक्षा के लिए तैयार! (Ready to defend Dharma!) 🛡️\n")
    
    def prakat_senapatis(self, X_train: np.ndarray, y_train: np.ndarray,
                        feature_names: List[str] = None):
        """
        प्रकट सेनापतिस (Prakat Senapatis) - Manifest Challengers
        Spawn initial pool of challengers
        """
        print(f"⚔️ प्रकट {self.n_senapatis} सेनापतिस (Manifesting {self.n_senapatis} Senapatis)...")
        
        shastra_types = ['rf', 'xgb', 'lr', 'svm']
        
        for i in range(self.n_senapatis):
            shastra_type = random.choice(shastra_types)
            varna = random.choice(VARNA_TYPES)
            beeja = random.randint(0, 1000000)
            
            senapati = Yoddha(
                yoddha_id=f"Senapati_{i:03d}",
                shastra_type=shastra_type,
                varna=varna,
                beeja=beeja,
                janma=0
            )
            
            senapati.tapas(X_train, y_train, feature_names=feature_names)
            
            if senapati.is_tejasvi:
                self.senapatis.append(senapati)
                print(f"   ✓ {senapati.yoddha_id} ({shastra_type}, {varna}) तैयार!")
            else:
                print(f"   ✗ Senapati {i} failed to manifest")
        
        print(f"   {len(self.senapatis)} सेनापतिस युद्ध के लिए तैयार!\n")
    
    def yuddha(self, senapati: Yoddha, 
              X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """
        युद्ध (Yuddha) - Battle
        Run a single battle between Maharathi and Senapati
        """
        batch_size = min(self.senas_batch, len(X_val))
        indices = np.random.choice(len(X_val), batch_size, replace=False)
        X_batch = X_val[indices]
        y_batch = y_val[indices]
        
        # 🛠️ Convert to numpy if GPU arrays (for Visvarupa avatars with sklearn models)
        if hasattr(X_batch, 'get'):
            X_batch = X_batch.get()
        else:
            X_batch = np.asarray(X_batch)
        
        maharathi_preds = self.maharathi.yuddha(X_batch)
        senapati_preds = senapati.yuddha(X_batch)
        
        maharathi_f1 = f1_score(y_batch, maharathi_preds, zero_division=0)
        senapati_f1 = f1_score(y_batch, senapati_preds, zero_division=0)
        
        if senapati_f1 > maharathi_f1:
            vijeta = senapati
            parajit = self.maharathi
            maharathi_won = False
            margin = senapati_f1 - maharathi_f1
        elif maharathi_f1 > senapati_f1:
            vijeta = self.maharathi
            parajit = senapati
            maharathi_won = True
            margin = maharathi_f1 - senapati_f1
        else:
            vijeta = None
            maharathi_won = None
            margin = 0
        
        senapati.total_yuddhas += 1
        self.maharathi.total_yuddhas += 1
        
        if vijeta is None:
            senapati.samayas += 1
            self.maharathi.samayas += 1
            senapati.recent_vijayas.append(0.5)
            self.maharathi.recent_vijayas.append(0.5)
        elif vijeta == senapati:
            senapati.vijayas += 1
            senapati.recent_vijayas.append(1)
            self.maharathi.recent_vijayas.append(0)
            senapati.varna_vijayas[senapati.varna] += 1
        else:
            self.maharathi.vijayas += 1
            self.maharathi.recent_vijayas.append(1)
            senapati.recent_vijayas.append(0)
            self.maharathi.varna_vijayas[self.maharathi.varna] += 1
        
        senapati.f1_itihasa.append(senapati_f1)
        self.maharathi.f1_itihasa.append(maharathi_f1)
        
        shastras = []
        if vijeta is not None:
            fraud_mask = y_batch == 1
            if vijeta == senapati:
                correct_frauds = np.where((senapati_preds == 1) & fraud_mask)[0]
            else:
                correct_frauds = np.where((maharathi_preds == 1) & fraud_mask)[0]
            
            if len(correct_frauds) > 0:
                original_indices = indices[correct_frauds]
                shastras = vijeta.sphot_shastra(X_val, y_val, original_indices.tolist())
                
                for shastra in shastras:
                    self.kshetra_shastras.append(shastra)
                    parajit.shun_shastra(shastra)
        
        return {
            'senapati': senapati.yoddha_id,
            'maharathi_f1': maharathi_f1,
            'senapati_f1': senapati_f1,
            'vijeta': vijeta.yoddha_id if vijeta else 'Samaya',
            'maharathi_won': maharathi_won,
            'margin': margin,
            'shastras_shared': len(shastras),
            'senapati_varna': senapati.varna
        }
    
    def patan_maharathi(self, new_maharathi: Yoddha):
        """
        पतन महारथी (Patan Maharathi) - Dethrone Champion
        Replace the Maharathi
        """
        print(f"\n🏆 महारथी का पतन! (Fall of the Maharathi!)")
        print(f"   पूर्व महारथी: {self.maharathi.yoddha_id} (पितामह बन गया)")
        print(f"   नवीन महारथी: {new_maharathi.yoddha_id}")
        print(f"   विजय अंक: {new_maharathi.vijayas}")
        print(f"   वर्ण '{new_maharathi.varna}' में विजय: {new_maharathi.varna_vijayas[new_maharathi.varna]}")
        
        self.pitamahas.append(self.maharathi)
        self.maharathi = new_maharathi
        
        if new_maharathi in self.senapatis:
            self.senapatis.remove(new_maharathi)
        
        print(f"   👑 नवीन महारथी की जय! (Victory to the new Maharathi!)\n")
    
    def punarjanma_sena(self, X_train: np.ndarray, y_train: np.ndarray,
                       feature_names: List[str] = None):
        """
        पुनर्जन्म सेना (Punarjanma Sena) - Reincarnate Army
        Evolve the challenger population
        """
        print("🧬 पुनर्जन्म सेना (Reincarnating the Army)...")
        
        if len(self.senapatis) == 0:
            print("   No senapatis to evolve!")
            return
        
        sorted_senapatis = sorted(
            self.senapatis, 
            key=lambda x: x.get_yoddha_bal(), 
            reverse=True
        )
        
        n_parents = max(1, len(sorted_senapatis) // 3)
        parents = sorted_senapatis[:n_parents]
        
        new_children = []
        for parent in parents:
            for _ in range(2):
                child = parent.punarjanma()
                child.tapas(X_train, y_train, feature_names=feature_names)
                
                if child.is_tejasvi:
                    new_children.append(child)
                    print(f"   🌱 {child.yoddha_id} born from {parent.yoddha_id} ({child.varna})")
        
        n_to_remove = len(new_children)
        if n_to_remove > 0 and len(self.senapatis) > n_to_remove:
            worst = sorted_senapatis[-n_to_remove:]
            for w in worst:
                if w in self.senapatis:
                    self.senapatis.remove(w)
                    print(f"   💀 {w.yoddha_id} retired (low bal: {w.get_yoddha_bal():.3f})")
        
        self.senapatis.extend(new_children)
        
        print(f"   सेना: {len(self.senapatis)} सेनापतिस")
        print(f"   पितामह: {len(self.pitamahas)}")
        print(f"   पुनर्जन्म पूर्ण!\n")
    
    def maha_yuddha(self, X_val: np.ndarray, y_val: np.ndarray):
        """
        महा युद्ध (Maha Yuddha) - Great Battle
        Run a full tournament round
        """
        print(f"⚔️ महा युद्ध प्रारम्भ... (Maha Yuddha Starting...)")
        print(f"   युद्ध: {len(self.senapatis)}")
        
        round_results = []
        
        for i, senapati in enumerate(self.senapatis):
            result = self.yuddha(senapati, X_val, y_val)
            round_results.append(result)
            
            if result['vijeta'] == senapati.yoddha_id:
                if senapati.vijayas >= DHARMA_CONFIG['min_vijayas_for_maharathi']:
                    if result['senapati_f1'] > result['maharathi_f1'] + 0.01:
                        self.patan_maharathi(senapati)
            
            if (i + 1) % 5 == 0:
                print(f"   प्रगति: {i+1}/{len(self.senapatis)} युद्ध पूर्ण")
        
        self.yuddha_itihasa.extend(round_results)
        
        vijayas = sum(1 for r in round_results if r['maharathi_won'] == True)
        parajayas = sum(1 for r in round_results if r['maharathi_won'] == False)
        samayas = sum(1 for r in round_results if r['maharathi_won'] is None)
        
        print(f"   परिणाम: महारथी {vijayas}V-{parajayas}D-{samayas}S")
        print(f"   महा युद्ध पूर्ण!\n")
        
        return round_results
    
    def tapasya(self, X_train: np.ndarray, y_train: np.ndarray,
               X_val: np.ndarray, y_val: np.ndarray,
               feature_names: List[str] = None,
               existing_model: Optional = None) -> 'Kurukshetra':
        """
        तपस्या (Tapasya) - Spiritual Training
        Full training cycle in Kurukshetra
        """
        print("="*70)
        print("⚔️ कुरुक्षेत्र - धर्मक्षेत्र में तपस्या ⚔️")
        print("="*70)
        print()
        
        self.abhisheka_maharathi(X_train, y_train, feature_names, existing_model)
        self.prakat_senapatis(X_train, y_train, feature_names)
        
        for yuddha_num in range(self.yuddha_rounds):
            print(f"\n{'='*70}")
            print(f"⚔️ युद्ध राउंड {yuddha_num + 1}/{self.yuddha_rounds}")
            print(f"{'='*70}")
            
            self.maha_yuddha(X_val, y_val)
            
            if (yuddha_num + 1) % 5 == 0:
                self.punarjanma_sena(X_train, y_train, feature_names)
            
            if (yuddha_num + 1) % 10 == 0:
                self._update_sreshtha_f1(X_val, y_val)
                self._print_sthiti()
        
        print("\n" + "="*70)
        print("⚔️ कुरुक्षेत्र युद्ध सम्पूर्ण! (Kurukshetra Battle Complete!)")
        print("="*70)
        
        self._build_sreshtha_sena(X_val, y_val)
        
        return self
    
    def _print_sthiti(self):
        """Print current Kurukshetra status"""
        print(f"\n📊 कुरुक्षेत्र स्थिति:")
        print(f"   महारथी: {self.maharathi.yoddha_id} (विजय: {self.maharathi.vijayas})")
        print(f"   श्रेष्ठ सेनापति: {max(self.senapatis, key=lambda x: x.get_yoddha_bal()).yoddha_id}")
        print(f"   पितामह: {len(self.pitamahas)}")
        print(f"   कुल शास्त्र: {len(self.kshetra_shastras)}")
        print(f"   श्रेष्ठ F1: {self.sreshtha_f1:.4f}")
    
    def _update_sreshtha_f1(self, X_val: np.ndarray, y_val: np.ndarray):
        """Evaluate current sena and update best F1"""
        try:
            # 🛠️ Convert GPU arrays to numpy for sklearn models
            if hasattr(X_val, 'get'):
                X_val = X_val.get()
            else:
                X_val = np.asarray(X_val)
            
            all_yoddhas = [self.maharathi] + self.pitamahas + self.senapatis
            all_yoddhas.sort(key=lambda x: x.get_yoddha_bal(), reverse=True)
            top_yoddhas = all_yoddhas[:7]
            
            predictions = []
            for yoddha in top_yoddhas:
                if yoddha.is_tejasvi:
                    try:
                        pred = yoddha.yuddha(X_val)
                        if CUPY_AVAILABLE and hasattr(pred, 'get'):
                            pred = pred.get()
                        predictions.append(pred)
                    except:
                        pass
            
            if len(predictions) > 0:
                sena_preds = np.round(np.mean(predictions, axis=0)).astype(int)
                current_f1 = f1_score(y_val, sena_preds)
                
                if current_f1 > self.sreshtha_f1:
                    self.sreshtha_f1 = current_f1
                    
        except Exception as e:
            pass
    
    def _build_sreshtha_sena(self, X_val: np.ndarray, y_val: np.ndarray):
        """Build the final sena from top performers"""
        print("\n🏗️ निर्माण श्रेष्ठ सेना (Building Supreme Army)...")
        
        all_yoddhas = [self.maharathi] + self.pitamahas + self.senapatis
        all_yoddhas.sort(key=lambda x: x.get_yoddha_bal(), reverse=True)
        
        self.sreshtha_sena = all_yoddhas[:7]
        
        sena_preds = self._sena_yuddha(X_val, use_sreshtha=True)
        self.sreshtha_f1 = f1_score(y_val, sena_preds)
        
        print(f"   चयनित {len(self.sreshtha_sena)} योद्धा:")
        for i, yoddha in enumerate(self.sreshtha_sena, 1):
            title = get_yoddha_title(yoddha.vijayas)
            print(f"   {i}. {yoddha.yoddha_id} ({title}, {yoddha.shastra_type}, {yoddha.varna}) - बल: {yoddha.get_yoddha_bal():.3f}")
        
        print(f"\n   🎯 श्रेष्ठ सेना F1: {self.sreshtha_f1:.4f}")
    
    def _sena_yuddha(self, X: np.ndarray, 
                    use_sreshtha: bool = True) -> np.ndarray:
        """Make sena prediction"""
        # 🛠️ Convert GPU arrays to numpy for sklearn models
        if hasattr(X, 'get'):
            X = X.get()
        else:
            X = np.asarray(X)
        
        if use_sreshtha and self.sreshtha_sena:
            yoddhas = self.sreshtha_sena
        else:
            yoddhas = [self.maharathi] + self.senapatis[:5]
        
        weights = [y.get_yoddha_bal() for y in yoddhas]
        total_weight = sum(weights)
        
        if total_weight == 0:
            weights = [1.0] * len(yoddhas)
            total_weight = len(yoddhas)
        
        weights = [w / total_weight for w in weights]
        
        votes = np.zeros(len(X))
        for yoddha, weight in zip(yoddhas, weights):
            preds = yoddha.yuddha(X)
            votes += preds * weight
        
        return (votes >= 0.5).astype(int)
    
    def yuddha_predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the trained sena"""
        return self._sena_yuddha(X, use_sreshtha=True)
    
    def yuddha_sambhavana(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions"""
        if not self.sreshtha_sena:
            return np.zeros((len(X), 2))
        
        weights = [y.get_yoddha_bal() for y in self.sreshtha_sena]
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        probs = np.zeros((len(X), 2))
        for yoddha, weight in zip(self.sreshtha_sena, weights):
            yoddha_probs = yoddha.yuddha_sambhavana(X)
            probs += yoddha_probs * weight
        
        return probs
    
    def save(self, filepath: str):
        """Save the Kurukshetra state"""
        state = {
            'maharathi': self.maharathi,
            'senapatis': self.senapatis,
            'pitamahas': self.pitamahas,
            'sreshtha_sena': self.sreshtha_sena,
            'sreshtha_f1': self.sreshtha_f1,
            'yuddha_itihasa': self.yuddha_itihasa,
            'config': DHARMA_CONFIG
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        print(f"💾 कुरुक्षेत्र सहेजा गया: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'Kurukshetra':
        """Load a saved Kurukshetra"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        kshetra = cls(
            n_senapatis=len(state['senapatis']),
            yuddha_rounds=state['config']['yuddha_rounds'],
            senas_batch=state['config']['senas_batch']
        )
        
        kshetra.maharathi = state['maharathi']
        kshetra.senapatis = state['senapatis']
        kshetra.pitamahas = state['pitamahas']
        kshetra.sreshtha_sena = state['sreshtha_sena']
        kshetra.sreshtha_f1 = state['sreshtha_f1']
        kshetra.yuddha_itihasa = state['yuddha_itihasa']
        
        print(f"💾 कुरुक्षेत्र लोड किया गया: {filepath}")
        return kshetra


def run_kurukshetra_experiment(X_train: np.ndarray, y_train: np.ndarray,
                              X_val: np.ndarray, y_val: np.ndarray,
                              feature_names: List[str] = None,
                              existing_model: Optional = None,
                              save_path: str = None) -> Tuple[Kurukshetra, float]:
    """Run a full Kurukshetra experiment"""
    kshetra = Kurukshetra(
        n_senapatis=20,
        yuddha_rounds=50,
        senas_batch=1000
    )
    
    kshetra.tapasya(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        feature_names=feature_names,
        existing_model=existing_model
    )
    
    if save_path:
        kshetra.save(save_path)
    
    return kshetra, kshetra.sreshtha_f1


if __name__ == "__main__":
    print("⚔️ कुरुक्षेत्र - धर्मक्षेत्र में युद्ध (Battle in the Field of Righteousness)")
    print("   Run with your data using run_kurukshetra_experiment()")
    print("   GPU Acceleration: ENABLED")
    print("   Shastrartha Sharing: ENABLED")
    print("   Punarjanma: ENABLED")
    print("   धर्म की रक्षा के लिए तैयार! (Ready to defend Dharma!) 🕉️")
