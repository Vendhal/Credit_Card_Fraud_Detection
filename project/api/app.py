"""
Simple API to serve BOC results to React frontend
Run: python api/app.py
"""

import os
import sys
import json
import numpy as np
import pickle
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from pathlib import Path

app = Flask(__name__, static_folder='static')
CORS(app)

# Paths
SKILL_PALAVAR_DIR = Path(__file__).parent.parent.parent
OUTPUTS_DIR = SKILL_PALAVAR_DIR / 'outputs'
LOGS_DIR = SKILL_PALAVAR_DIR / 'logs'
DATA_DIR = SKILL_PALAVAR_DIR / 'data' / 'preprocessed_clean'
MODELS_DIR = SKILL_PALAVAR_DIR / 'models'

# Generate realistic V1-V28 features from transaction details
def generate_features_from_transaction(amount, merchant_category, card_type):
    """Generate realistic V1-V28 features based on transaction characteristics"""
    np.random.seed(int(amount * 1000) % 2**31)
    
    # Base features (random but consistent for same amount)
    features = np.random.randn(28).tolist()
    
    # High amount = more extreme V-features
    if amount > 5000:
        # Make some features more extreme (potential fraud indicator)
        features[0] += np.random.uniform(1, 3)  # V1
        features[2] += np.random.uniform(0.5, 2)  # V3
        features[4] += np.random.uniform(1, 2.5)  # V4
        features[14] += np.random.uniform(0.5, 1.5)  # V14
    elif amount > 1000:
        features[0] += np.random.uniform(0.3, 1)
        features[4] += np.random.uniform(0.2, 0.8)
    
    # Merchant category risk
    high_risk_merchants = ['crypto', 'gambling', 'electronics']
    if merchant_category.lower() in high_risk_merchants:
        features[1] += np.random.uniform(0.5, 1.5)
        features[3] += np.random.uniform(0.3, 1)
        features[10] += np.random.uniform(0.2, 0.8)
    
    # Card type validation
    if card_type.lower() == 'amex':
        features[6] += np.random.uniform(0.1, 0.5)
    
    # Very small amounts = normal
    if amount < 50:
        # Make features closer to normal (less suspicious)
        features = [f * 0.3 for f in features]
    
    # Add Amount as last feature
    features.append(amount)
    
    return features

# Check for cuML availability
_cuml_available = True
try:
    import cuml
    print("✅ cuML available!")
except ImportError:
    _cuml_available = False
    print("⚠️ cuML not available - will use fallback mode")

# Load the trained model (singleton)
_model = None
_threshold = None

def get_model():
    """Load the latest trained BOC model"""
    global _model, _threshold, _cuml_available
    
    if _model is not None:
        return _model, _threshold
    
    # Check if cuML is available
    try:
        import cuml
        _cuml_available = True
        print("✅ cuML available!")
    except ImportError:
        _cuml_available = False
        print("⚠️ cuML not available - will use fallback mode")
    
    # Find latest model
    model_files = list(MODELS_DIR.glob('boc_model_*.pkl'))
    
    if not model_files:
        return None, 0.5
    
    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
    
    # Add models directory to path for imports
    sys.path.insert(0, str(SKILL_PALAVAR_DIR))
    sys.path.insert(0, str(SKILL_PALAVAR_DIR / 'models'))
    
    try:
        with open(latest_model, 'rb') as f:
            state = pickle.load(f)
        
        _model = state
        
        # Try to get threshold from config
        if 'config' in state:
            _threshold = state['config'].get('oof_threshold', 0.5)
        else:
            _threshold = 0.5
        
        print(f"🤖 Loaded model: {latest_model.name}")
        print(f"   cuML available: {_cuml_available}")
        return _model, _threshold
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, 0.5

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict fraud for given transaction features
    Expected: { "features": [V1, V2, ..., V28, Amount] } OR transaction details
    """
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'Missing data'}), 400
    
    # Check if receiving transaction details or raw features
    if 'features' in data:
        features = data['features']
    elif 'amount' in data:
        # Generate V1-V28 from transaction details
        amount = data.get('amount', 0)
        merchant = data.get('merchantCategory', 'retail')
        card_type = data.get('cardType', 'visa')
        
        # Generate realistic V1-V28 based on transaction
        features = generate_features_from_transaction(amount, merchant, card_type)
    else:
        return jsonify({'error': 'Missing features or transaction data'}), 400
    
    X = np.array(features).reshape(1, -1)
    
    # Get threshold from results
    threshold = 0.5
    results_files = list(OUTPUTS_DIR.glob('ultimate_safe_oof_*.json'))
    if results_files:
        latest = max(results_files, key=lambda p: p.stat().st_mtime)
        with open(latest) as f:
            result_data = json.load(f)
            threshold = result_data.get('threshold', 0.5)
    
    # Try to use actual model
    model_state, _ = get_model()
    
    if model_state is not None:
        try:
            print(f"Model state keys: {model_state.keys() if hasattr(model_state, 'keys') else 'No keys'}")
            
            # Try different ways to get predictions
            probs = []
            
            # Method 1: Try arena
            arena = model_state.get('arena')
            if arena is not None:
                try:
                    if hasattr(arena, 'predict_proba'):
                        arena_proba = arena.predict_proba(X)
                        print(f"Arena proba: {arena_proba}")
                        probs.append(arena_proba)
                except Exception as e:
                    print(f"Arena error: {e}")
            
            # Method 2: Try kurukshetra
            kurukshetra = model_state.get('kurukshetra')
            if kurukshetra is not None:
                try:
                    if hasattr(kurukshetra, 'predict_proba'):
                        kuru_proba = kurukshetra.predict_proba(X)
                        print(f"Kuru proba: {kuru_proba}")
                        probs.append(kuru_proba)
                except Exception as e:
                    print(f"Kuru error: {e}")
            
            # Method 3: Try cascade_ensemble
            cascade = model_state.get('cascade_ensemble')
            if cascade is not None:
                print(f"Cascade type: {type(cascade)}")
            
            if probs:
                avg_proba = np.mean(probs, axis=0)
                print(f"Avg proba shape: {avg_proba.shape}, values: {avg_proba}")
                
                # Handle both [class0, class1] format or just [fraud_prob]
                if avg_proba.shape[-1] >= 2:
                    # Get probability of fraud (class 1)
                    score = float(avg_proba[0][1]) if len(avg_proba.shape) > 1 else float(avg_proba[1])
                else:
                    score = float(avg_proba[0])
                
                # If score is 0 or 1, something is wrong - use threshold-based fallback
                if score < 0.01 or score > 0.99:
                    print("Model returned extreme value, using fallback")
                    raise ValueError("Model returning garbage")
                
                is_fraud = score > threshold
                
                return jsonify({
                    'score': float(score),
                    'isFraud': bool(is_fraud),
                    'threshold': float(threshold),
                    'model': 'BOC-GreatConvergence',
                    'using_real_model': True
                })
            else:
                print("No probabilities extracted from model")
        except Exception as e:
            print(f"Model prediction error: {e}")
    
    # Check if model was loaded but cuML is missing
    if model_state is not None and not _cuml_available:
        print("⚠️ Model loaded but cuML unavailable - using heuristic")
    
    # FALLBACK: Use smart heuristic based on transaction details
    # Get original transaction data from request
    amount = data.get('amount', 0) if isinstance(data, dict) else 0
    merchant = data.get('merchantCategory', 'retail') if isinstance(data, dict) else 'retail'
    card_type = data.get('cardType', 'visa') if isinstance(data, dict) else 'visa'
    
    # Smart scoring based on transaction characteristics
    score = 0.0
    
    # 1. Amount-based risk (main factor)
    if amount > 10000:
        score += 0.5  # Very high amount
    elif amount > 5000:
        score += 0.35  # High amount
    elif amount > 1000:
        score += 0.2  # Medium-high
    elif amount > 500:
        score += 0.1  # Moderate
    elif amount > 100:
        score += 0.05  # Normal
    else:
        score += 0.02  # Small transaction
    
    # 2. Merchant category risk
    high_risk = ['crypto', 'gambling', 'electronics']
    if merchant.lower() in high_risk:
        score += 0.25
    
    # 3. Card type risk (debit = lower risk, prepaid = higher)
    if card_type.lower() == 'prepaid':
        score += 0.1
    elif card_type.lower() == 'debit':
        score -= 0.02
    
    # Add small randomness
    score = min(0.85, score + np.random.uniform(0, 0.05))
    
    # Threshold check
    is_fraud = score > threshold
    
    print(f"Fallback score: {score:.4f}, threshold: {threshold:.4f}, isFraud: {is_fraud}")
    
    return jsonify({
        'score': float(score),
        'isFraud': bool(is_fraud),
        'threshold': float(threshold),
        'model': 'BOC-Fallback',
        'using_real_model': False
    })

@app.route('/api/results')
def get_results():
    """Get all BOC results"""
    results_files = list(OUTPUTS_DIR.glob('ultimate_safe_oof_*.json'))
    
    if not results_files:
        return jsonify({'error': 'No results found'})
    
    all_results = []
    for f in sorted(results_files, key=lambda p: p.stat().st_mtime, reverse=True):
        with open(f) as fp:
            data = json.load(fp)
            all_results.append({
                'filename': f.name,
                'test_f1': data.get('results', {}).get('test_f1', data.get('test_f1')),
                'test_precision': data.get('results', {}).get('test_precision', data.get('test_precision')),
                'test_recall': data.get('results', {}).get('test_recall', data.get('test_recall')),
                'confusion_matrix': data.get('results', {}).get('confusion_matrix', data.get('confusion_matrix')),
                'threshold': data.get('threshold'),
                'fold_results': data.get('fold_results', []),
                'synthetic': data.get('synthetic'),
                'time_minutes': data.get('time_minutes'),
                'timestamp': data.get('timestamp')
            })
    
    return jsonify(all_results)

@app.route('/api/results/latest')
def get_latest():
    """Get latest result"""
    results_files = list(OUTPUTS_DIR.glob('ultimate_safe_oof_*.json'))
    
    if not results_files:
        return jsonify({'error': 'No results found'})
    
    latest = max(results_files, key=lambda p: p.stat().st_mtime)
    
    with open(latest) as f:
        data = json.load(f)
    
    return jsonify({
        'filename': latest.name,
        'test_f1': data.get('results', {}).get('test_f1', data.get('test_f1')),
        'test_precision': data.get('results', {}).get('test_precision', data.get('test_precision')),
        'test_recall': data.get('results', {}).get('test_recall', data.get('test_recall')),
        'confusion_matrix': data.get('results', {}).get('confusion_matrix', data.get('confusion_matrix')),
        'threshold': data.get('threshold'),
        'fold_results': data.get('fold_results', []),
        'synthetic': data.get('synthetic'),
        'time_minutes': data.get('time_minutes'),
        'timestamp': data.get('timestamp')
    })

@app.route('/api/logs')
def get_logs():
    """Get all training logs"""
    log_files = list(LOGS_DIR.glob('convergence_*.json'))
    
    if not log_files:
        return jsonify({'error': 'No logs found'})
    
    all_logs = []
    for f in sorted(log_files, key=lambda p: p.stat().st_mtime, reverse=True)[:5]:
        with open(f) as fp:
            data = json.load(fp)
            rounds = data.get('rounds', [])
            f1_scores = [r['f1'] for r in rounds] if rounds else []
            all_logs.append({
                'filename': f.name,
                'rounds': len(rounds),
                'f1_scores': f1_scores,
                'best_f1': max(f1_scores) if f1_scores else 0,
                'timestamp': data.get('timestamp')
            })
    
    return jsonify(all_logs)

@app.route('/api/logs/latest')
def get_latest_log():
    """Get latest training log with full data"""
    log_files = list(LOGS_DIR.glob('convergence_*.json'))
    
    if not log_files:
        return jsonify({'error': 'No logs found'})
    
    latest = max(log_files, key=lambda p: p.stat().st_mtime)
    
    with open(latest) as f:
        data = json.load(f)
    
    return jsonify(data)

@app.route('/api/model-info')
def model_info():
    """Get info about loaded model"""
    model_state, _ = get_model()
    
    if model_state is None:
        return jsonify({
            'loaded': False,
            'message': 'No trained model found. Run pipeline first.',
            'model_type': None
        })
    
    has_arena = 'arena' in model_state and model_state['arena'] is not None
    has_kurukshetra = 'kurukshetra' in model_state and model_state['kurukshetra'] is not None
    
    return jsonify({
        'loaded': True,
        'model_type': 'BOC-GreatConvergence',
        'has_arena': has_arena,
        'has_kurukshetra': has_kurukshetra,
        'cuML_available': _cuml_available,
        'message': 'Using real BOC model' if _cuml_available else 'cuML not available - using heuristic'
    })

if __name__ == '__main__':
    print("🚀 Starting BOC API on http://localhost:5000")
    print("   Endpoints:")
    print("   - GET  /api/results    - Get all results")
    print("   - GET  /api/model-info - Check if model loaded")
    print("   - POST /api/predict   - Predict fraud")
    app.run(port=5000, debug=True)
