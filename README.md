# 🎯 BOC - Battle of Civilizations

Fraud Detection System using AI Battle Strategy

## What is BOC?

Battle of Civilizations (BOC) is an AI framework that uses:
- **Hive GAN** - Generates synthetic fraud data using 4 different GANs
- **Great Convergence** - East (LightGBM) vs West (XGBoost) battle system
- **Safe OOF** - Rigorous evaluation to prevent overfitting

**Results: Up to 89% F1 on credit card fraud detection!**

---

## Prerequisites

### 1. Install Dependencies

```bash
# Python - install all required packages
pip install -r requirements.txt

# React - install frontend dependencies
cd project
npm install
```

### 2. Get the Data

Download the Credit Card Fraud dataset from Kaggle:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place `creditcard.csv` in `data/raw/`

---

## Complete Pipeline (3 Steps)

### Step 1: Preprocess Data
```bash
python preprocess_clean.py
```
Creates train/test splits in `data/preprocessed_clean/`

### Step 2: Train Hive GAN (Optional - generates synthetic frauds)
```bash
python train_hive_gan.py
```
Creates synthetic fraud samples in `outputs/hive_synthetic.npy`

### Step 3: Run BOC Pipeline
```bash
python ultimate_hive_convergence_safe_oof.py
```

This trains the full model and saves results to `outputs/`

**Time**: ~1.5-2 hours on GPU

---

## Running the Dashboard

### Option A: Windows
```bash
# Terminal 1: Start API
cd project/api
python app.py

# Terminal 2: Start React
cd project
npm run dev
```

### Option B: WSL (with GPU/cuML)
```bash
# Run in WSL:
bash project/api/run_api_wsl.sh

# Then in another terminal:
cd project
npm run dev
```

Open http://localhost:5173

---

## Files

| File | Description |
|------|-------------|
| `ultimate_hive_convergence_safe_oof.py` | Main pipeline |
| `preprocess_clean.py` | Data preprocessing |
| `train_hive_gan.py` | Train GANs |
| `project/` | React Dashboard |

---

## Results

- **OOF F1**: 0.92-0.93
- **Test F1**: 0.85-0.90
- **Precision**: ~0.91
- **Recall**: ~0.89
