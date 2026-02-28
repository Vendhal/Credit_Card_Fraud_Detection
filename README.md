# 🎯 BOC - Battle of Civilizations

Fraud Detection System using AI Battle Strategy

## Quick Start

### 1. Install Dependencies

```bash
# Python
pip install -r requirements.txt

# React (in project folder)
cd project
npm install
```

### 2. Run the Pipeline

```bash
cd skill_palavar
python ultimate_hive_convergence_safe_oof.py
```

Results will be saved to `outputs/`

### 3. Run the Dashboard

#### Option A: Windows (Simple - fallback mode)
```bash
# Terminal 1: Start API
cd project/api
python app.py

# Terminal 2: Start React
cd project
npm run dev
```

#### Option B: WSL (Full - REAL BOC model with cuML!)
```bash
# Run in WSL terminal:
bash project/api/run_api_wsl.sh

# Then in another terminal:
cd project
npm run dev
```

Open http://localhost:5173 in your browser!

---

## What It Does

1. **Hive GAN** - Generates synthetic fraud data
2. **Great Convergence** - East vs West AI battle
3. **Safe OOF** - No overfitting evaluation
4. **Results** - F1 ~0.90 (Top 10% on Kaggle!)

---

## Files

| File | Description |
|------|-------------|
| `ultimate_hive_convergence_safe_oof.py` | Main pipeline |
| `preprocess_clean.py` | Data preprocessing |
| `train_hive_gan.py` | Train GANs |
| `app.py` | Streamlit UI |
| `project/` | React Dashboard |

---

## Results(One of the Results)

- **OOF F1**: 0.92-0.93
- **Test F1**: 0.8984
- **Precision**: 0.91
- **Recall**: 0.89
