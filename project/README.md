# BOC Dashboard

## Quick Start

### 1. Start the API (Terminal 1)
```bash
cd project/api
pip install -r requirements.txt
python app.py
```
API runs at: http://localhost:5000

### 2. Start the React App (Terminal 2)
```bash
cd project
npm install
npm run dev
```
App runs at: http://localhost:5173

### 3. Run BOC Pipeline (Terminal 3)
```bash
cd ..
python ultimate_hive_convergence_safe_oof.py
```

The dashboard will auto-detect results from `outputs/` folder!
