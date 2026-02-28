"""
🎯 BOC Framework - Main App
=============================

Wrapper for the complete BOC pipeline:
1. Preprocess data (clean, no leakage)
2. Train Hive GAN (synthetic data generation)
3. Run Great Convergence (model training & evaluation)

Usage:
    python app.py
"""

import streamlit as st
import subprocess
import os
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from plotly.subplots import make_subplots

st.set_page_config(page_title="BOC Framework", page_icon="⚔️", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center; }
    .step-box { padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #1f77b4; margin: 0.5rem 0; }
    .success-box { background-color: #d4edda; border-color: #28a745; }
    .warning-box { background-color: #fff3cd; border-color: #ffc107; }
    .info-box { background-color: #d1ecf1; border-color: #17a2b8; }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">Credit Card Fraud Detection</h1>', unsafe_allow_html=True)
st.markdown("**Domain-Agnostic ML Pipeline with East vs West Model Evolution**")
st.markdown("---")

with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("Pipeline Steps")
    run_preprocess = st.checkbox("Step 1: Preprocess Data", value=False)
    run_hive_gan = st.checkbox("Step 2: Train Hive GAN", value=False)
    run_convergence = st.checkbox("Step 3: Run Great Convergence", value=False)
    
    st.markdown("---")
    
    st.subheader("Hive GAN Settings")
    epochs = st.number_input("GAN Epochs", min_value=50, max_value=1000, value=200)
    batch_size = st.number_input("Batch Size", min_value=32, max_value=512, value=128)
    
    st.markdown("---")
    
    st.subheader("Great Convergence Settings")
    n_folds = st.number_input("OOF Folds", min_value=3, max_value=10, value=5)
    battle_rounds = st.number_input("Battle Rounds", min_value=10, max_value=200, value=50)
    n_models = st.number_input("Models per Side", min_value=5, max_value=50, value=25)

def check_data_exists():
    """Check if preprocessed data exists"""
    data_dir = Path("data/preprocessed_clean")
    if not data_dir.exists():
        return False
    required = ["fraud_train.npy", "normal_train.npy", "fraud_test.npy", "normal_test.npy"]
    return all((data_dir / f).exists() for f in required)

def check_synthetic_exists():
    """Check if synthetic data exists"""
    return Path("outputs/hive_synthetic.npy").exists()

def run_step1_preprocess():
    """Run preprocessing"""
    st.markdown("### Step 1: Data Preprocessing")
    st.info("Running preprocess_clean.py...")
    
    try:
        result = subprocess.run(
            ["python", "preprocess_clean.py"],
            capture_output=True,
            text=True,
            timeout=300
        )
        if result.returncode == 0:
            st.success("✅ Preprocessing complete!")
            return True
        else:
            st.error(f"❌ Error: {result.stderr}")
            return False
    except Exception as e:
        st.error(f"❌ Exception: {e}")
        return False

def run_step2_hive_gan(epochs, batch_size):
    """Train Hive GAN"""
    st.markdown("### Step 2: Hive GAN Training")
    st.info(f"Training for {epochs} epochs...")
    
    try:
        result = subprocess.run(
            ["python", "train_hive_gan.py", "--epochs", str(epochs), "--batch-size", str(batch_size)],
            capture_output=True,
            text=True,
            timeout=3600
        )
        if result.returncode == 0:
            st.success("✅ Hive GAN training complete!")
            return True
        else:
            st.error(f"❌ Error: {result.stderr}")
            return False
    except Exception as e:
        st.error(f"❌ Exception: {e}")
        return False

def run_step3_convergence(n_folds, battle_rounds, n_models):
    """Run Great Convergence"""
    st.markdown("### Step 3: Great Convergence")
    st.info("Running ultimate pipeline...")
    
    try:
        result = subprocess.run(
            ["python", "ultimate_hive_convergence_safe_oof.py"],
            capture_output=True,
            text=True,
            timeout=7200
        )
        if result.returncode == 0:
            st.success("✅ Great Convergence complete!")
            
            # Try to extract F1 score
            output = result.stdout
            if "F1:" in output:
                for line in output.split('\n'):
                    if "F1:" in line:
                        st.markdown(f"**{line}**")
            return True
        else:
            st.error(f"❌ Error: {result.stderr}")
            return False
    except Exception as e:
        st.error(f"❌ Exception: {e}")
        return False

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🚀 Run Pipeline", "📊 Results", "📈 Training", "📁 Files", "❓ Help"])

with tab1:
    st.header("Run Complete Pipeline")
    
    # Status
    st.subheader("Current Status")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if check_data_exists():
            st.success("✅ Data preprocessed")
        else:
            st.warning("⚠️ Data not ready")
    
    with col2:
        if check_synthetic_exists():
            st.success("✅ Hive GAN trained")
        else:
            st.warning("⚠️ Hive GAN not trained")
    
    with col3:
        results_files = list(Path("outputs").glob("ultimate_safe_oof_*.json"))
        if results_files:
            st.success(f"✅ Results ready ({len(results_files)} runs)")
        else:
            st.warning("⚠️ No results yet")
    
    st.markdown("---")
    
    # Run buttons
    if st.button("▶️ Run Selected Steps", type="primary"):
        progress_bar = st.progress(0)
        log_area = st.empty()
        
        logs = []
        
        # Step 1
        if run_preprocess:
            logs.append("=" * 50)
            logs.append("STEP 1: PREPROCESSING")
            logs.append("=" * 50)
            log_area.code("\n".join(logs))
            progress_bar.progress(10)
            
            if not run_step1_preprocess():
                st.error("Step 1 failed!")
                st.stop()
            progress_bar.progress(30)
        
        # Step 2
        if run_hive_gan:
            logs.append("\n" + "=" * 50)
            logs.append("STEP 2: HIVE GAN TRAINING")
            logs.append("=" * 50)
            log_area.code("\n".join(logs))
            
            if not run_step2_hive_gan(epochs, batch_size):
                st.error("Step 2 failed!")
                st.stop()
            progress_bar.progress(60)
        
        # Step 3
        if run_convergence:
            logs.append("\n" + "=" * 50)
            logs.append("STEP 3: GREAT CONVERGENCE")
            logs.append("=" * 50)
            log_area.code("\n".join(logs))
            
            if not run_step3_convergence(n_folds, battle_rounds, n_models):
                st.error("Step 3 failed!")
                st.stop()
            progress_bar.progress(100)
        
        st.success("🎉 Pipeline complete!")
        st.balloons()

with tab2:
    st.header("📊 Results Visualization")
    
    results_files = list(Path("outputs").glob("ultimate_safe_oof_*.json"))
    
    if not results_files:
        st.info("No results yet. Run the pipeline first!")
        st.markdown("""
        ### Quick Demo (sample data)
        """)
        demo_data = {
            "test_f1": 0.8984,
            "test_precision": 0.91,
            "test_recall": 0.89,
            "fold_results": [{"fold": i+1, "f1": 0.9 + np.random.randn()*0.03} for i in range(5)],
            "confusion_matrix": {"tn": 56856, "fp": 8, "fn": 17, "tp": 81}
        }
        selected_result = None
    else:
        # Dropdown to select which run to view
        results_options = {f.name: f for f in sorted(results_files, key=lambda p: p.stat().st_mtime, reverse=True)}
        
        col_selector, col_refresh = st.columns([3, 1])
        
        with col_selector:
            selected_name = st.selectbox(
                "📂 Select Run",
                options=list(results_options.keys()),
                index=0,
                help="Choose which pipeline run to visualize"
            )
        
        with col_refresh:
            st.write("")
            if st.button("🔄 Refresh", help="Scan for new results"):
                st.rerun()
        
        selected_result = results_options[selected_name]
        
        with open(selected_result) as f:
            raw_data = json.load(f)
        
        # Extract results (handle nested "results" key)
        demo_data = {
            'test_f1': raw_data.get('results', {}).get('test_f1', raw_data.get('test_f1', 0)),
            'test_precision': raw_data.get('results', {}).get('test_precision', raw_data.get('test_precision', 0)),
            'test_recall': raw_data.get('results', {}).get('test_recall', raw_data.get('test_recall', 0)),
            'confusion_matrix': raw_data.get('results', {}).get('confusion_matrix', raw_data.get('confusion_matrix', {})),
            'threshold': raw_data.get('threshold', 0),
            'fold_results': raw_data.get('fold_results', []),
            'synthetic': raw_data.get('synthetic', 0),
            'time_minutes': raw_data.get('time_minutes', 0)
        }
        
        st.success(f"Loaded: {selected_name}")
    
    # Show synthetic data info if available
    if demo_data.get('synthetic'):
        st.info(f"🎨 Synthetic frauds used: {demo_data.get('synthetic')}")
    
    # Metrics Overview
    st.subheader("🎯 Test Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("F1 Score", f"{demo_data.get('test_f1', 0):.4f}")
    with col2:
        st.metric("Precision", f"{demo_data.get('test_precision', 0):.4f}")
    with col3:
        st.metric("Recall", f"{demo_data.get('test_recall', 0):.4f}")
    with col4:
        threshold = demo_data.get('threshold', 0)
        st.metric("Threshold", f"{threshold:.4f}")
    
    # Charts
    st.markdown("---")
    
    # 1. Fold F1 Scores
    st.subheader("📈 OOF Fold F1 Scores")
    
    fold_data = demo_data.get('fold_results', [])
    if fold_data:
        folds = [f['fold'] for f in fold_data]
        f1_scores = [f['f1'] for f in fold_data]
        
        fig_folds = px.bar(
            x=folds, 
            y=f1_scores,
            labels={'x': 'Fold', 'y': 'F1 Score'},
            color=f1_scores,
            color_continuous_scale='Viridis',
            title="F1 Score per Fold (OOF Validation)"
        )
        fig_folds.update_layout(yaxis_range=[0.8, 1.0])
        fig_folds.add_hline(y=np.mean(f1_scores), line_dash="dash", annotation_text=f"Mean: {np.mean(f1_scores):.4f}")
        st.plotly_chart(fig_folds, use_container_width=True)
    
    # 2. Confusion Matrix
    st.subheader("🔥 Confusion Matrix")
    
    cm = demo_data.get('confusion_matrix', {})
    if cm:
        fig_cm = px.imshow(
            [[cm.get('tn', 0), cm.get('fp', 0)], 
             [cm.get('fn', 0), cm.get('tp', 0)]],
            text_auto=True,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['Normal', 'Fraud'],
            y=['Normal', 'Fraud'],
            color_continuous_scale='Blues',
            title="Test Set Confusion Matrix"
        )
        st.plotly_chart(fig_cm, use_container_width=True)
    
    # 3. Metrics Radar Chart
    st.subheader("🎯 Metrics Radar")
    
    metrics = ['F1', 'Precision', 'Recall']
    values = [
        demo_data.get('test_f1', 0),
        demo_data.get('test_precision', 0),
        demo_data.get('test_recall', 0)
    ]
    
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=metrics + [metrics[0]],
        fill='toself',
        name='Test Metrics',
        line_color='#1f77b4'
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        title="Performance Metrics Radar"
    )
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # 4. Historical Results
    st.markdown("---")
    st.subheader("📜 Historical Results")
    
    all_results = []
    for rf in sorted(results_files, key=lambda p: p.stat().st_mtime, reverse=True)[:10]:
        with open(rf) as f:
            d = json.load(f)
            all_results.append({
                'file': rf.name,
                'f1': d.get('test_f1', 0),
                'precision': d.get('test_precision', 0),
                'recall': d.get('test_recall', 0),
                'time': d.get('timestamp', '')
            })
    
    if all_results:
        hist_df = px.bar(
            all_results,
            x='file',
            y='f1',
            title="Historical F1 Scores",
            labels={'file': 'Run', 'f1': 'F1 Score'},
            color='f1',
            color_continuous_scale='RdYlGn'
        )
        hist_df.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(hist_df, use_container_width=True)
    
    # Raw JSON
    with st.expander("📄 View Raw JSON"):
        st.json(demo_data)

with tab3:
    st.header("📈 Training Progress")
    
    log_files = list(Path("logs").glob("convergence_*.json"))
    
    if not log_files:
        st.info("No training logs found. Run the pipeline first!")
    else:
        # Dropdown + Refresh
        log_options = {f.name: f for f in sorted(log_files, key=lambda p: p.stat().st_mtime, reverse=True)}
        
        col_selector, col_refresh = st.columns([3, 1])
        
        with col_selector:
            selected_log_name = st.selectbox(
                "📂 Select Training Run",
                options=list(log_options.keys()),
                index=0,
                help="Choose which training run to visualize"
            )
        
        with col_refresh:
            st.write("")
            if st.button("🔄 Refresh", key="log_refresh", help="Scan for new logs"):
                st.rerun()
        
        selected_log = log_options[selected_log_name]
        
        with open(selected_log) as f:
            log_data = json.load(f)
        
        rounds = log_data.get('rounds', [])
        
        if rounds:
            round_nums = [r['round'] for r in rounds]
            f1_scores = [r['f1'] for r in rounds]
            
            # F1 over rounds
            fig_train = px.line(
                x=round_nums,
                y=f1_scores,
                labels={'x': 'Round', 'y': 'F1 Score'},
                title="F1 Score Progression During Training"
            )
            fig_train.add_scatter(
                x=round_nums, 
                y=f1_scores, 
                mode='lines+markers',
                name='F1',
                line=dict(color='#1f77b4', width=2)
            )
            fig_train.add_hline(
                y=np.mean(f1_scores), 
                line_dash="dash", 
                annotation_text=f"Final Mean: {np.mean(f1_scores):.4f}",
                annotation_position="bottom right"
            )
            fig_train.update_layout(yaxis_range=[0.8, 1.0])
            st.plotly_chart(fig_train, use_container_width=True)
            
            # Stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rounds", len(rounds))
            with col2:
                st.metric("Best F1", f"{max(f1_scores):.4f}")
            with col3:
                st.metric("Final F1", f"{f1_scores[-1]:.4f}")
            
            # Raw log data
            with st.expander("📄 View Training Log"):
                st.json(log_data)

with tab4:
    st.header("📁 Files")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data")
        if Path("data/preprocessed_clean").exists():
            for f in Path("data/preprocessed_clean").glob("*.npy"):
                st.write(f"📄 {f.name}")
        else:
            st.info("No preprocessed data")
    
    with col2:
        st.subheader("Outputs")
        if Path("outputs").exists():
            for f in Path("outputs").glob("*"):
                st.write(f"📄 {f.name}")
        else:
            st.info("No outputs")

with tab5:
    st.header("❓ Help")
    st.markdown("""
    ## BOC Framework Pipeline
    
    ### Step 1: Preprocess
    - Cleans and splits data (80/20 train/test)
    - Fits scaler on train only (no leakage)
    - Saves to `data/preprocessed_clean/`
    
    ### Step 2: Train Hive GAN
    - Trains 4 GANs (WGAN, CGAN, Vanilla, CTGAN)
    - Uses Byzantine Consensus for quality
    - Generates synthetic fraud samples
    - Saves to `outputs/hive_synthetic.npy`
    
    ### Step 3: Great Convergence
    - Trains models using East vs West battle system
    - Uses Safe OOF (no overfitting)
    - Evaluates on test set
    - Returns F1, Precision, Recall
    
    ## Quick Start
    
    ```bash
    # Just run the app
    streamlit run app.py
    ```
    
    Or run manually:
    ```bash
    python preprocess_clean.py
    python train_hive_gan.py --epochs 200
    python ultimate_hive_convergence_safe_oof.py
    ```
    """)

if __name__ == "__main__":
    pass
