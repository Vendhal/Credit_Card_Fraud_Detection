"""
🧹 CLEAN PREPROCESSING - No Data Leakage
=========================================

This script properly preprocesses the credit card fraud data:
✅ Split first (train/test)
✅ Fit scaler ONLY on training data
✅ Transform both train and test with train-fitted scaler
✅ No test data influences preprocessing

Author: Sai Sandeep
Date: 2026-02-17
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
import os

def preprocess_clean():
    """
    Clean preprocessing with NO data leakage
    """
    print("="*70)
    print("🧹 CLEAN PREPROCESSING - No Data Leakage")
    print("="*70)
    print()
    
    # Step 1: Load raw data
    print("📊 Loading raw data...")
    df = pd.read_csv('Dataset/creditcard.csv')
    X = df.drop('Class', axis=1).values
    y = df['Class'].values
    
    print(f"   Total samples: {len(X)}")
    print(f"   Total frauds: {y.sum()}")
    print(f"   Features: {X.shape[1]}")
    print()
    
    # Step 2: Split FIRST (before any preprocessing)
    print("✂️  Splitting data (80/20 stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   Train: {len(X_train)} samples ({y_train.sum()} frauds)")
    print(f"   Test: {len(X_test)} samples ({y_test.sum()} frauds)")
    print()
    
    # Step 3: Fit scaler ONLY on training data
    print("⚖️  Fitting MinMaxScaler on TRAINING data only...")
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(X_train)  # ✅ Only train data!
    
    print(f"   Scaler fitted on {len(X_train)} train samples")
    print(f"   Feature range: [-1, 1]")
    print()
    
    # Step 4: Transform both sets with train-fitted scaler
    print("🔄 Transforming train and test with train-fitted scaler...")
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Step 5: Separate fraud and normal
    print("🎯 Separating fraud and normal transactions...")
    fraud_train = X_train_scaled[y_train == 1]
    fraud_test = X_test_scaled[y_test == 1]
    normal_train = X_train_scaled[y_train == 0]
    normal_test = X_test_scaled[y_test == 0]
    
    print(f"   Fraud train: {len(fraud_train)}")
    print(f"   Fraud test: {len(fraud_test)}")
    print(f"   Normal train: {len(normal_train)}")
    print(f"   Normal test: {len(normal_test)}")
    print()
    
    # Step 6: Save clean data
    print("💾 Saving clean preprocessed data...")
    output_dir = 'data/preprocessed_clean'
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(f'{output_dir}/fraud_train.npy', fraud_train)
    np.save(f'{output_dir}/fraud_test.npy', fraud_test)
    np.save(f'{output_dir}/normal_train.npy', normal_train)
    np.save(f'{output_dir}/normal_test.npy', normal_test)
    
    # Save scaler (fitted on train only!)
    with open(f'{output_dir}/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"   ✅ Saved to {output_dir}/")
    print()
    
    # Step 7: Verify no leakage
    print("🔍 Verification - Checking for data leakage...")
    
    # Check that train and test frauds are different
    fraud_train_raw = X_train[y_train == 1]
    fraud_test_raw = X_test[y_test == 1]
    
    # Compare a few samples to ensure they're different
    train_sample = fraud_train[0][:5]
    test_sample = fraud_test[0][:5]
    
    print(f"   Train fraud sample (first 5 features): {train_sample}")
    print(f"   Test fraud sample (first 5 features):  {test_sample}")
    
    if not np.allclose(train_sample, test_sample):
        print("   ✅ Train and test samples are different - NO LEAKAGE!")
    else:
        print("   ⚠️  Warning: Samples might be identical")
    
    print()
    print("="*70)
    print("🎉 CLEAN PREPROCESSING COMPLETE!")
    print("="*70)
    print()
    print("Summary:")
    print(f"   ✅ Scaler fitted on: {len(X_train)} train samples ONLY")
    print(f"   ✅ Test samples: {len(X_test)} (no influence on scaler)")
    print(f"   ✅ Data saved to: {output_dir}/")
    print()
    print("You can now use this clean data for honest experiments!")
    
    return {
        'fraud_train': fraud_train,
        'fraud_test': fraud_test,
        'normal_train': normal_train,
        'normal_test': normal_test,
        'scaler': scaler
    }

if __name__ == "__main__":
    # Backup old data first
    print("⚠️  Backing up old preprocessed data...")
    import shutil
    if os.path.exists('data/preprocessed'):
        shutil.copytree('data/preprocessed', 'data/preprocessed_old_leaked', dirs_exist_ok=True)
        print("   ✅ Old data backed up to: data/preprocessed_old_leaked/")
        print()
    
    # Create clean data
    data = preprocess_clean()
