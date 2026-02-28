"""
Data Loader for Credit Card Fraud Detection Dataset

This module handles loading, preprocessing, and preparing the creditcard.csv dataset for:
1. GAN training (fraud transactions only)
2. ML classifier training (balanced dataset with synthetic fraud)
3. Validation and testing
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import os


class CreditCardDataset(Dataset):
    """PyTorch Dataset for credit card transactions"""
    
    def __init__(self, data, labels=None):
        """
        Args:
            data: numpy array of features
            labels: numpy array of labels (optional for GAN)
        """
        self.data = torch.FloatTensor(data)
        self.labels = torch.FloatTensor(labels) if labels is not None else None
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.data[idx], self.labels[idx]
        return self.data[idx]


def load_creditcard_data(data_path='Dataset/creditcard.csv', test_size=0.2, random_state=42):
    """
    Load and split creditcard.csv into train/test sets
    
    Args:
        data_path: Path to creditcard.csv
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
    
    Returns:
        dict containing:
            - fraud_train: Fraudulent transactions (training)
            - fraud_test: Fraudulent transactions (testing)
            - normal_train: Normal transactions (training)
            - normal_test: Normal transactions (testing)
            - scaler: Fitted StandardScaler
            - feature_names: List of feature column names
    """
    print("Loading creditcard.csv...")
    df = pd.read_csv(data_path)
    
    print(f"Dataset loaded: {df.shape[0]} transactions, {df.shape[1]} features")
    print(f"Fraud cases: {df['Class'].sum()} ({df['Class'].sum() / len(df) * 100:.2f}%)")
    
    # Separate features and labels (keep Time for fuzzy logic)
    X = df.drop('Class', axis=1).values
    y = df['Class'].values
    
    feature_names = df.drop('Class', axis=1).columns.tolist()
    print(f"Using {len(feature_names)} features (Time + V1-V28 + Amount)")
    
    # Split fraud and normal transactions
    fraud_indices = np.where(y == 1)[0]
    normal_indices = np.where(y == 0)[0]
    
    print(f"\nSplitting data (test_size={test_size})...")
    
    # Split fraud transactions
    fraud_train_idx, fraud_test_idx = train_test_split(
        fraud_indices, test_size=test_size, random_state=random_state
    )
    
    # Split normal transactions
    normal_train_idx, normal_test_idx = train_test_split(
        normal_indices, test_size=test_size, random_state=random_state
    )
    
    # Extract fraud and normal data
    fraud_train = X[fraud_train_idx]
    fraud_test = X[fraud_test_idx]
    normal_train = X[normal_train_idx]
    normal_test = X[normal_test_idx]
    
    print(f"Fraud train: {len(fraud_train)}, Fraud test: {len(fraud_test)}")
    print(f"Normal train: {len(normal_train)}, Normal test: {len(normal_test)}")
    
    return {
        'fraud_train': fraud_train,
        'fraud_test': fraud_test,
        'normal_train': normal_train,
        'normal_test': normal_test,
        'feature_names': feature_names,
        'scaler': None  # Will be set after normalization
    }


def load_data(data_path='Dataset/creditcard.csv', test_size=0.2, random_state=42):
    """Wrapper function to match the expected import name"""
    return load_creditcard_data(data_path, test_size, random_state)


def normalize_features(data_dict, fit_on='fraud_train'):
    """
    Normalize features using MinMaxScaler to [-1, 1] range
    (Essential for GANs using Tanh activation)
    
    Args:
        data_dict: Dictionary from load_creditcard_data()
        fit_on: Which dataset to fit the scaler on ('fraud_train' or 'all_train')
    
    Returns:
        Updated data_dict with normalized data and fitted scaler
    """
    print(f"\nNormalizing features (fitting on {fit_on})...")
    
    scaler = MinMaxScaler(feature_range=(-1, 1))
    
    if fit_on == 'fraud_train':
        scaler.fit(data_dict['fraud_train'])
    elif fit_on == 'all_train':
        all_train = np.vstack([data_dict['fraud_train'], data_dict['normal_train']])
        scaler.fit(all_train)
    else:
        raise ValueError("fit_on must be 'fraud_train' or 'all_train'")
    
    # Normalize all datasets
    data_dict['fraud_train'] = scaler.transform(data_dict['fraud_train'])
    data_dict['fraud_test'] = scaler.transform(data_dict['fraud_test'])
    data_dict['normal_train'] = scaler.transform(data_dict['normal_train'])
    data_dict['normal_test'] = scaler.transform(data_dict['normal_test'])
    data_dict['scaler'] = scaler
    
    print("Normalization complete!")
    return data_dict


def get_fraud_dataloader(fraud_data, batch_size=64, shuffle=True):
    """
    Create PyTorch DataLoader for fraud transactions (GAN training)
    
    Args:
        fraud_data: Numpy array of fraud transactions
        batch_size: Batch size for training
        shuffle: Whether to shuffle data
    
    Returns:
        PyTorch DataLoader
    """
    dataset = CreditCardDataset(fraud_data)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=0  # Windows compatibility
    )
    return dataloader


def create_balanced_dataset(fraud_data, normal_data, fraud_synthetic=None, balance_ratio=1.0):
    """
    Create balanced dataset for ML classifier training
    
    Args:
        fraud_data: Real fraud transactions
        normal_data: Normal transactions
        fraud_synthetic: Synthetic fraud from GAN (optional)
        balance_ratio: Ratio of fraud to normal (1.0 = equal, 0.5 = half)
    
    Returns:
        X_balanced, y_balanced: Balanced features and labels
    """
    # Combine real and synthetic fraud
    if fraud_synthetic is not None:
        all_fraud = np.vstack([fraud_data, fraud_synthetic])
    else:
        all_fraud = fraud_data
    
    # Calculate how many normal samples to use
    num_fraud = len(all_fraud)
    num_normal = int(num_fraud / balance_ratio)
    
    # Sample normal transactions
    if num_normal > len(normal_data):
        print(f"Warning: Requested {num_normal} normal samples but only {len(normal_data)} available")
        num_normal = len(normal_data)
    
    normal_sampled = normal_data[np.random.choice(len(normal_data), num_normal, replace=False)]
    
    # Combine and create labels
    X_balanced = np.vstack([all_fraud, normal_sampled])
    y_balanced = np.hstack([np.ones(len(all_fraud)), np.zeros(len(normal_sampled))])
    
    # Shuffle
    shuffle_idx = np.random.permutation(len(X_balanced))
    X_balanced = X_balanced[shuffle_idx]
    y_balanced = y_balanced[shuffle_idx]
    
    print(f"\nBalanced dataset created:")
    print(f"  Fraud: {np.sum(y_balanced == 1)} ({np.sum(y_balanced == 1) / len(y_balanced) * 100:.1f}%)")
    print(f"  Normal: {np.sum(y_balanced == 0)} ({np.sum(y_balanced == 0) / len(y_balanced) * 100:.1f}%)")
    print(f"  Total: {len(X_balanced)} samples")
    
    return X_balanced, y_balanced


def save_preprocessed_data(data_dict, output_dir='data/preprocessed'):
    """Save preprocessed data and scaler for later use"""
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(f'{output_dir}/fraud_train.npy', data_dict['fraud_train'])
    np.save(f'{output_dir}/fraud_test.npy', data_dict['fraud_test'])
    np.save(f'{output_dir}/normal_train.npy', data_dict['normal_train'])
    np.save(f'{output_dir}/normal_test.npy', data_dict['normal_test'])
    
    # Save scaler
    import pickle
    with open(f'{output_dir}/scaler.pkl', 'wb') as f:
        pickle.dump(data_dict['scaler'], f)
    
    print(f"\nPreprocessed data saved to {output_dir}/")


if __name__ == "__main__":
    # Test data loading
    print("="*50)
    print("Testing Data Loader")
    print("="*50)
    
    data = load_creditcard_data()
    data = normalize_features(data, fit_on='all_train')
    
    # Test DataLoader
    fraud_loader = get_fraud_dataloader(data['fraud_train'], batch_size=32)
    print(f"\nDataLoader created with {len(fraud_loader)} batches")
    
    # Test balanced dataset
    X_bal, y_bal = create_balanced_dataset(
        data['fraud_train'], 
        data['normal_train'], 
        balance_ratio=1.0
    )
    
    # Save preprocessed data
    save_preprocessed_data(data)
    
    print("\n Data loader test complete!")
