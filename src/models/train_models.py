"""
Model training script for fraud detection
"""
import os
import sys
import pickle
import json
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report
)
from catboost import CatBoostClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
import torch
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocess import create_features, clean_data


def load_and_prepare_data(data_path):
    """
    Load and prepare data for training
    
    Args:
        data_path: Path to the CSV file
        
    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    print("Loading data...")
    df = pd.read_csv(data_path)
    
    print(f"Loaded {len(df)} transactions")
    print(f"Fraud cases: {df['is_fraud'].sum()} ({df['is_fraud'].sum()/len(df)*100:.2f}%)")
    
    # Clean data
    print("Cleaning data...")
    df = clean_data(df)
    
    # Create features
    print("Engineering features...")
    df = create_features(df)
    
    # Sort by time
    if 'transaction_time' in df.columns:
        df = df.sort_values(by='transaction_time').reset_index(drop=True)
    
    # Drop unnecessary columns
    drop_cols = ['transaction_id', 'user_id', 'transaction_time', 'amount']
    df = df.drop(columns=drop_cols)
    
    # Separate features and target
    X = df.drop(columns=['is_fraud'])
    y = df['is_fraud']
    
    # Define column types
    num_cols = ['account_age_days', 'total_transactions_user', 'shipping_distance_km', 'amount_ratio', 'log_amount', 'avg_amount_user', 'hour', 'hour_sin', 'hour_cos' ]
    cat_cols = ['country', 'bin_country', 'merchant_category', 'channel']
    
    # Train-test split (time-based)
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    # Scale numerical features
    print("Scaling features...")
    scaler = RobustScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test, scaler, cat_cols


def train_logistic_regression(X_train, X_test, y_train, y_test, cat_cols):
    """Train Logistic Regression model"""
    print("\n" + "="*50)
    print("Training Logistic Regression")
    print("="*50)
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ],
        remainder='passthrough'
    )
    
    # Create pipeline without SMOTE for faster training
    cat_idx = [X_train.columns.get_loc(c) for c in cat_cols]

    lr_pipeline = ImbPipeline(steps=[
        ('smote', SMOTENC(
            categorical_features=cat_idx,
            sampling_strategy=0.1,
            random_state=42
        )),
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, n_jobs=-1))
    ])
    
    # Train
    lr_pipeline.fit(X_train, y_train)
    
    # Predict
    y_pred = lr_pipeline.predict(X_test)
    y_prob = lr_pipeline.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'auc_roc': roc_auc_score(y_test, y_prob)
    }
    
    print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return lr_pipeline, metrics


def train_catboost(X_train, X_test, y_train, y_test, cat_cols):
    """Train CatBoost model"""
    print("\n" + "="*50)
    print("Training CatBoost")
    print("="*50)
    
    # Prepare data for CatBoost
    X_train_cb = X_train.copy()
    X_test_cb = X_test.copy()
    
    for col in cat_cols:
        X_train_cb[col] = X_train_cb[col].fillna("Missing").astype(str)
        X_test_cb[col] = X_test_cb[col].fillna("Missing").astype(str)
    
    # Create model
    cb_model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.02,
        depth=6,
        cat_features=cat_cols,
        scale_pos_weight=25,
        verbose=100,
        eval_metric='AUC'
    )
    
    # Train
    cb_model.fit(X_train_cb, y_train, eval_set=(X_test_cb, y_test))
    
    # Predict
    y_pred = cb_model.predict(X_test_cb)
    y_prob = cb_model.predict_proba(X_test_cb)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'auc_roc': roc_auc_score(y_test, y_prob)
    }
    
    print(f"\nAUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return cb_model, metrics


def train_tabnet(X_train, X_test, y_train, y_test, cat_cols):
    """Train TabNet model"""
    print("\n" + "="*50)
    print("Training TabNet")
    print("="*50)
    
    # Prepare data for TabNet
    X_train_tab = X_train.copy()
    X_test_tab = X_test.copy()

    ord_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

    X_train_tab[cat_cols] = ord_encoder.fit_transform(X_train_tab[cat_cols].astype(str))
    X_test_tab[cat_cols] = ord_encoder.transform(X_test_tab[cat_cols].astype(str))

    for col in cat_cols:
        X_train_tab[col] = X_train_tab[col].astype(int) + 1
        X_test_tab[col] = X_test_tab[col].astype(int) + 1

    cat_idxs = [X_train.columns.get_loc(col) for col in cat_cols]

    cat_dims = []
    for col in cat_cols:
        vocab_size = int(X_train_tab[col].max()) + 1
        cat_dims.append(vocab_size)


    for col in X_train.columns:
        if col not in cat_cols:
            med = X_train_tab[col].median()
            X_train_tab[col] = X_train_tab[col].fillna(med)
            X_test_tab[col] = X_test_tab[col].fillna(med)

    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()

    weight_for_1 = math.sqrt(neg_count / pos_count)

    class_weights = torch.tensor([1.0, weight_for_1], dtype=torch.float32)

    clf_tabnet = TabNetClassifier(
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        cat_emb_dim=20,
        n_d=16,
        n_a=16,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_params={"step_size":10, "gamma":0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        verbose=10
    )

    print('Training TabNet')

    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    clf_tabnet.fit(
        X_train=X_train_tab.values, 
        y_train=y_train.values.astype(int), 
        eval_set=[
            (X_train_tab.values, y_train.values.astype(int)), 
            (X_test_tab.values, y_test.values.astype(int))
        ],
        eval_name=['train', 'valid'],
        eval_metric=['auc'],
        loss_fn=loss_fn,
        max_epochs=200,
        patience=10,
        batch_size=1024, 
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False
    )
    
    # Predict
    y_pred = clf_tabnet.predict(X_test_tab.values)
    y_prob = clf_tabnet.predict_proba(X_test_tab.values)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'auc_roc': roc_auc_score(y_test, y_prob)
    }
    
    print(f"\nAUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return clf_tabnet, metrics


def save_model(model, scaler, metrics, model_name, output_dir='models'):
    """Save model, scaler, and metrics"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, f'{model_name}_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")
    
    # Save scaler (only once)
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    if not os.path.exists(scaler_path):
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Scaler saved to {scaler_path}")
    
    # Save metrics
    metrics_path = os.path.join(output_dir, f'{model_name}_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}")


def main():
    """Main training function"""
    # Configuration
    DATA_PATH = 'data/transactions.csv'
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        print("Please ensure the transactions.csv file is in the data/ directory")
        return
    
    # Load and prepare data
    X_train, X_test, y_train, y_test, scaler, cat_cols = load_and_prepare_data(DATA_PATH)
    
    # Train models
    print("\n" + "="*70)
    print("STARTING MODEL TRAINING")
    print("="*70)
    
    # 1. Logistic Regression
    lr_model, lr_metrics = train_logistic_regression(
        X_train, X_test, y_train, y_test, cat_cols
    )
    save_model(lr_model, scaler, lr_metrics, 'lr')
    
    # 2. CatBoost
    cb_model, cb_metrics = train_catboost(
        X_train, X_test, y_train, y_test, cat_cols
    )
    save_model(cb_model, scaler, cb_metrics, 'catboost')
    
    # 3. TabNet
    tabnet_model, tabnet_metrics = train_tabnet(
        X_train, X_test, y_train, y_test, cat_cols
    )
    save_model(tabnet_model, scaler, tabnet_metrics, 'tabnet')
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE - MODEL COMPARISON")
    print("="*70)
    print(f"{'Model':<20} {'AUC-ROC':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 70)
    print(f"{'Logistic Regression':<20} {lr_metrics['auc_roc']:<12.4f} {lr_metrics['accuracy']:<12.4f} "
          f"{lr_metrics['precision']:<12.4f} {lr_metrics['recall']:<12.4f} {lr_metrics['f1_score']:<12.4f}")
    print(f"{'CatBoost':<20} {cb_metrics['auc_roc']:<12.4f} {cb_metrics['accuracy']:<12.4f} "
          f"{cb_metrics['precision']:<12.4f} {cb_metrics['recall']:<12.4f} {cb_metrics['f1_score']:<12.4f}")
    print(f"{'TabNet':<20} {tabnet_metrics['auc_roc']:<12.4f} {tabnet_metrics['accuracy']:<12.4f} "
          f"{tabnet_metrics['precision']:<12.4f} {tabnet_metrics['recall']:<12.4f} {tabnet_metrics['f1_score']:<12.4f}")
    print("="*70)
    
    print("\n✅ All models trained and saved successfully!")
    print(f"Models saved in: {os.path.abspath('models')}")
    print("\nTo run the Streamlit app: streamlit run src/app/app.py")


if __name__ == "__main__":
    main()
