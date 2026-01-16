"""
Data preprocessing utilities for fraud detection
"""
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import RobustScaler


def create_features(df):
    """
    Create engineered features from transaction data
    
    Args:
        df: DataFrame with transaction data
        
    Returns:
        DataFrame with additional features
    """
    df = df.copy()
    
    # Country mismatch feature
    df['country_mismatch'] = (df['country'] != df['bin_country']).astype(int)
    
    # Amount ratio (current transaction vs user average)
    df['amount_ratio'] = df['amount'] / (df['avg_amount_user'] + 1e-5)
    
    # Log transform of amount to reduce skewness
    df['log_amount'] = np.log1p(df['amount'])
    
    # Time-based features
    if 'transaction_time' in df.columns:
        df['transaction_time'] = pd.to_datetime(df['transaction_time'])
        df['hour'] = df['transaction_time'].dt.hour
    else:
        # For single predictions, hour should be provided
        if 'hour' not in df.columns:
            df['hour'] = 12  # Default to noon if not provided
    
    # Cyclical encoding of hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    return df


def clean_data(df):
    """
    Clean and validate transaction data
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    df = df.copy()
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values in categorical columns
    cat_cols = ['country', 'bin_country', 'channel', 'merchant_category']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
    
    # Handle missing values in numerical columns
    num_cols = ['account_age_days', 'total_transactions_user', 'avg_amount_user', 
                'amount', 'shipping_distance_km']
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    return df


def prepare_for_prediction(df, scaler=None, fit=False):
    """
    Prepare data for model prediction
    
    Args:
        df: DataFrame with features
        scaler: RobustScaler instance (optional)
        fit: Whether to fit the scaler (True for training, False for prediction)
        
    Returns:
        Processed DataFrame and scaler
    """
    df = df.copy()
    
    # Define feature columns
    num_cols = ['account_age_days', 'total_transactions_user', 'shipping_distance_km', 
                'amount_ratio', 'log_amount', 'avg_amount_user', 'hour', 'hour_sin', 'hour_cos']
    
    # Initialize scaler if not provided
    if scaler is None:
        scaler = RobustScaler()
    
    # Scale numerical features
    if fit:
        df[num_cols] = scaler.fit_transform(df[num_cols])
    else:
        df[num_cols] = scaler.transform(df[num_cols])
    
    return df, scaler


def preprocess_single_transaction(transaction_dict, scaler=None):
    """
    Preprocess a single transaction for prediction
    
    Args:
        transaction_dict: Dictionary with transaction features
        scaler: Fitted RobustScaler instance
        
    Returns:
        Preprocessed DataFrame ready for prediction
    """
    # Convert to DataFrame
    df = pd.DataFrame([transaction_dict])
    
    # Clean data
    df = clean_data(df)
    
    # Create engineered features
    df = create_features(df)
    
    # Drop columns not needed for prediction
    drop_cols = []
    if 'transaction_id' in df.columns:
        drop_cols.append('transaction_id')
    if 'user_id' in df.columns:
        drop_cols.append('user_id')
    if 'transaction_time' in df.columns:
        drop_cols.append('transaction_time')
    if 'amount' in df.columns:
        drop_cols.append('amount')
    if 'is_fraud' in df.columns:
        drop_cols.append('is_fraud')
    
    if drop_cols:
        df = df.drop(columns=drop_cols, errors='ignore')
    
    # Scale features
    if scaler is not None:
        df, _ = prepare_for_prediction(df, scaler, fit=False)
    
    return df


def preprocess_batch(df, scaler=None):
    """
    Preprocess a batch of transactions for prediction
    
    Args:
        df: DataFrame with multiple transactions
        scaler: Fitted RobustScaler instance
        
    Returns:
        Preprocessed DataFrame ready for prediction
    """
    # Clean data
    df = clean_data(df)
    
    # Create engineered features
    df = create_features(df)
    
    # Drop columns not needed for prediction
    drop_cols = []
    if 'transaction_id' in df.columns:
        drop_cols.append('transaction_id')
    if 'user_id' in df.columns:
        drop_cols.append('user_id')
    if 'transaction_time' in df.columns:
        drop_cols.append('transaction_time')
    if 'amount' in df.columns:
        drop_cols.append('amount')
    if 'is_fraud' in df.columns:
        drop_cols.append('is_fraud')
    
    if drop_cols:
        df = df.drop(columns=drop_cols, errors='ignore')
    
    # Scale features
    if scaler is not None:
        df, _ = prepare_for_prediction(df, scaler, fit=False)
    
    return df


def get_feature_names():
    """
    Get the list of features expected by the model
    
    Returns:
        List of feature names in correct order
    """
    return [
        'account_age_days', 'total_transactions_user', 'avg_amount_user',
        'country', 'bin_country', 'channel', 'merchant_category',
        'promo_used', 'avs_match', 'cvv_result', 'three_ds_flag',
        'shipping_distance_km', 'country_mismatch', 'amount_ratio',
        'log_amount', 'hour', 'hour_sin', 'hour_cos'
    ]


def validate_input(transaction_dict):
    """
    Validate transaction input data
    
    Args:
        transaction_dict: Dictionary with transaction features
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    required_features = [
        'account_age_days', 'total_transactions_user', 'avg_amount_user', 'amount',
        'country', 'bin_country', 'channel', 'merchant_category',
        'promo_used', 'avs_match', 'cvv_result', 'three_ds_flag',
        'shipping_distance_km'
    ]
    
    # Check for missing required features
    missing = [f for f in required_features if f not in transaction_dict]
    if missing:
        return False, f"Missing required features: {', '.join(missing)}"
    
    # Validate numerical ranges
    if transaction_dict.get('account_age_days', 0) < 0:
        return False, "Account age days must be non-negative"
    
    if transaction_dict.get('amount', 0) <= 0:
        return False, "Amount must be positive"
    
    if transaction_dict.get('shipping_distance_km', 0) < 0:
        return False, "Shipping distance must be non-negative"
    
    # Validate binary fields
    binary_fields = ['promo_used', 'avs_match', 'cvv_result', 'three_ds_flag']
    for field in binary_fields:
        if transaction_dict.get(field) not in [0, 1]:
            return False, f"{field} must be 0 or 1"
    
    return True, ""
