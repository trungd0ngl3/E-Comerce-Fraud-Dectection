"""
Utility functions for the Streamlit fraud detection app
"""
import json
import pickle
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import confusion_matrix, roc_curve, auc


def load_model(model_path):
    """
    Load a saved model
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        Loaded model
    """
    if not os.path.exists(model_path):
        return None
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def load_scaler(scaler_path):
    """
    Load a saved scaler
    
    Args:
        scaler_path: Path to the saved scaler file
        
    Returns:
        Loaded scaler
    """
    if not os.path.exists(scaler_path):
        return None
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return scaler


def load_metrics(metrics_path):
    """
    Load saved model metrics
    
    Args:
        metrics_path: Path to the metrics JSON file
        
    Returns:
        Dictionary of metrics
    """
    if not os.path.exists(metrics_path):
        return None
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    return metrics


def prepare_data_for_model(data, model_type='lr'):
    """
    Prepare data for specific model type
    
    Args:
        data: Preprocessed DataFrame
        model_type: Type of model ('lr', 'catboost', 'tabnet')
        
    Returns:
        Prepared data for the model
    """
    cat_cols = ['country', 'bin_country', 'merchant_category', 'channel']
    
    if model_type == 'catboost':
        # CatBoost handles categorical features natively, but it expects strings (not objects)
        data_prepared = data.copy()
        for col in cat_cols:
            if col in data_prepared.columns:
                data_prepared[col] = data_prepared[col].fillna("Missing").astype(str)
        return data_prepared
    
    elif model_type == 'tabnet':
        # TabNet needs integer encoding for categorical features and homogeneous numeric dtype
        data_prepared = data.copy()
        
        # Define encoding maps for categorical features as used during training
        # These are standard from the training script logic
        encoding_maps = {
            'country': {'US': 0, 'GB': 1, 'FR': 2, 'DE': 3, 'ES': 4, 'IT': 5, 'NL': 6, 'PL': 7, 'TR': 8},
            'bin_country': {'US': 0, 'GB': 1, 'FR': 2, 'DE': 3, 'ES': 4, 'IT': 5, 'NL': 6, 'PL': 7, 'TR': 8},
            'channel': {'web': 0, 'app': 1, 'mobile': 2},
            'merchant_category': {'electronics': 0, 'fashion': 1, 'travel': 2, 'grocery': 3, 'gaming': 4}
        }
        
        for col in cat_cols:
            if col in data_prepared.columns:
                # Map categorical values to integers, unknown values get -1
                data_prepared[col] = data_prepared[col].map(encoding_maps[col]).fillna(-1).astype(int) + 1
        
        # Ensure all columns are numeric - convert any remaining object columns to float/int
        for col in data_prepared.columns:
            if data_prepared[col].dtype == 'object':
                data_prepared[col] = pd.to_numeric(data_prepared[col], errors='coerce').fillna(0)
        
        # Final conversion to float32 for TabNet compatibility (avoid object dtypes)
        data_prepared = data_prepared.astype(np.float32)
        
        return data_prepared
    
    else:  # Logistic Regression
        # Logistic Regression uses the scikit-learn pipeline which handles categorical encoding (OneHot) internally
        return data


def predict_single(model, data, model_type='lr'):
    """
    Make prediction for a single transaction
    
    Args:
        model: Trained model
        data: Preprocessed DataFrame
        model_type: Type of model ('lr', 'catboost', 'tabnet')
        
    Returns:
        Tuple of (prediction, probability)
    """
    # Prepare data for specific model type
    data_prepared = prepare_data_for_model(data, model_type)
    
    if model_type == 'tabnet':
        # Convert to numpy array with proper dtype for PyTorch
        data_array = data_prepared.values.astype(np.float32)
        pred = model.predict(data_array)
        prob = model.predict_proba(data_array)[:, 1]
    else:
        pred = model.predict(data_prepared)
        prob = model.predict_proba(data_prepared)[:, 1]
    
    return int(pred[0]), float(prob[0])


def predict_batch(model, data, model_type='lr'):
    """
    Make predictions for multiple transactions
    
    Args:
        model: Trained model
        data: Preprocessed DataFrame
        model_type: Type of model ('lr', 'catboost', 'tabnet')
        
    Returns:
        Tuple of (predictions array, probabilities array)
    """
    # Prepare data for specific model type
    data_prepared = prepare_data_for_model(data, model_type)
    
    if model_type == 'tabnet':
        # Convert to numpy array with proper dtype
        data_array = data_prepared.values.astype(np.float32)
        preds = model.predict(data_array)
        probs = model.predict_proba(data_array)[:, 1]
    else:
        preds = model.predict(data_prepared)
        probs = model.predict_proba(data_prepared)[:, 1]
    
    return preds, probs


def create_gauge_chart(probability):
    """
    Create a gauge chart for fraud probability
    
    Args:
        probability: Fraud probability (0-1)
        
    Returns:
        Plotly figure
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Fraud Probability (%)"},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkred" if probability > 0.5 else "darkgreen"},
            'steps': [
                {'range': [0, 25], 'color': "lightgreen"},
                {'range': [25, 50], 'color': "yellow"},
                {'range': [50, 75], 'color': "orange"},
                {'range': [75, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': 75
            }
        }
    ))
    
    fig.update_layout(height=350, margin=dict(t=50, b=0, l=0, r=0))
    return fig


def create_probability_distribution(probabilities):
    """
    Create histogram of fraud probabilities
    
    Args:
        probabilities: Array of fraud probabilities
        
    Returns:
        Plotly figure
    """
    fig = px.histogram(
        x=probabilities,
        nbins=50,
        labels={'x': 'Fraud Probability', 'y': 'Count'},
        title='Distribution of Fraud Probabilities',
        color_discrete_sequence=['#1f77b4']
    )
    
    fig.update_layout(
        xaxis_title="Fraud Probability",
        yaxis_title="Number of Transactions",
        showlegend=False,
        bargap=0.05
    )
    
    return fig


def create_confusion_matrix_plot(y_true, y_pred):
    """
    Create confusion matrix heatmap
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Plotly figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Legitimate', 'Fraud'],
        y=['Legitimate', 'Fraud'],
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 16}
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        height=400
    )
    
    return fig


def create_roc_curve(y_true, y_prob):
    """
    Create ROC curve plot
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        
    Returns:
        Plotly figure
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {roc_auc:.4f})',
        line=dict(color='darkorange', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='navy', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title='Receiver Operating Characteristic (ROC) Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        showlegend=True,
        height=500
    )
    
    return fig


def format_prediction_result(prediction, probability):
    """
    Format prediction result for display
    
    Args:
        prediction: Binary prediction (0 or 1)
        probability: Fraud probability
        
    Returns:
        Formatted string with emoji and color
    """
    if prediction == 1:
        if probability > 0.90:
            risk_level = "CRITICAL RISK"
            emoji = "🚨"
        elif probability > 0.75:
            risk_level = "HIGH RISK"
            emoji = "🛑"
        else:
            risk_level = "MEDIUM RISK"
            emoji = "⚠️"
        return f"{emoji} **FRAUDULENT** ({risk_level})"
    else:
        confidence = (1 - probability) * 100
        if confidence > 95:
            emoji = "✅"
            level = "SECURE"
        else:
            emoji = "🟢"
            level = "LOW RISK"
        return f"{emoji} **LEGITIMATE** ({confidence:.1f}% confidence)"


def get_feature_importance_mock(model_type):
    """
    Get mock feature importance (placeholder for actual implementation)
    
    Args:
        model_type: Type of model
        
    Returns:
        Dictionary of feature importances
    """
    # This is a placeholder for actual feature importance extraction
    features = {
        'amount_ratio': 0.18,
        'shipping_distance_km': 0.14,
        'country_mismatch': 0.12,
        'three_ds_flag': 0.10,
        'cvv_result': 0.08,
        'avs_match': 0.07,
        'account_age_days': 0.07,
        'log_amount': 0.06,
        'total_transactions_user': 0.06,
        'hour': 0.05,
        'merchant_category': 0.04,
        'channel': 0.03
    }
    return features


def load_all_model_metrics(models_dir='models'):
    """
    Load metrics for all available models
    
    Args:
        models_dir: Directory containing model metrics files
        
    Returns:
        Dictionary with model names as keys and their metrics as values
    """
    metrics_files = {
        'Logistic Regression': 'lr_metrics.json',
        'CatBoost': 'catboost_metrics.json',
        'TabNet': 'tabnet_metrics.json'
    }
    
    all_metrics = {}
    for model_name, filename in metrics_files.items():
        filepath = os.path.join(models_dir, filename)
        metrics = load_metrics(filepath)
        if metrics:
            all_metrics[model_name] = metrics
    
    return all_metrics


def create_model_comparison_radar(all_metrics, metrics_list=None):
    """
    Create radar chart for model comparison
    
    Args:
        all_metrics: Dictionary of model metrics
        metrics_list: List of metrics to compare
        
    Returns:
        Plotly figure
    """
    if metrics_list is None:
        metrics_list = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
    
    fig = go.Figure()
    
    for model_name, metrics in all_metrics.items():
        values = [metrics.get(metric, 0) for metric in metrics_list]
        values += values[:1]  # Complete the loop for radar chart
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics_list + [metrics_list[0]],
            fill='toself',
            name=model_name,
            opacity=0.7
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        height=500,
        showlegend=True,
        title='Model Performance Comparison'
    )
    
    return fig


def create_metrics_heatmap(comparison_df):
    """
    Create heatmap for metrics comparison
    
    Args:
        comparison_df: DataFrame with models and metrics
        
    Returns:
        Plotly figure
    """
    fig = px.imshow(
        comparison_df,
        labels=dict(x="Metric", y="Model", color="Score"),
        x=comparison_df.columns,
        y=comparison_df.index,
        color_continuous_scale='RdYlGn',
        aspect='auto',
        text_auto='.4f',
        height=400
    )
    
    fig.update_layout(title='Model Metrics Heatmap')
    
    return fig


def get_model_rankings(all_metrics, weights=None):
    """
    Calculate overall model rankings based on metrics
    
    Args:
        all_metrics: Dictionary of model metrics
        weights: Dictionary of metric weights (default equal weights)
        
    Returns:
        List of tuples (model_name, score) sorted by score descending
    """
    if weights is None:
        weights = {
            'accuracy': 0.2,
            'precision': 0.2,
            'recall': 0.2,
            'f1_score': 0.2,
            'auc_roc': 0.2
        }
    
    scores = {}
    metrics_list = list(weights.keys())
    
    for model_name, metrics in all_metrics.items():
        score = sum(metrics.get(metric, 0) * weights[metric] for metric in metrics_list)
        scores[model_name] = score
    
    ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranking
