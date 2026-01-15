"""
Fraud Detection Streamlit App
"""
import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add current and parent directory to path to ensure imports work correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.append(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from data.preprocess import (
    preprocess_single_transaction,
    preprocess_batch,
    validate_input,
    create_features
)

try:
    import utils
except ImportError:
    from app import utils

from utils import (
    load_model,
    load_scaler,
    load_metrics,
    predict_single,
    predict_batch,
    create_gauge_chart,
    create_probability_distribution,
    create_confusion_matrix_plot,
    create_roc_curve,
    format_prediction_result
)

# Page configuration
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 3rem;
    }
    .fraud-alert {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #f44336;
    }
    .safe-alert {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)


def main():
    # Title
    st.markdown('<p class="main-header">🔍 E-Commerce Fraud Detection System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Transaction Analysis</p>', unsafe_allow_html=True)
    
    # Sidebar - Model Selection
    st.sidebar.title("⚙️ Settings")
    
    model_type = st.sidebar.selectbox(
        "Select Model",
        ["Logistic Regression", "CatBoost", "TabNet"],
        help="Choose the machine learning model for fraud detection"
    )
    
    model_map = {
        "Logistic Regression": "lr",
        "CatBoost": "catboost",
        "TabNet": "tabnet"
    }
    selected_model = model_map[model_type]
    
    # Sidebar - Prediction Mode
    st.sidebar.title("📊 Prediction Mode")
    mode = st.sidebar.radio(
        "Choose prediction mode:",
        ["Single Transaction", "Batch Prediction (CSV)"]
    )
    
    # Load model and scaler
    model_path = f"models/{selected_model}_model.pkl"
    scaler_path = "models/scaler.pkl"
    metrics_path = f"models/{selected_model}_metrics.json"
    
    # Check if models exist
    if not os.path.exists(model_path):
        st.error(f"⚠️ Model not found at {model_path}. Please train the model first using `train_models.py`")
        st.info("💡 **How to train models:**\n1. Run `python src/models/train_models.py`\n2. This will create trained models in the `models/` directory")
        return
    
    model = load_model(model_path)
    scaler = load_scaler(scaler_path)
    metrics = load_metrics(metrics_path)
    
    # Display model metrics if available
    if metrics and st.sidebar.checkbox("Show Model Performance", value=True):
        st.sidebar.markdown("### 📈 Model Metrics")
        st.sidebar.metric("AUC-ROC", f"{metrics.get('auc_roc', 0):.4f}")
        st.sidebar.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
        st.sidebar.metric("Precision", f"{metrics.get('precision', 0):.4f}")
        st.sidebar.metric("Recall", f"{metrics.get('recall', 0):.4f}")
        st.sidebar.metric("F1-Score", f"{metrics.get('f1_score', 0):.4f}")
    
    # Main content area
    if mode == "Single Transaction":
        show_single_transaction_form(model, scaler, selected_model)
    else:
        show_batch_prediction(model, scaler, selected_model)


def show_single_transaction_form(model, scaler, model_type):
    """Display form for single transaction prediction"""
    
    st.markdown("## 📝 Enter Transaction Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### User Information")
        account_age_days = st.number_input("Account Age (days)", min_value=0, value=365, help="Number of days since account creation")
        total_transactions_user = st.number_input("Total Transactions", min_value=0, value=50, help="Total number of transactions by this user")
        avg_amount_user = st.number_input("Average Transaction Amount ($)", min_value=0.0, value=100.0, help="User's average transaction amount")
    
    with col2:
        st.markdown("### Transaction Details")
        amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=150.0, help="Current transaction amount")
        shipping_distance_km = st.number_input("Shipping Distance (km)", min_value=0.0, value=200.0, help="Distance between billing and shipping address")
        hour = st.slider("Transaction Hour", 0, 23, 12, help="Hour of the day (0-23)")
    
    with col3:
        st.markdown("### Location & Category")
        country = st.selectbox("Country", ["US", "GB", "FR", "DE", "ES", "IT", "NL", "PL", "TR"], help="Transaction country")
        bin_country = st.selectbox("Card BIN Country", ["US", "GB", "FR", "DE", "ES", "IT", "NL", "PL", "TR"], help="Country from card BIN")
        merchant_category = st.selectbox("Merchant Category", ["electronics", "fashion", "travel", "grocery", "gaming"], help="Type of merchant")
        channel = st.selectbox("Channel", ["web", "app", "mobile"], help="Transaction channel")
    
    col4, col5 = st.columns(2)
    
    with col4:
        st.markdown("### Security Flags")
        promo_used = st.selectbox("Promo Code Used", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        avs_match = st.selectbox("AVS Match", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", help="Address Verification System match")
    
    with col5:
        st.markdown("### ")
        st.write("")  # Spacing
        cvv_result = st.selectbox("CVV Verified", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", help="Card Verification Value check")
        three_ds_flag = st.selectbox("3D Secure", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", help="3D Secure authentication")
    
    # Predict button
    st.markdown("---")
    if st.button("🔎 Analyze Transaction", type="primary", width='stretch'):
        # Prepare transaction data
        transaction = {
            'account_age_days': account_age_days,
            'total_transactions_user': total_transactions_user,
            'avg_amount_user': avg_amount_user,
            'amount': amount,
            'country': country,
            'bin_country': bin_country,
            'channel': channel,
            'merchant_category': merchant_category,
            'promo_used': promo_used,
            'avs_match': avs_match,
            'cvv_result': cvv_result,
            'three_ds_flag': three_ds_flag,
            'shipping_distance_km': shipping_distance_km,
            'hour': hour
        }
        
        # Validate input
        is_valid, error_msg = validate_input(transaction)
        if not is_valid:
            st.error(f"❌ Invalid input: {error_msg}")
            return
        
        # Preprocess
        with st.spinner("Processing transaction..."):
            preprocessed_data = preprocess_single_transaction(transaction, scaler)
            
            # Make prediction
            prediction, probability = predict_single(model, preprocessed_data, model_type)
        
        # Display results
        st.markdown("---")
        st.markdown("## 📊 Analysis Results")
        
        result_col1, result_col2 = st.columns([1, 2])
        
        with result_col1:
            # Fraud probability gauge
            fig_gauge = create_gauge_chart(probability)
            st.plotly_chart(fig_gauge, width='stretch')
        
        with result_col2:
            # Prediction result
            result_text = format_prediction_result(prediction, probability)
            if prediction == 1:
                st.markdown(f'<div class="fraud-alert"><h2>{result_text}</h2></div>', unsafe_allow_html=True)
                st.warning(
                    f"**Risk Assessment:**\n\n"
                    f"- Fraud Probability: {probability*100:.2f}%\n"
                    f"- Recommendation: Block or review transaction\n"
                    f"- Suggested Action: Contact customer for verification"
                )
            else:
                st.markdown(f'<div class="safe-alert"><h2>{result_text}</h2></div>', unsafe_allow_html=True)
                st.success(
                    f"**Risk Assessment:**\n\n"
                    f"- Fraud Probability: {probability*100:.2f}%\n"
                    f"- Recommendation: Approve transaction\n"
                    f"- Confidence Level: {(1-probability)*100:.1f}%"
                )
            
            # Transaction summary
            st.markdown("### Transaction Summary")
            st.write(f"💰 Amount: ${amount:.2f}")
            st.write(f"📍 Location: {country}")
            st.write(f"🏪 Merchant: {merchant_category}")
            st.write(f"📱 Channel: {channel}")
            st.write(f"🔐 Security: {'✅' if three_ds_flag == 1 else '❌'} 3DS, {'✅' if cvv_result == 1 else '❌'} CVV")


def show_batch_prediction(model, scaler, model_type):
    """Display batch prediction interface"""
    
    st.markdown("## 📂 Batch Transaction Analysis")
    
    st.info("💡 Upload a CSV file containing multiple transactions for bulk analysis")
    
    # Sample data format
    with st.expander("📋 CSV Format Requirements"):
        st.markdown("""
        Your CSV file should include the following columns:
        
        - `account_age_days`: Number of days since account creation
        - `total_transactions_user`: Total transactions by user
        - `avg_amount_user`: Average transaction amount for user
        - `amount`: Current transaction amount
        - `country`: Transaction country code (e.g., US, GB, FR)
        - `bin_country`: Card BIN country code
        - `channel`: Transaction channel (web, app, mobile)
        - `merchant_category`: Merchant category (electronics, fashion, travel, grocery, gaming)
        - `promo_used`: Promo code used (0 or 1)
        - `avs_match`: AVS match (0 or 1)
        - `cvv_result`: CVV verified (0 or 1)
        - `three_ds_flag`: 3D Secure flag (0 or 1)
        - `shipping_distance_km`: Shipping distance in kilometers
        - `hour`: Transaction hour (0-23) - optional, will use transaction_time if available
        """)
        
        # Sample data
        sample_df = pd.DataFrame({
            'account_age_days': [365, 180, 90],
            'total_transactions_user': [50, 25, 10],
            'avg_amount_user': [100.0, 75.0, 50.0],
            'amount': [150.0, 200.0, 500.0],
            'country': ['US', 'GB', 'FR'],
            'bin_country': ['US', 'GB', 'US'],
            'channel': ['web', 'app', 'web'],
            'merchant_category': ['electronics', 'fashion', 'travel'],
            'promo_used': [0, 1, 0],
            'avs_match': [1, 1, 0],
            'cvv_result': [1, 1, 0],
            'three_ds_flag': [1, 1, 0],
            'shipping_distance_km': [200.0, 150.0, 1000.0],
            'hour': [14, 22, 3]
        })
        
        st.dataframe(sample_df)
        
        # Download sample CSV
        csv = sample_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Sample CSV",
            data=csv,
            file_name="sample_transactions.csv",
            mime="text/csv"
        )
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! Found {len(df)} transactions.")
            
            # Show preview
            with st.expander("👀 Preview Data"):
                st.dataframe(df.head(10))
            
            # Predict button
            if st.button("🚀 Analyze All Transactions", type="primary", width='stretch'):
                with st.spinner("Analyzing transactions..."):
                    # Preprocess batch
                    preprocessed_data = preprocess_batch(df.copy(), scaler)
                    
                    # Make predictions
                    predictions, probabilities = predict_batch(model, preprocessed_data, model_type)
                
                # Add results to dataframe
                results_df = df.copy()
                results_df['fraud_prediction'] = predictions
                results_df['fraud_probability'] = probabilities
                results_df['risk_level'] = results_df['fraud_probability'].apply(
                    lambda x: 'High' if x > 0.75 else 'Medium' if x > 0.5 else 'Low'
                )
                
                # Summary statistics
                st.markdown("## 📊 Analysis Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Transactions", len(results_df))
                
                with col2:
                    fraud_count = (predictions == 1).sum()
                    st.metric("Flagged as Fraud", fraud_count, delta=f"{fraud_count/len(results_df)*100:.1f}%")
                
                with col3:
                    high_risk = (results_df['risk_level'] == 'High').sum()
                    st.metric("High Risk", high_risk)
                
                with col4:
                    avg_prob = probabilities.mean()
                    st.metric("Avg Fraud Probability", f"{avg_prob*100:.2f}%")
                
                # Visualizations
                st.markdown("### 📈 Visualizations")
                
                viz_col1, viz_col2 = st.columns(2)
                
                with viz_col1:
                    # Probability distribution
                    fig_dist = create_probability_distribution(probabilities)
                    st.plotly_chart(fig_dist, width='stretch')
                
                with viz_col2:
                    # Risk level pie chart
                    import plotly.express as px
                    risk_counts = results_df['risk_level'].value_counts()
                    fig_pie = px.pie(
                        values=risk_counts.values,
                        names=risk_counts.index,
                        title='Risk Level Distribution',
                        color=risk_counts.index,
                        color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'}
                    )
                    st.plotly_chart(fig_pie, width='stretch')
                
                # Results table
                st.markdown("### 📋 Detailed Results")
                
                # Filter options
                filter_option = st.selectbox(
                    "Filter results:",
                    ["All Transactions", "Flagged as Fraud Only", "High Risk Only"]
                )
                
                if filter_option == "Flagged as Fraud Only":
                    display_df = results_df[results_df['fraud_prediction'] == 1]
                elif filter_option == "High Risk Only":
                    display_df = results_df[results_df['risk_level'] == 'High']
                else:
                    display_df = results_df
                
                # Highlight fraudulent transactions
                def highlight_fraud(row):
                    if row['fraud_prediction'] == 1:
                        return ['background-color: #ffebee'] * len(row)
                    return [''] * len(row)
                
                st.dataframe(
                    display_df.style.apply(highlight_fraud, axis=1),
                    width='stretch'
                )
                
                # Download results
                csv_results = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results (CSV)",
                    data=csv_results,
                    file_name="fraud_detection_results.csv",
                    mime="text/csv",
                    width='stretch'
                )
                
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.exception(e)


if __name__ == "__main__":
    main()
