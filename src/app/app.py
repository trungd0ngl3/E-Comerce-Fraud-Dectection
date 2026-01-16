"""
Fraud Detection Streamlit App
"""
import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import plotly.express as px
import plotly.graph_objects as go

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
        color: #555;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #f44336;
    }
    .safe-alert {
        background-color: #e8f5e9;
        color: #555;
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
    
    # Sidebar - Navigation
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.radio(
        "Choose page:",
        ["Single Transaction", "Batch Prediction", "📈 Data Dashboard", "⚖️ Model Comparison"]
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
    
    # Display model metrics if available and on prediction pages
    if page in ["Single Transaction", "Batch Prediction"] and metrics and st.sidebar.checkbox("Show Model Performance", value=True):
        st.sidebar.markdown("### 📈 Model Metrics")
        st.sidebar.metric("AUC-ROC", f"{metrics.get('auc_roc', 0):.4f}")
        st.sidebar.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
        st.sidebar.metric("Precision", f"{metrics.get('precision', 0):.4f}")
        st.sidebar.metric("Recall", f"{metrics.get('recall', 0):.4f}")
        st.sidebar.metric("F1-Score", f"{metrics.get('f1_score', 0):.4f}")
    
    # Main content area - routing
    if page == "Single Transaction":
        show_single_transaction_form(model, scaler, selected_model)
    elif page == "Batch Prediction":
        show_batch_prediction(model, scaler, selected_model)
    elif page == "📈 Data Dashboard":
        show_data_dashboard()
    elif page == "⚖️ Model Comparison":
        show_model_comparison()


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


def show_data_dashboard():
    """Display data statistics and analytics dashboard"""
    
    st.markdown("## 📊 Data Statistics Dashboard")
    st.markdown("---")
    
    # Load data
    data_path = "data/transactions.csv"
    if not os.path.exists(data_path):
        st.error(f"⚠️ Data file not found at {data_path}")
        return
    
    try:
        df = pd.read_csv(data_path)
        
        # Key Statistics
        st.markdown("### 📈 Key Statistics")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Transactions", f"{len(df):,}")
        
        with col2:
            fraud_count = (df['is_fraud'] == 1).sum() if 'is_fraud' in df.columns else 0
            fraud_rate = (fraud_count / len(df) * 100) if len(df) > 0 else 0
            st.metric("Fraud Cases", f"{fraud_count:,}", delta=f"{fraud_rate:.2f}%")
        
        with col3:
            if 'amount' in df.columns:
                st.metric("Avg Transaction", f"${df['amount'].mean():.2f}")
            else:
                st.metric("Avg Transaction", "N/A")
        
        with col4:
            if 'amount' in df.columns:
                st.metric("Max Transaction", f"${df['amount'].max():.2f}")
            else:
                st.metric("Max Transaction", "N/A")
        
        with col5:
            if 'account_age_days' in df.columns:
                st.metric("Avg Account Age", f"{df['account_age_days'].mean():.0f} days")
            else:
                st.metric("Avg Account Age", "N/A")
        
        st.markdown("---")
        
        # Data Analysis Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Amount Distribution", "🌍 Geographic Analysis", "⏰ Time Analysis", "📱 Channel & Category"])
        
        with tab1:
            st.markdown("### Transaction Amount Distribution")
            
            col_a1, col_a2 = st.columns(2)
            
            with col_a1:
                # Amount histogram
                if 'amount' in df.columns:
                    fig_amount = px.histogram(
                        df,
                        x='amount',
                        nbins=50,
                        title='Transaction Amount Distribution',
                        labels={'amount': 'Amount ($)', 'count': 'Frequency'},
                        color_discrete_sequence=['#1f77b4']
                    )
                    fig_amount.update_layout(height=400)
                    st.plotly_chart(fig_amount, width='stretch')
            
            with col_a2:
                # Box plot by fraud status
                if 'amount' in df.columns and 'is_fraud' in df.columns:
                    fig_box = px.box(
                        df,
                        x='is_fraud',
                        y='amount',
                        title='Amount Distribution by Fraud Status',
                        labels={'amount': 'Amount ($)', 'is_fraud': 'Fraud Status'},
                        color_discrete_sequence=['#1f77b4']
                    )
                    fig_box.update_layout(height=400, xaxis_title='Is Fraud')
                    st.plotly_chart(fig_box, width='stretch')
            
            # Amount statistics
            if 'amount' in df.columns:
                st.markdown("#### Amount Statistics")
                col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
                
                with col_stats1:
                    st.metric("Mean", f"${df['amount'].mean():.2f}")
                with col_stats2:
                    st.metric("Median", f"${df['amount'].median():.2f}")
                with col_stats3:
                    st.metric("Std Dev", f"${df['amount'].std():.2f}")
                with col_stats4:
                    st.metric("Min - Max", f"${df['amount'].min():.2f} - ${df['amount'].max():.2f}")
        
        with tab2:
            st.markdown("### Geographic Analysis")
            
            col_g1, col_g2 = st.columns(2)
            
            with col_g1:
                # Country distribution
                if 'country' in df.columns:
                    country_counts = df['country'].value_counts()
                    fig_country = px.bar(
                        x=country_counts.index,
                        y=country_counts.values,
                        title='Transactions by Country',
                        labels={'x': 'Country', 'y': 'Count'},
                        color_discrete_sequence=['#1f77b4']
                    )
                    fig_country.update_layout(height=400)
                    st.plotly_chart(fig_country, width='stretch')
            
            with col_g2:
                # BIN Country distribution
                if 'bin_country' in df.columns:
                    bin_country_counts = df['bin_country'].value_counts()
                    fig_bin = px.pie(
                        values=bin_country_counts.values,
                        names=bin_country_counts.index,
                        title='Card BIN Countries',
                        height=400
                    )
                    st.plotly_chart(fig_bin, width='stretch')
            
            # Country fraud rate
            if 'country' in df.columns and 'is_fraud' in df.columns:
                st.markdown("#### Fraud Rate by Country")
                fraud_by_country = df.groupby('country')['is_fraud'].agg(['sum', 'count'])
                fraud_by_country['fraud_rate'] = (fraud_by_country['sum'] / fraud_by_country['count'] * 100)
                fraud_by_country = fraud_by_country.sort_values('fraud_rate', ascending=False)
                
                fig_fraud_country = px.bar(
                    x=fraud_by_country.index,
                    y=fraud_by_country['fraud_rate'],
                    title='Fraud Rate by Country (%)',
                    labels={'x': 'Country', 'y': 'Fraud Rate (%)'},
                    color=fraud_by_country['fraud_rate'],
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig_fraud_country, width='stretch')
        
        with tab3:
            st.markdown("### Time Analysis")
            
            col_t1, col_t2 = st.columns(2)
            
            with col_t1:
                # Hour distribution
                if 'hour' in df.columns:
                    hour_counts = df['hour'].value_counts().sort_index()
                    fig_hour = px.bar(
                        x=hour_counts.index,
                        y=hour_counts.values,
                        title='Transactions by Hour of Day',
                        labels={'x': 'Hour (0-23)', 'y': 'Count'},
                        color_discrete_sequence=['#1f77b4']
                    )
                    fig_hour.update_layout(height=400)
                    st.plotly_chart(fig_hour, width='stretch')
            
            with col_t2:
                # Fraud by hour
                if 'hour' in df.columns and 'is_fraud' in df.columns:
                    fraud_by_hour = df.groupby('hour')['is_fraud'].agg(['sum', 'count'])
                    fraud_by_hour['fraud_rate'] = (fraud_by_hour['sum'] / fraud_by_hour['count'] * 100)
                    
                    fig_fraud_hour = px.line(
                        x=fraud_by_hour.index,
                        y=fraud_by_hour['fraud_rate'],
                        title='Fraud Rate by Hour',
                        labels={'x': 'Hour (0-23)', 'y': 'Fraud Rate (%)'},
                        markers=True
                    )
                    fig_fraud_hour.update_layout(height=400)
                    st.plotly_chart(fig_fraud_hour, width='stretch')
        
        with tab4:
            st.markdown("### Channel & Category Analysis")
            
            col_c1, col_c2 = st.columns(2)
            
            with col_c1:
                # Channel distribution
                if 'channel' in df.columns:
                    channel_counts = df['channel'].value_counts()
                    fig_channel = px.pie(
                        values=channel_counts.values,
                        names=channel_counts.index,
                        title='Transactions by Channel',
                        height=400
                    )
                    st.plotly_chart(fig_channel, width='stretch')
            
            with col_c2:
                # Merchant category distribution
                if 'merchant_category' in df.columns:
                    category_counts = df['merchant_category'].value_counts()
                    fig_category = px.bar(
                        x=category_counts.index,
                        y=category_counts.values,
                        title='Transactions by Merchant Category',
                        labels={'x': 'Category', 'y': 'Count'},
                        color_discrete_sequence=['#1f77b4']
                    )
                    fig_category.update_layout(height=400, xaxis_tickangle=-45)
                    st.plotly_chart(fig_category, width='stretch')
            
            # Fraud rate by channel
            if 'channel' in df.columns and 'is_fraud' in df.columns:
                st.markdown("#### Fraud Rate by Channel")
                fraud_by_channel = df.groupby('channel')['is_fraud'].agg(['sum', 'count'])
                fraud_by_channel['fraud_rate'] = (fraud_by_channel['sum'] / fraud_by_channel['count'] * 100)
                
                fig_fraud_channel = px.bar(
                    x=fraud_by_channel.index,
                    y=fraud_by_channel['fraud_rate'],
                    title='Fraud Rate by Channel (%)',
                    labels={'x': 'Channel', 'y': 'Fraud Rate (%)'},
                    color=fraud_by_channel['fraud_rate'],
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig_fraud_channel, width='stretch')
        
        # Data Details Section
        st.markdown("---")
        st.markdown("### 📋 Data Overview")
        
        if st.checkbox("Show detailed data information"):
            col_info1, col_info2 = st.columns(2)
            
            with col_info1:
                st.markdown("#### Data Types")
                st.write(df.dtypes)
            
            with col_info2:
                st.markdown("#### Missing Values")
                st.write(df.isnull().sum())
            
            st.markdown("#### First 20 Rows")
            st.dataframe(df.head(20), width='stretch')
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.exception(e)


def show_model_comparison():
    """Display model metrics comparison"""
    
    st.markdown("## ⚖️ Model Comparison")
    st.markdown("Compare the performance metrics of different fraud detection models")
    st.markdown("---")
    
    # Load all metrics
    models_info = {
        "Logistic Regression": "models/lr_metrics.json",
        "CatBoost": "models/catboost_metrics.json",
        "TabNet": "models/tabnet_metrics.json"
    }
    
    all_metrics = {}
    available_models = []
    
    for model_name, metrics_path in models_info.items():
        metrics = load_metrics(metrics_path)
        if metrics:
            all_metrics[model_name] = metrics
            available_models.append(model_name)
    
    if not available_models:
        st.error("⚠️ No model metrics found. Please train the models first.")
        return
    
    st.info(f"✅ Found metrics for: {', '.join(available_models)}")
    
    st.markdown("---")
    
    # Metrics Selection
    st.markdown("### 📊 Available Metrics")
    st.info("Select metrics to compare across models")
    
    metrics_list = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
    
    # Create comparison table
    st.markdown("### Detailed Metrics Comparison")
    
    comparison_data = []
    for model_name in available_models:
        row = {'Model': model_name}
        for metric in metrics_list:
            row[metric] = all_metrics[model_name].get(metric, 0)
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.set_index('Model')
    
    # Display metrics table with formatting
    st.dataframe(
        comparison_df.style.format("{:.4f}"),
        width='stretch'
    )
    
    st.markdown("---")
    
    # Visual Comparisons
    st.markdown("### 📈 Visual Comparisons")
    
    # Tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Radar Chart", "Bar Charts", "Heatmap", "Summary"])
    
    with tab1:
        st.markdown("#### Radar Chart Comparison")
        st.markdown("View all metrics for each model in a radar plot")
        
        # Prepare data for radar chart
        fig_radar = go.Figure()
        
        for model_name in available_models:
            values = [all_metrics[model_name].get(metric, 0) for metric in metrics_list]
            values += values[:1]  # Complete the loop for radar chart
            
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics_list + [metrics_list[0]],
                fill='toself',
                name=model_name,
                opacity=0.7
            ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig_radar, width='stretch')
    
    with tab2:
        st.markdown("#### Individual Metric Comparisons")
        
        for metric in metrics_list:
            values = [all_metrics[model_name].get(metric, 0) for model_name in available_models]
            
            fig_bar = px.bar(
                x=available_models,
                y=values,
                title=f"{metric.replace('_', ' ').title()} Comparison",
                labels={'x': 'Model', 'y': metric.replace('_', ' ').title()},
                color=values,
                color_continuous_scale='Viridis',
                height=300
            )
            
            st.plotly_chart(fig_bar, width='stretch')
    
    with tab3:
        st.markdown("#### Heatmap Comparison")
        
        # Normalize for better visualization
        comparison_normalized = comparison_df.copy()
        
        fig_heatmap = px.imshow(
            comparison_normalized,
            labels=dict(x="Metric", y="Model", color="Score"),
            x=comparison_normalized.columns,
            y=comparison_normalized.index,
            color_continuous_scale='RdYlGn',
            aspect='auto',
            text_auto='.4f',
            height=300
        )
        
        st.plotly_chart(fig_heatmap, width='stretch')
    
    with tab4:
        st.markdown("#### Performance Summary")
        
        # Find best model for each metric
        st.markdown("**Best Model by Metric:**")
        
        summary_data = []
        for metric in metrics_list:
            values = {model: all_metrics[model].get(metric, 0) for model in available_models}
            best_model = max(values, key=values.get)
            best_score = values[best_model]
            
            summary_data.append({
                'Metric': metric.replace('_', ' ').title(),
                'Best Model': best_model,
                'Score': f"{best_score:.4f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, width='stretch', hide_index=True)
        
        # Overall ranking
        st.markdown("**Overall Model Ranking:**")
        
        # Calculate weighted score
        weights = {
            'accuracy': 0.2,
            'precision': 0.2,
            'recall': 0.2,
            'f1_score': 0.2,
            'auc_roc': 0.2
        }
        
        scores = {}
        for model_name in available_models:
            score = sum(all_metrics[model_name].get(metric, 0) * weights[metric] for metric in metrics_list)
            scores[model_name] = score
        
        ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        ranking_data = []
        for rank, (model_name, score) in enumerate(ranking, 1):
            ranking_data.append({
                'Rank': rank,
                'Model': model_name,
                'Overall Score': f"{score:.4f}"
            })
        
        ranking_df = pd.DataFrame(ranking_data)
        st.dataframe(ranking_df, width='stretch', hide_index=True)
        
        # Recommendations
        st.markdown("---")
        st.markdown("**Recommendations:**")
        
        best_model = ranking[0][0]
        best_score = ranking[0][1]
        
        col_rec1, col_rec2 = st.columns(2)
        
        with col_rec1:
            st.success(f"✅ **Recommended Model: {best_model}**\n\nOverall Score: {best_score:.4f}")
        
        with col_rec2:
            st.info(f"📊 **Metrics Overview:**\n\n" +
                   "\n".join([f"- {k.replace('_', ' ').title()}: {v:.4f}" 
                             for k, v in all_metrics[best_model].items()]))
    
    st.markdown("---")
    
    # Model Details Expander
    st.markdown("### 🔍 Detailed Model Metrics")
    
    for model_name in available_models:
        with st.expander(f"📋 {model_name} - Full Details"):
            col1, col2, col3 = st.columns(3)
            
            metrics_data = all_metrics[model_name]
            
            with col1:
                st.metric("Accuracy", f"{metrics_data.get('accuracy', 0):.4f}")
                st.metric("F1-Score", f"{metrics_data.get('f1_score', 0):.4f}")
            
            with col2:
                st.metric("Precision", f"{metrics_data.get('precision', 0):.4f}")
                st.metric("AUC-ROC", f"{metrics_data.get('auc_roc', 0):.4f}")
            
            with col3:
                st.metric("Recall", f"{metrics_data.get('recall', 0):.4f}")
            
            st.markdown("#### Metrics Explanation")
            st.write(f"""
            - **Accuracy**: Overall correctness of predictions ({metrics_data.get('accuracy', 0):.1%})
            - **Precision**: Correct fraud predictions / Total fraud predictions ({metrics_data.get('precision', 0):.1%})
            - **Recall**: Fraud cases caught / Total fraud cases ({metrics_data.get('recall', 0):.1%})
            - **F1-Score**: Harmonic mean of precision and recall ({metrics_data.get('f1_score', 0):.4f})
            - **AUC-ROC**: Area under ROC curve - Model's ability to distinguish classes ({metrics_data.get('auc_roc', 0):.4f})
            """)


if __name__ == "__main__":
    main()
