# E-Commerce Fraud Detection System

A comprehensive fraud detection system built with Streamlit, featuring multiple ML models (Logistic Regression, CatBoost, TabNet) for detecting fraudulent e-commerce transactions.

## Features

- 🤖 **Multiple Models**: Choose between Logistic Regression, CatBoost, and TabNet
- 📊 **Single Transaction Analysis**: Input transaction details manually for instant fraud detection
- 📂 **Batch Processing**: Upload CSV files for bulk transaction analysis
- 📈 **Rich Visualizations**: Gauge charts, probability distributions, confusion matrices, ROC curves
- 🎯 **Real-time Predictions**: Get instant fraud probability and risk assessments
- 💾 **Export Results**: Download analysis results as CSV files

## Project Structure

```
project/
├── data/
│   └── transactions.csv          # Training data
├── models/                        # Trained models (created after training)
│   ├── lr_model.pkl
│   ├── catboost_model.pkl
│   ├── tabnet_model.pkl
│   ├── scaler.pkl
│   └── *_metrics.json
├── src/
│   ├── app/
│   │   ├── app.py                # Streamlit application
│   │   └── utils.py              # Utility functions
│   ├── data/
│   │   └── preprocess.py         # Data preprocessing
│   └── models/
│       └── train_models.py       # Model training script
└── requirements.txt
```

## Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Train Models

Before using the app, you need to train the models:

```bash
python src/models/train_models.py
```

This will:
- Load and preprocess the transaction data
- Train three different models (Logistic Regression, CatBoost, TabNet)
- Save trained models, scaler, and metrics to the `models/` directory
- Display performance comparison

**Expected output:**
```
Training set: 239,756 samples
Test set: 59,939 samples

Training Logistic Regression...
AUC-ROC: 0.9490
...

Training complete! Models saved in models/
```

### Step 2: Run Streamlit App

```bash
streamlit run src/app/app.py
```

The app will open in your browser at `http://localhost:8501`

## Using the App

### Single Transaction Prediction

1. Select a model from the sidebar
2. Choose "Single Transaction" mode
3. Fill in transaction details:
   - User Information (account age, transaction history, average amount)
   - Transaction Details (amount, shipping distance, hour)
   - Location & Category (country, merchant category, channel)
   - Security Flags (promo code, AVS, CVV, 3D Secure)
4. Click "Analyze Transaction"
5. View results:
   - Fraud probability gauge
   - Risk assessment
   - Transaction summary

### Batch Prediction

1. Select a model from the sidebar
2. Choose "Batch Prediction (CSV)" mode
3. Download the sample CSV template (optional)
4. Upload your CSV file with transaction data
5. Click "Analyze All Transactions"
6. View:
   - Summary statistics
   - Probability distribution
   - Risk level breakdown
   - Detailed results table
7. Download results as CSV

## CSV Format for Batch Prediction

Required columns:
- `account_age_days`: Days since account creation
- `total_transactions_user`: Total user transactions
- `avg_amount_user`: User's average transaction amount
- `amount`: Current transaction amount
- `country`: Transaction country (US, GB, FR, etc.)
- `bin_country`: Card BIN country
- `channel`: Transaction channel (web, app, mobile)
- `merchant_category`: Merchant type (electronics, fashion, travel, grocery, gaming)
- `promo_used`: Promo code used (0 or 1)
- `avs_match`: Address verification (0 or 1)
- `cvv_result`: CVV verified (0 or 1)
- `three_ds_flag`: 3D Secure flag (0 or 1)
- `shipping_distance_km`: Shipping distance
- `hour`: Transaction hour (0-23, optional if transaction_time provided)

## Model Performance

Based on the fraud detection notebook, expected performance:

| Model | AUC-ROC | Accuracy | Precision | Recall | F1-Score |
|-------|---------|----------|-----------|--------|----------|
| Logistic Regression | 0.949 | 0.96 | 0.30 | 0.75 | 0.43 |
| CatBoost | 0.977 | 0.96 | 0.33 | 0.89 | 0.48 |
| TabNet | 0.961 | 0.91 | 0.18 | 0.90 | 0.30 |

## Feature Engineering

The system automatically creates enhanced features:
- **country_mismatch**: Billing vs shipping country difference
- **amount_ratio**: Current amount vs user average
- **log_amount**: Log-transformed transaction amount
- **hour_sin/hour_cos**: Cyclical encoding of transaction hour

## Technologies Used

- **Streamlit**: Interactive web interface
- **Scikit-learn**: Logistic Regression, preprocessing
- **CatBoost**: Gradient boosting model
- **PyTorch TabNet**: Deep learning model for tabular data
- **Plotly**: Interactive visualizations
- **Pandas/NumPy**: Data processing
- **imbalanced-learn**: Handling class imbalance

## Troubleshooting

**Models not found error:**
- Make sure you've run `python src/models/train_models.py` first
- Check that `models/` directory contains the model files

**Import errors:**
- Verify all dependencies are installed: `pip install -r requirements.txt`

**Data file not found:**
- Ensure `data/transactions.csv` exists in the project directory

## License

This project is for educational and demonstration purposes.
