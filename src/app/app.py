import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from preprocess import preprocess
from utils import load_model, predict

st.set_page_config(page_title="Fraud Detection", layout="wide")


@st.cache_data
def load_data():
    return pd.read_csv("../../data/transactions.csv")

df = load_data()

menu = st.sidebar.radio(
    "📌 Menu",
    [
        "Giới thiệu",
        "Dashboard dữ liệu",
        "So sánh mô hình",
        "Dự đoán gian lận"
    ]
)

# 1
if menu == "Giới thiệu":
    st.title("💳 Phát hiện gian lận thương mại điện tử")

    st.markdown("""
    **Mô hình:** Logistic Regression, CatBoost, TabNet  
    **Tiền xử lý:** Missing, imbalance, encoding, scaling  
    **Metric:** Accuracy, Precision, Recall, F1, AUC-ROC  
    """)

# 2
elif menu == "Dashboard dữ liệu":
    col1, col2, col3 = st.columns(3)
    col1.metric("Tổng giao dịch", len(df))
    col2.metric("Tỷ lệ gian lận (%)", round(df["is_fraud"].mean()*100, 3))
    col3.metric("Số feature", df.shape[1])

    st.subheader("📊 Phân bố gian lận")
    st.bar_chart(df["is_fraud"].value_counts())

    st.subheader("💰 Amount vs Fraud")
    fig, ax = plt.subplots()
    df.boxplot(column="amount", by="is_fraud", ax=ax)
    plt.suptitle("")
    st.pyplot(fig)

# 3, so sanh
elif menu == "So sánh mô hình":
    st.subheader("📈 Hiệu suất các mô hình")

    metrics = pd.DataFrame({
        "Model": ["Logistic Regression", "CatBoost", "TabNet"],
        "Accuracy": [0.94, 0.98, 0.97],
        "Precision": [0.71, 0.92, 0.88],
        "Recall": [0.65, 0.90, 0.86],
        "F1-score": [0.68, 0.91, 0.87],
        "AUC-ROC": [0.91, 0.98, 0.97]
    })

    st.dataframe(metrics)

# 4.
else:
    st.subheader("🔮 Kiểm tra giao dịch")

    model_name = st.selectbox(
        "Chọn mô hình",
        ["logistic", "catboost", "tabnet"]
    )

    model = load_model(model_name)

    option = st.radio("Nhập dữ liệu", ["Nhập tay", "Upload CSV"])

    if option == "Nhập tay":
        amount = st.number_input("Amount", 0.0)
        account_age = st.number_input("Account age (days)", 0)
        total_tx = st.number_input("Total transactions", 0)
        avg_amount = st.number_input("Avg amount", 0.0)
        shipping_distance = st.number_input("Shipping distance (km)", 0.0)

        if st.button("Dự đoán"):
            sample = pd.DataFrame([{
                "amount": amount,
                "account_age_days": account_age,
                "total_transactions_user": total_tx,
                "avg_amount_user": avg_amount,
                "shipping_distance_km": shipping_distance,
                "channel": "web",
                "merchant_category": "electronics",
                "country": "US",
                "bin_country": "US",
                "avs_match": "Y",
                "cvv_result": "M",
                "three_ds_flag": "Y"
            }])

            X = preprocess(sample)
            pred, proba = predict(model, X)

            if pred[0] == 1:
                st.error(f"⚠ Gian lận (prob={proba[0]:.2f})")
            else:
                st.success(f"✅ Hợp lệ (prob={proba[0]:.2f})")

    else:
        file = st.file_uploader("Upload CSV", type=["csv"])
        if file:
            df_test = pd.read_csv(file)
            X = preprocess(df_test)
            pred, proba = predict(model, X)

            df_test["fraud_probability"] = proba
            df_test["prediction"] = pred

            st.dataframe(df_test.head())
            st.success("Dự đoán hoàn tất")
