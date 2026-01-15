import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.linear_model import LogisticRegression

from catboost import CatBoostClassifier

from pytorch_tabnet.tab_model import TabNetClassifier
import torch

# ===================== CONFIG =====================
DATA_PATH = "../../data/transactions.csv"
MODEL_DIR = "../../src/models"
TARGET = "is_fraud"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# ===================== LOAD DATA =====================
df = pd.read_csv(DATA_PATH)

# ===================== FEATURE SET =====================
NUM_COLS = [
    "amount",
    "account_age_days",
    "total_transactions_user",
    "avg_amount_user",
    "shipping_distance_km"
]

CAT_COLS = [
    "channel",
    "merchant_category",
    "country",
    "bin_country",
    "avs_match",
    "cvv_result",
    "three_ds_flag"
]

X = df[NUM_COLS + CAT_COLS]
y = df[TARGET]

# ===================== HANDLE MISSING =====================
X[NUM_COLS] = X[NUM_COLS].fillna(X[NUM_COLS].median())
X[CAT_COLS] = X[CAT_COLS].fillna("Unknown")

# ===================== TRAIN TEST SPLIT =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE
)

# ==========================================================
# 1️⃣ LOGISTIC REGRESSION
# ==========================================================
print("\n🚀 Training Logistic Regression...")

X_train_lr = pd.get_dummies(X_train, columns=CAT_COLS, drop_first=True)
X_test_lr = pd.get_dummies(X_test, columns=CAT_COLS, drop_first=True)

# align columns
X_train_lr, X_test_lr = X_train_lr.align(X_test_lr, join="left", axis=1, fill_value=0)

scaler = StandardScaler()
X_train_lr[NUM_COLS] = scaler.fit_transform(X_train_lr[NUM_COLS])
X_test_lr[NUM_COLS] = scaler.transform(X_test_lr[NUM_COLS])

logistic = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    n_jobs=-1
)

logistic.fit(X_train_lr, y_train)
y_pred = logistic.predict(X_test_lr)
y_proba = logistic.predict_proba(X_test_lr)[:, 1]

print("Logistic Metrics:")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1-score :", f1_score(y_test, y_pred))
print("AUC-ROC  :", roc_auc_score(y_test, y_proba))

with open(f"{MODEL_DIR}/logistic.pkl", "wb") as f:
    pickle.dump(logistic, f)

# ==========================================================
# 2️⃣ CATBOOST
# ==========================================================
print("\n🚀 Training CatBoost...")

cat_features_idx = [X.columns.get_loc(col) for col in CAT_COLS]

catboost = CatBoostClassifier(
    iterations=300,
    depth=6,
    learning_rate=0.1,
    loss_function="Logloss",
    eval_metric="AUC",
    auto_class_weights="Balanced",
    verbose=False
)

catboost.fit(
    X_train,
    y_train,
    cat_features=cat_features_idx
)

y_pred = catboost.predict(X_test)
y_proba = catboost.predict_proba(X_test)[:, 1]

print("CatBoost Metrics:")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1-score :", f1_score(y_test, y_pred))
print("AUC-ROC  :", roc_auc_score(y_test, y_proba))

catboost.save_model(f"{MODEL_DIR}/catboost.pkl")

# ==========================================================
# 3️⃣ TABNET
# ==========================================================
print("\n🚀 Training TabNet...")

# encode categorical to integers
for col in CAT_COLS:
    X[col] = X[col].astype("category").cat.codes

X_train_tab, X_test_tab, y_train_tab, y_test_tab = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE
)

X_train_tab = X_train_tab.values
X_test_tab = X_test_tab.values

tabnet = TabNetClassifier(
    n_d=16,
    n_a=16,
    n_steps=5,
    gamma=1.5,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    mask_type="entmax"
)

tabnet.fit(
    X_train_tab,
    y_train_tab.values,
    eval_set=[(X_test_tab, y_test_tab.values)],
    eval_metric=["auc"],
    max_epochs=30,
    patience=5,
    batch_size=1024,
    virtual_batch_size=128,
    num_workers=0,
    drop_last=False
)

y_pred = tabnet.predict(X_test_tab)
y_proba = tabnet.predict_proba(X_test_tab)[:, 1]

print("TabNet Metrics:")
print("Accuracy :", accuracy_score(y_test_tab, y_pred))
print("Precision:", precision_score(y_test_tab, y_pred))
print("Recall   :", recall_score(y_test_tab, y_pred))
print("F1-score :", f1_score(y_test_tab, y_pred))
print("AUC-ROC  :", roc_auc_score(y_test_tab, y_proba))

with open(f"{MODEL_DIR}/tabnet.pkl", "wb") as f:
    pickle.dump(tabnet, f)

print("\n✅ TRAINING COMPLETE – Models saved to /models")
