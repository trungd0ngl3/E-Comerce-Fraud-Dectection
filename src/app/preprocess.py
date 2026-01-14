import pandas as pd
from sklearn.preprocessing import StandardScaler

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

scaler = StandardScaler()

def preprocess(df, fit=False):
    df = df.copy()

    # fill missing
    df[NUM_COLS] = df[NUM_COLS].fillna(df[NUM_COLS].median())
    df[CAT_COLS] = df[CAT_COLS].fillna("Unknown")

    # one-hot
    df = pd.get_dummies(df, columns=CAT_COLS, drop_first=True)

    if fit:
        df[NUM_COLS] = scaler.fit_transform(df[NUM_COLS])
    else:
        df[NUM_COLS] = scaler.transform(df[NUM_COLS])

    return df
