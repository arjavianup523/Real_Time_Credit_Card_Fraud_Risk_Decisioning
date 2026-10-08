from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "transactions.csv"
)

PROCESSED_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_dataset.csv"
)


# ---------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------

def create_features(df):
    df = df.copy()

    # -----------------------------------------------------
    # TIMESTAMP FEATURES
    # -----------------------------------------------------

    df["transaction_timestamp"] = pd.to_datetime(
        df["transaction_timestamp"]
    )

    df["transaction_day_of_week"] = (
        df["transaction_timestamp"]
        .dt.dayofweek
    )

    df["transaction_day"] = (
        df["transaction_timestamp"]
        .dt.day
    )

    df["transaction_month"] = (
        df["transaction_timestamp"]
        .dt.month
    )

    # -----------------------------------------------------
    # AMOUNT FEATURES
    # -----------------------------------------------------

    df["amount_log"] = (
        (df["amount"] + 1)
        .apply(lambda value: __import__("math").log(value))
    )

    df["amount_vs_customer_average"] = (
        df["amount"]
        / df["avg_amount_last_30d"]
    )

    # -----------------------------------------------------
    # VELOCITY FEATURES
    # -----------------------------------------------------

    df["velocity_ratio"] = (
        df["transactions_last_1h"]
        / (
            df["transactions_last_24h"]
            + 1
        )
    )

    df["high_velocity_flag"] = (
        df["transactions_last_1h"] >= 6
    ).astype(int)

    df["high_daily_activity_flag"] = (
        df["transactions_last_24h"] > 20
    ).astype(int)

    # -----------------------------------------------------
    # CUSTOMER RISK FEATURES
    # -----------------------------------------------------

    df["young_customer_flag"] = (
        df["customer_age"] < 25
    ).astype(int)

    df["new_account_flag"] = (
        df["account_age_days"] < 180
    ).astype(int)

    # -----------------------------------------------------
    # COMBINED RISK FEATURES
    # -----------------------------------------------------

    df["device_location_risk"] = (
        df["is_new_device"]
        + df["is_new_location"]
    )

    df["night_and_new_device"] = (
        df["is_night"]
        * df["is_new_device"]
    )

    df["night_and_new_location"] = (
        df["is_night"]
        * df["is_new_location"]
    )

    df["high_amount_and_new_device"] = (
        (
            df["amount_deviation"] > 3
        ).astype(int)
        * df["is_new_device"]
    )

    # -----------------------------------------------------
    # REMOVE IDENTIFIERS / RAW TIMESTAMP
    # -----------------------------------------------------

    df = df.drop(
        columns=[
            "transaction_id",
            "customer_id",
            "transaction_timestamp",
        ]
    )

    return df


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    print("Loading transaction data...")

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_DATA_PATH}"
        )

    df = pd.read_csv(
        RAW_DATA_PATH
    )

    print(
        f"Original rows: {len(df):,}"
    )

    print(
        f"Original columns: {len(df.columns)}"
    )

    print()
    print("Creating machine-learning features...")

    ml_df = create_features(df)

    ml_df.to_csv(
        PROCESSED_DATA_PATH,
        index=False,
    )

    print()
    print("========================================")
    print("ML DATASET CREATED")
    print("========================================")

    print(
        f"Rows: {len(ml_df):,}"
    )

    print(
        f"Columns: {len(ml_df.columns)}"
    )

    print()
    print("Feature columns:")

    for column in ml_df.columns:
        print(
            f"- {column}"
        )

    print()
    print(
        f"Saved to: {PROCESSED_DATA_PATH}"
    )


if __name__ == "__main__":
    main()