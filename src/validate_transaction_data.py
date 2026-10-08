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


# ---------------------------------------------------------
# EXPECTED COLUMNS
# ---------------------------------------------------------

REQUIRED_COLUMNS = [
    "transaction_id",
    "customer_id",
    "transaction_timestamp",
    "amount",
    "merchant_category",
    "country",
    "device_type",
    "is_new_device",
    "is_new_location",
    "transactions_last_1h",
    "transactions_last_24h",
    "avg_amount_last_30d",
    "amount_deviation",
    "transaction_hour",
    "is_night",
    "customer_age",
    "account_age_days",
    "fraud",
]


# ---------------------------------------------------------
# VALIDATION FUNCTIONS
# ---------------------------------------------------------

def validate_columns(df):
    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    print("✓ Required columns present")


def validate_duplicates(df):
    duplicate_transactions = df[
        "transaction_id"
    ].duplicated().sum()

    if duplicate_transactions > 0:
        raise ValueError(
            f"Duplicate transaction IDs found: "
            f"{duplicate_transactions}"
        )

    print("✓ No duplicate transaction IDs")


def validate_missing_values(df):
    missing_values = df[
        REQUIRED_COLUMNS
    ].isnull().sum()

    total_missing = missing_values.sum()

    if total_missing > 0:
        print("Missing values found:")
        print(
            missing_values[
                missing_values > 0
            ]
        )
        raise ValueError(
            "Dataset contains missing values"
        )

    print("✓ No missing values")


def validate_numeric_ranges(df):
    if (df["amount"] <= 0).any():
        raise ValueError(
            "Transaction amount must be positive"
        )

    if (
        df["avg_amount_last_30d"] <= 0
    ).any():
        raise ValueError(
            "Average historical amount must be positive"
        )

    if (
        df["transactions_last_1h"] < 0
    ).any():
        raise ValueError(
            "Transactions in last hour cannot be negative"
        )

    if (
        df["transactions_last_24h"] < 0
    ).any():
        raise ValueError(
            "Transactions in last 24 hours cannot be negative"
        )

    if (
        df["customer_age"] < 18
    ).any():
        raise ValueError(
            "Customer age cannot be below 18"
        )

    if (
        df["account_age_days"] <= 0
    ).any():
        raise ValueError(
            "Account age must be positive"
        )

    print("✓ Numeric ranges are valid")


def validate_binary_columns(df):
    binary_columns = [
        "is_new_device",
        "is_new_location",
        "is_night",
        "fraud",
    ]

    for column in binary_columns:
        unique_values = set(
            df[column].unique()
        )

        if not unique_values.issubset(
            {0, 1}
        ):
            raise ValueError(
                f"{column} must contain only 0 and 1"
            )

    print("✓ Binary columns contain only 0/1")


def validate_categorical_columns(df):
    valid_devices = {
        "Mobile",
        "Desktop",
        "Tablet",
    }

    valid_merchants = {
        "Grocery",
        "Electronics",
        "Travel",
        "Fuel",
        "Restaurant",
        "Online Shopping",
        "Entertainment",
        "Healthcare",
    }

    valid_countries = {
        "India",
        "USA",
        "UK",
        "Germany",
        "Singapore",
        "UAE",
    }

    if not set(
        df["device_type"].unique()
    ).issubset(valid_devices):
        raise ValueError(
            "Unexpected device type found"
        )

    if not set(
        df["merchant_category"].unique()
    ).issubset(valid_merchants):
        raise ValueError(
            "Unexpected merchant category found"
        )

    if not set(
        df["country"].unique()
    ).issubset(valid_countries):
        raise ValueError(
            "Unexpected country found"
        )

    print("✓ Categorical values are valid")


def validate_timestamp(df):
    timestamp = pd.to_datetime(
        df["transaction_timestamp"],
        errors="coerce",
    )

    if timestamp.isnull().any():
        raise ValueError(
            "Invalid transaction timestamps found"
        )

    print("✓ Transaction timestamps are valid")


def validate_fraud_rate(df):
    fraud_rate = df["fraud"].mean()

    print(
        f"✓ Fraud rate: {fraud_rate:.2%}"
    )

    if fraud_rate <= 0:
        raise ValueError(
            "No fraudulent transactions found"
        )

    if fraud_rate >= 0.20:
        raise ValueError(
            "Fraud rate is unrealistically high"
        )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    print("Loading transaction dataset...")
    print()

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_DATA_PATH}"
        )

    df = pd.read_csv(
        RAW_DATA_PATH
    )

    print(
        f"Rows loaded: {len(df):,}"
    )
    print()

    print("Running validation checks...")
    print()

    validate_columns(df)
    validate_duplicates(df)
    validate_missing_values(df)
    validate_numeric_ranges(df)
    validate_binary_columns(df)
    validate_categorical_columns(df)
    validate_timestamp(df)
    validate_fraud_rate(df)

    print()
    print(
        "========================================"
    )
    print(
        "DATA VALIDATION PASSED"
    )
    print(
        "========================================"
    )


if __name__ == "__main__":
    main()