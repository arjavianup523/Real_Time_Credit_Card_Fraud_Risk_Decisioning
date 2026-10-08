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

PROCESSED_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

PROCESSED_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------
# MAIN ANALYSIS
# ---------------------------------------------------------

def main():
    print("Loading transaction data...")

    df = pd.read_csv(
        RAW_DATA_PATH
    )

    df["transaction_timestamp"] = pd.to_datetime(
        df["transaction_timestamp"]
    )

    # -----------------------------------------------------
    # OVERALL SUMMARY
    # -----------------------------------------------------

    total_transactions = len(df)

    fraud_transactions = int(
        df["fraud"].sum()
    )

    legitimate_transactions = (
        total_transactions
        - fraud_transactions
    )

    fraud_rate = (
        fraud_transactions
        / total_transactions
    )

    total_transaction_value = (
        df["amount"].sum()
    )

    fraud_transaction_value = (
        df.loc[
            df["fraud"] == 1,
            "amount",
        ].sum()
    )

    legitimate_transaction_value = (
        df.loc[
            df["fraud"] == 0,
            "amount",
        ].sum()
    )

    print()
    print("========================================")
    print("OVERALL FRAUD SUMMARY")
    print("========================================")
    print(
        f"Total transactions: "
        f"{total_transactions:,}"
    )
    print(
        f"Legitimate transactions: "
        f"{legitimate_transactions:,}"
    )
    print(
        f"Fraud transactions: "
        f"{fraud_transactions:,}"
    )
    print(
        f"Fraud rate: "
        f"{fraud_rate:.2%}"
    )
    print(
        f"Total transaction value: "
        f"₹{total_transaction_value:,.2f}"
    )
    print(
        f"Fraud transaction value: "
        f"₹{fraud_transaction_value:,.2f}"
    )
    print(
        f"Legitimate transaction value: "
        f"₹{legitimate_transaction_value:,.2f}"
    )

    # -----------------------------------------------------
    # FRAUD VS LEGITIMATE AMOUNT
    # -----------------------------------------------------

    amount_summary = (
        df.groupby("fraud")["amount"]
        .agg(
            [
                "count",
                "mean",
                "median",
                "min",
                "max",
            ]
        )
        .reset_index()
    )

    amount_summary["transaction_type"] = (
        amount_summary["fraud"]
        .map(
            {
                0: "Legitimate",
                1: "Fraud",
            }
        )
    )

    amount_summary = amount_summary[
        [
            "transaction_type",
            "count",
            "mean",
            "median",
            "min",
            "max",
        ]
    ]

    amount_summary.to_csv(
        PROCESSED_DATA_DIR
        / "fraud_amount_summary.csv",
        index=False,
    )

    print()
    print("Fraud vs legitimate amount:")
    print(amount_summary.to_string(index=False))

    # -----------------------------------------------------
    # FRAUD BY DEVICE
    # -----------------------------------------------------

    device_summary = (
        df.groupby("device_type")
        .agg(
            transactions=(
                "transaction_id",
                "count",
            ),
            fraud_transactions=(
                "fraud",
                "sum",
            ),
            fraud_rate=(
                "fraud",
                "mean",
            ),
        )
        .reset_index()
    )

    device_summary.to_csv(
        PROCESSED_DATA_DIR
        / "fraud_by_device.csv",
        index=False,
    )

    print()
    print("Fraud by device:")
    print(device_summary.to_string(index=False))

    # -----------------------------------------------------
    # FRAUD BY COUNTRY
    # -----------------------------------------------------

    country_summary = (
        df.groupby("country")
        .agg(
            transactions=(
                "transaction_id",
                "count",
            ),
            fraud_transactions=(
                "fraud",
                "sum",
            ),
            fraud_rate=(
                "fraud",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            "fraud_rate",
            ascending=False,
        )
    )

    country_summary.to_csv(
        PROCESSED_DATA_DIR
        / "fraud_by_country.csv",
        index=False,
    )

    print()
    print("Fraud by country:")
    print(country_summary.to_string(index=False))

    # -----------------------------------------------------
    # FRAUD BY MERCHANT CATEGORY
    # -----------------------------------------------------

    merchant_summary = (
        df.groupby("merchant_category")
        .agg(
            transactions=(
                "transaction_id",
                "count",
            ),
            fraud_transactions=(
                "fraud",
                "sum",
            ),
            fraud_rate=(
                "fraud",
                "mean",
            ),
            average_amount=(
                "amount",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            "fraud_rate",
            ascending=False,
        )
    )

    merchant_summary.to_csv(
        PROCESSED_DATA_DIR
        / "fraud_by_merchant_category.csv",
        index=False,
    )

    print()
    print(
        "Fraud by merchant category:"
    )
    print(
        merchant_summary.to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # FRAUD BY TRANSACTION HOUR
    # -----------------------------------------------------

    hourly_summary = (
        df.groupby("transaction_hour")
        .agg(
            transactions=(
                "transaction_id",
                "count",
            ),
            fraud_transactions=(
                "fraud",
                "sum",
            ),
            fraud_rate=(
                "fraud",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            "transaction_hour"
        )
    )

    hourly_summary.to_csv(
        PROCESSED_DATA_DIR
        / "fraud_by_hour.csv",
        index=False,
    )

    print()
    print("Fraud by transaction hour:")
    print(
        hourly_summary.to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # NEW DEVICE IMPACT
    # -----------------------------------------------------

    new_device_summary = (
        df.groupby("is_new_device")
        .agg(
            transactions=(
                "transaction_id",
                "count",
            ),
            fraud_transactions=(
                "fraud",
                "sum",
            ),
            fraud_rate=(
                "fraud",
                "mean",
            ),
        )
        .reset_index()
    )

    new_device_summary[
        "device_status"
    ] = new_device_summary[
        "is_new_device"
    ].map(
        {
            0: "Known device",
            1: "New device",
        }
    )

    new_device_summary.to_csv(
        PROCESSED_DATA_DIR
        / "fraud_by_device_status.csv",
        index=False,
    )

    print()
    print("New device impact:")
    print(
        new_device_summary.to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # NEW LOCATION IMPACT
    # -----------------------------------------------------

    new_location_summary = (
        df.groupby("is_new_location")
        .agg(
            transactions=(
                "transaction_id",
                "count",
            ),
            fraud_transactions=(
                "fraud",
                "sum",
            ),
            fraud_rate=(
                "fraud",
                "mean",
            ),
        )
        .reset_index()
    )

    new_location_summary[
        "location_status"
    ] = new_location_summary[
        "is_new_location"
    ].map(
        {
            0: "Usual country",
            1: "New country",
        }
    )

    new_location_summary.to_csv(
        PROCESSED_DATA_DIR
        / "fraud_by_location_status.csv",
        index=False,
    )

    print()
    print("New location impact:")
    print(
        new_location_summary.to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # VELOCITY IMPACT
    # -----------------------------------------------------

    velocity_summary = (
        df.groupby(
            pd.cut(
                df["transactions_last_1h"],
                bins=[
                    -1,
                    2,
                    5,
                    10,
                    float("inf"),
                ],
                labels=[
                    "0-2",
                    "3-5",
                    "6-10",
                    "11+",
                ],
            )
        )
        .agg(
            transactions=(
                "transaction_id",
                "count",
            ),
            fraud_transactions=(
                "fraud",
                "sum",
            ),
            fraud_rate=(
                "fraud",
                "mean",
            ),
        )
        .reset_index()
    )

    velocity_summary.to_csv(
        PROCESSED_DATA_DIR
        / "fraud_by_velocity.csv",
        index=False,
    )

    print()
    print("Transaction velocity impact:")
    print(
        velocity_summary.to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # AMOUNT DEVIATION IMPACT
    # -----------------------------------------------------

    deviation_summary = (
        df.groupby(
            pd.cut(
                df["amount_deviation"],
                bins=[
                    0,
                    1.5,
                    3,
                    5,
                    float("inf"),
                ],
                labels=[
                    "Normal",
                    "Moderately high",
                    "High",
                    "Very high",
                ],
            )
        )
        .agg(
            transactions=(
                "transaction_id",
                "count",
            ),
            fraud_transactions=(
                "fraud",
                "sum",
            ),
            fraud_rate=(
                "fraud",
                "mean",
            ),
        )
        .reset_index()
    )

    deviation_summary.to_csv(
        PROCESSED_DATA_DIR
        / "fraud_by_amount_deviation.csv",
        index=False,
    )

    print()
    print(
        "Amount deviation impact:"
    )
    print(
        deviation_summary.to_string(
            index=False
        )
    )

    print()
    print("========================================")
    print("FRAUD ANALYSIS COMPLETE")
    print("========================================")


if __name__ == "__main__":
    main()