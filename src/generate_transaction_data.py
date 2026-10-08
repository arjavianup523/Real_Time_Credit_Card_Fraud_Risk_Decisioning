from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

RANDOM_SEED = 42
NUM_TRANSACTIONS = 100_000

np.random.seed(RANDOM_SEED)


# ---------------------------------------------------------
# REFERENCE VALUES
# ---------------------------------------------------------

MERCHANT_CATEGORIES = [
    "Grocery",
    "Electronics",
    "Travel",
    "Fuel",
    "Restaurant",
    "Online Shopping",
    "Entertainment",
    "Healthcare",
]

COUNTRIES = [
    "India",
    "USA",
    "UK",
    "Germany",
    "Singapore",
    "UAE",
]

DEVICE_TYPES = [
    "Mobile",
    "Desktop",
    "Tablet",
]

FRAUD_RATE = 0.025


# ---------------------------------------------------------
# CUSTOMER GENERATION
# ---------------------------------------------------------

def generate_customers(num_customers=10_000):
    customer_ids = np.arange(1, num_customers + 1)

    customers = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "customer_age": np.random.randint(18, 75, num_customers),
            "account_age_days": np.random.randint(
                30,
                3650,
                num_customers,
            ),
            "home_country": np.random.choice(
                COUNTRIES,
                num_customers,
                p=[
                    0.60,
                    0.10,
                    0.08,
                    0.07,
                    0.08,
                    0.07,
                ],
            ),
        }
    )

    return customers


# ---------------------------------------------------------
# TRANSACTION GENERATION
# ---------------------------------------------------------

def generate_transactions(customers):
    num_transactions = NUM_TRANSACTIONS

    customer_ids = np.random.choice(
        customers["customer_id"],
        num_transactions,
    )

    customer_lookup = customers.set_index("customer_id")

    transaction_timestamps = pd.to_datetime(
        np.random.randint(
            pd.Timestamp("2026-01-01").value // 10**9,
            pd.Timestamp("2026-03-31").value // 10**9,
            num_transactions,
        ),
        unit="s",
    )

    merchant_category = np.random.choice(
        MERCHANT_CATEGORIES,
        num_transactions,
    )

    countries = np.random.choice(
        COUNTRIES,
        num_transactions,
        p=[
            0.60,
            0.10,
            0.08,
            0.07,
            0.08,
            0.07,
        ],
    )

    device_types = np.random.choice(
        DEVICE_TYPES,
        num_transactions,
        p=[
            0.70,
            0.20,
            0.10,
        ],
    )

    amount = np.round(
        np.random.lognormal(
            mean=7.0,
            sigma=1.0,
            size=num_transactions,
        ),
        2,
    )

    is_new_device = np.random.binomial(
        1,
        0.12,
        num_transactions,
    )

    is_new_location = (
        countries
        != customer_lookup.loc[
            customer_ids,
            "home_country",
        ].to_numpy()
    ).astype(int)

    transactions_last_1h = np.random.poisson(
        2.0,
        num_transactions,
    )

    transactions_last_24h = (
        transactions_last_1h
        + np.random.poisson(
            6.0,
            num_transactions,
        )
    )

    avg_amount_last_30d = np.round(
        np.random.lognormal(
            mean=6.5,
            sigma=0.6,
            size=num_transactions,
        ),
        2,
    )

    amount_deviation = np.round(
        amount / np.maximum(avg_amount_last_30d, 1),
        2,
    )

    transaction_hour = transaction_timestamps.hour

    is_night = (
        (transaction_hour >= 0)
        & (transaction_hour <= 5)
    ).astype(int)

    # -----------------------------------------------------
    # FRAUD RISK SCORE
    # -----------------------------------------------------

    risk_score = (
        0.0
        + 1.2 * is_new_device
        + 1.0 * is_new_location
        + 0.12 * transactions_last_1h
        + 0.8 * is_night
        + 0.6 * (amount_deviation > 3).astype(int)
        + 0.8 * (amount > 50_000).astype(int)
        + 0.5 * (transactions_last_24h > 20).astype(int)
    )

    fraud_probability = (
        1
        / (
            1
            + np.exp(
                -(
                    risk_score - 3.2
                )
            )
        )
    )

    # Scale probabilities so fraud remains a rare event.
    fraud_probability = (
        fraud_probability
        * FRAUD_RATE
        / fraud_probability.mean()
    )

    fraud_probability = np.clip(
        fraud_probability,
        0,
        0.95,
    )

    fraud = np.random.binomial(
        1,
        fraud_probability,
    )

    transactions = pd.DataFrame(
        {
            "transaction_id": np.arange(
                1,
                num_transactions + 1,
            ),
            "customer_id": customer_ids,
            "transaction_timestamp": transaction_timestamps,
            "amount": amount,
            "merchant_category": merchant_category,
            "country": countries,
            "device_type": device_types,
            "is_new_device": is_new_device,
            "is_new_location": is_new_location,
            "transactions_last_1h": transactions_last_1h,
            "transactions_last_24h": transactions_last_24h,
            "avg_amount_last_30d": avg_amount_last_30d,
            "amount_deviation": amount_deviation,
            "transaction_hour": transaction_hour,
            "is_night": is_night,
            "customer_age": customer_lookup.loc[
                customer_ids,
                "customer_age",
            ].to_numpy(),
            "account_age_days": customer_lookup.loc[
                customer_ids,
                "account_age_days",
            ].to_numpy(),
            "fraud": fraud,
        }
    )

    return transactions


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    print("Generating customer profiles...")

    customers = generate_customers()

    print("Generating transaction events...")

    transactions = generate_transactions(
        customers
    )

    output_path = (
        RAW_DATA_DIR
        / "transactions.csv"
    )

    transactions.to_csv(
        output_path,
        index=False,
    )

    fraud_count = transactions["fraud"].sum()
    fraud_rate = transactions["fraud"].mean()

    print()
    print("Transaction dataset generated successfully.")
    print(f"Rows: {len(transactions):,}")
    print(
        f"Fraud transactions: "
        f"{fraud_count:,}"
    )
    print(
        f"Fraud rate: "
        f"{fraud_rate:.2%}"
    )
    print(
        f"Saved to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()