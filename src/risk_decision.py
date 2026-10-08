from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

THRESHOLD_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "threshold_results.csv"
)


def get_optimal_threshold():
    results = pd.read_csv(
        THRESHOLD_PATH
    )

    best_row = results.loc[
        results["f1_score"].idxmax()
    ]

    return float(
        best_row["threshold"]
    )


def probability_to_risk_score(
    fraud_probability,
):
    probability = max(
        0.0,
        min(
            1.0,
            float(fraud_probability),
        ),
    )

    return round(
        probability * 100,
        2,
    )


def make_risk_decision(
    fraud_probability,
):
    optimal_threshold = (
        get_optimal_threshold()
    )

    probability = float(
        fraud_probability
    )

    risk_score = (
        probability_to_risk_score(
            probability
        )
    )

    review_threshold = (
        optimal_threshold * 0.50
    )

    if probability >= optimal_threshold:
        decision = "BLOCK"
        risk_level = "HIGH"

    elif probability >= review_threshold:
        decision = "REVIEW"
        risk_level = "MEDIUM"

    else:
        decision = "APPROVE"
        risk_level = "LOW"

    return {
        "fraud_probability": round(
            probability,
            6,
        ),
        "risk_score": risk_score,
        "risk_level": risk_level,
        "decision": decision,
        "block_threshold": round(
            optimal_threshold,
            4,
        ),
        "review_threshold": round(
            review_threshold,
            4,
        ),
    }