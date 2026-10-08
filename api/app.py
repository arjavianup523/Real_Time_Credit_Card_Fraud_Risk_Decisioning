from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.risk_decision import (
    make_risk_decision,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "fraud_detection_model.joblib"
)


app = FastAPI(
    title="Real-Time Credit Card Fraud Risk API",
    description=(
        "Real-time fraud detection "
        "and risk decisioning API."
    ),
    version="1.0.0",
)


model = joblib.load(
    MODEL_PATH
)


class TransactionRequest(BaseModel):

    amount: float = Field(
        ...,
        ge=0,
    )

    merchant_category: str

    country: str

    device_type: str

    is_new_device: int = Field(
        ...,
        ge=0,
        le=1,
    )

    is_new_location: int = Field(
        ...,
        ge=0,
        le=1,
    )

    transactions_last_1h: int = Field(
        ...,
        ge=0,
    )

    transactions_last_24h: int = Field(
        ...,
        ge=0,
    )

    avg_amount_last_30d: float = Field(
        ...,
        ge=0,
    )

    amount_deviation: float

    transaction_hour: int = Field(
        ...,
        ge=0,
        le=23,
    )

    is_night: int = Field(
        ...,
        ge=0,
        le=1,
    )

    customer_age: int = Field(
        ...,
        ge=18,
    )

    account_age_days: int = Field(
        ...,
        ge=0,
    )

    transaction_day_of_week: int = Field(
        ...,
        ge=0,
        le=6,
    )

    transaction_day: int = Field(
        ...,
        ge=1,
        le=31,
    )

    transaction_month: int = Field(
        ...,
        ge=1,
        le=12,
    )

    amount_log: float

    amount_vs_customer_average: float

    velocity_ratio: float

    high_velocity_flag: int = Field(
        ...,
        ge=0,
        le=1,
    )

    high_daily_activity_flag: int = Field(
        ...,
        ge=0,
        le=1,
    )

    young_customer_flag: int = Field(
        ...,
        ge=0,
        le=1,
    )

    new_account_flag: int = Field(
        ...,
        ge=0,
        le=1,
    )

    device_location_risk: int = Field(
        ...,
        ge=0,
        le=1,
    )

    night_and_new_device: int = Field(
        ...,
        ge=0,
        le=1,
    )

    night_and_new_location: int = Field(
        ...,
        ge=0,
        le=1,
    )

    high_amount_and_new_device: int = Field(
        ...,
        ge=0,
        le=1,
    )


@app.get("/")
def root():
    return {
        "service": (
            "Real-Time Credit Card "
            "Fraud Risk API"
        ),
        "status": "running",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
    }


@app.post("/predict")
def predict(
    transaction: TransactionRequest,
):
    try:
        transaction_data = pd.DataFrame(
            [
                transaction.model_dump()
            ]
        )

        fraud_probability = float(
            model.predict_proba(
                transaction_data
            )[0][1]
        )

        decision = make_risk_decision(
            fraud_probability
        )

        return {
            "fraud_probability": (
                decision[
                    "fraud_probability"
                ]
            ),
            "risk_score": (
                decision[
                    "risk_score"
                ]
            ),
            "risk_level": (
                decision[
                    "risk_level"
                ]
            ),
            "decision": (
                decision[
                    "decision"
                ]
            ),
            "thresholds": {
                "review": (
                    decision[
                        "review_threshold"
                    ]
                ),
                "block": (
                    decision[
                        "block_threshold"
                    ]
                ),
            },
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )