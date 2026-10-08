from fastapi.testclient import TestClient

from api.app import app


client = TestClient(app)


def create_transaction():
    return {
        "amount": 2500.0,
        "merchant_category": "Online Shopping",
        "country": "India",
        "device_type": "Mobile",
        "is_new_device": 0,
        "is_new_location": 0,
        "transactions_last_1h": 2,
        "transactions_last_24h": 5,
        "avg_amount_last_30d": 1500.0,
        "amount_deviation": 0.5,
        "transaction_hour": 14,
        "is_night": 0,
        "customer_age": 30,
        "account_age_days": 500,
        "transaction_day_of_week": 2,
        "transaction_day": 15,
        "transaction_month": 6,
        "amount_log": 7.824,
        "amount_vs_customer_average": 1.667,
        "velocity_ratio": 0.4,
        "high_velocity_flag": 0,
        "high_daily_activity_flag": 0,
        "young_customer_flag": 0,
        "new_account_flag": 0,
        "device_location_risk": 0,
        "night_and_new_device": 0,
        "night_and_new_location": 0,
        "high_amount_and_new_device": 0,
    }


def test_health():
    response = client.get(
        "/health"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_prediction():
    response = client.post(
        "/predict",
        json=create_transaction(),
    )

    assert response.status_code == 200

    data = response.json()

    assert "fraud_probability" in data
    assert "risk_score" in data
    assert "risk_level" in data
    assert "decision" in data

    assert (
        0
        <= data["fraud_probability"]
        <= 1
    )

    assert (
        0
        <= data["risk_score"]
        <= 100
    )

    assert data["risk_level"] in [
        "LOW",
        "MEDIUM",
        "HIGH",
    ]

    assert data["decision"] in [
        "APPROVE",
        "REVIEW",
        "BLOCK",
    ]