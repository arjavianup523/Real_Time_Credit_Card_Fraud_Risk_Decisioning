# Real-Time Credit Card Fraud Detection & Risk Decisioning

A production-style machine learning system for detecting potentially fraudulent credit card transactions and making automated risk-based decisions in real time.

The project combines machine learning, risk scoring, API-based inference, Streamlit visualization, Docker, testing, and transaction simulation into an end-to-end fraud detection and decisioning platform.

## Project Overview

Credit card fraud detection is a real-world machine learning problem where predictions need to be generated quickly and translated into actionable business decisions.

This project simulates a real-time fraud detection workflow:

1. A transaction is submitted to the system.
2. Transaction features are processed.
3. The trained machine learning model generates a fraud probability.
4. The probability is converted into a risk score.
5. The transaction is assigned a risk level.
6. A decision is automatically generated:
   - Approve
   - Review
   - Block

The system is designed to demonstrate how a machine learning model can be integrated into a production-style application rather than being used only inside a notebook.

## Key Features

- Machine learning-based fraud detection
- Real-time transaction simulation
- Fraud probability prediction
- Risk score generation
- Automated transaction decisioning
- Configurable review and block thresholds
- Streamlit interactive dashboard
- REST API for model inference
- Docker support
- Unit testing
- Structured project architecture
- Model persistence
- Separate raw and processed data
- Git and GitHub version control

## Risk Decisioning

The system converts the model's fraud probability into a business decision.

| Fraud Probability | Risk Level | Decision |
|---|---|---|
| Below 35% | Low | Approve |
| 35% - 70% | Medium | Review |
| Above 70% | High | Block |

These thresholds are configurable and can be adjusted according to the requirements of a financial institution.

The decisioning layer is important because a fraud detection model does not directly tell a business what action to take. The model prediction must be translated into an operational decision.

## System Architecture

```text
Transaction
     |
     v
Transaction Input
     |
     v
Feature Processing
     |
     v
Machine Learning Model
     |
     v
Fraud Probability
     |
     v
Risk Scoring
     |
     v
Risk Classification
     |
     v
Decision Engine
     |
     +----------------+
     |                |
     v                v
  Approve          Review / Block
     |
     v
Dashboard / API Response