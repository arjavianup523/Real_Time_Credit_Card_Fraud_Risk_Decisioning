from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ML_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "ml_dataset.csv"
RESULTS_PATH = PROJECT_ROOT / "data" / "processed" / "threshold_results.csv"


def main():
    print("Loading ML dataset...")

    df = pd.read_csv(ML_DATA_PATH)

    X = df.drop(columns=["fraud"])
    y = df["fraud"]

    categorical_features = [
        "merchant_category",
        "country",
        "device_type",
    ]

    numeric_features = [
        column
        for column in X.columns
        if column not in categorical_features
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                categorical_features,
            ),
            (
                "numeric",
                StandardScaler(),
                numeric_features,
            ),
        ]
    )

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    print("Training Logistic Regression...")

    pipeline.fit(X_train, y_train)

    probabilities = pipeline.predict_proba(X_test)[:, 1]

    pr_auc = average_precision_score(
        y_test,
        probabilities,
    )

    thresholds = [
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
    ]

    results = []

    for threshold in thresholds:
        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0,
        )

        results.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "predicted_fraud_count": int(
                    predictions.sum()
                ),
                "actual_fraud_count": int(
                    y_test.sum()
                ),
                "pr_auc": pr_auc,
            }
        )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        RESULTS_PATH,
        index=False,
    )

    best_row = results_df.loc[
        results_df["f1_score"].idxmax()
    ]

    print()
    print("=" * 50)
    print("THRESHOLD OPTIMIZATION COMPLETE")
    print("=" * 50)
    print(
        f"Best threshold: {best_row['threshold']:.2f}"
    )
    print(
        f"Precision: {best_row['precision']:.4f}"
    )
    print(
        f"Recall: {best_row['recall']:.4f}"
    )
    print(
        f"F1: {best_row['f1_score']:.4f}"
    )
    print(
        f"PR-AUC: {pr_auc:.4f}"
    )
    print(
        f"Results saved to: {RESULTS_PATH}"
    )


if __name__ == "__main__":
    main()