from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ML_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "ml_dataset.csv"
THRESHOLD_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "threshold_results.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODEL_DIR / "fraud_detection_model.joblib"
METRICS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_model_metrics.csv"
)


def main():
    print("Loading dataset...")

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

    threshold_results = pd.read_csv(
        THRESHOLD_PATH
    )

    best_row = threshold_results.loc[
        threshold_results["f1_score"].idxmax()
    ]

    optimal_threshold = float(
        best_row["threshold"]
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

    print("Training final Logistic Regression model...")

    pipeline.fit(X_train, y_train)

    probabilities = pipeline.predict_proba(X_test)[:, 1]

    predictions = (
        probabilities >= optimal_threshold
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

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        pipeline,
        MODEL_PATH,
    )

    metrics = pd.DataFrame(
        [
            {
                "model": "Logistic Regression",
                "optimal_threshold": optimal_threshold,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "training_rows": len(X_train),
                "testing_rows": len(X_test),
            }
        ]
    )

    metrics.to_csv(
        METRICS_PATH,
        index=False,
    )

    print()
    print("=" * 50)
    print("FINAL MODEL COMPLETE")
    print("=" * 50)
    print(
        f"Optimal threshold: {optimal_threshold:.2f}"
    )
    print(
        f"Precision: {precision:.4f}"
    )
    print(
        f"Recall: {recall:.4f}"
    )
    print(
        f"F1: {f1:.4f}"
    )
    print(
        f"ROC-AUC: {roc_auc:.4f}"
    )
    print(
        f"PR-AUC: {pr_auc:.4f}"
    )

    print()
    print("Confusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions,
        )
    )

    print()
    print("Classification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    print(
        f"Model saved to: {MODEL_PATH}"
    )


if __name__ == "__main__":
    main()