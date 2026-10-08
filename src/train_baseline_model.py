from pathlib import Path

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
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ML_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_dataset.csv"
)

MODEL_RESULTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "baseline_model_results.csv"
)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    print("Loading ML dataset...")

    if not ML_DATA_PATH.exists():
        raise FileNotFoundError(
            f"ML dataset not found: {ML_DATA_PATH}"
        )

    df = pd.read_csv(
        ML_DATA_PATH
    )

    print(
        f"Rows: {len(df):,}"
    )

    # -----------------------------------------------------
    # FEATURES AND TARGET
    # -----------------------------------------------------

    X = df.drop(
        columns=["fraud"]
    )

    y = df["fraud"]

    print()
    print(
        f"Fraud rate: {y.mean():.2%}"
    )

    # -----------------------------------------------------
    # CATEGORICAL FEATURES
    # -----------------------------------------------------

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

    print()
    print(
        f"Numeric features: "
        f"{len(numeric_features)}"
    )

    print(
        f"Categorical features: "
        f"{len(categorical_features)}"
    )

    # -----------------------------------------------------
    # PREPROCESSING
    # -----------------------------------------------------

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
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

    # -----------------------------------------------------
    # MODEL
    # -----------------------------------------------------

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    # -----------------------------------------------------
    # TRAIN / TEST SPLIT
    # -----------------------------------------------------

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y,
        )
    )

    print()
    print(
        f"Training rows: {len(X_train):,}"
    )

    print(
        f"Testing rows: {len(X_test):,}"
    )

    # -----------------------------------------------------
    # TRAIN MODEL
    # -----------------------------------------------------

    print()
    print(
        "Training Logistic Regression..."
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    print(
        "Training completed."
    )

    # -----------------------------------------------------
    # PREDICT FRAUD PROBABILITIES
    # -----------------------------------------------------

    y_probability = (
        pipeline.predict_proba(
            X_test
        )[:, 1]
    )

    # -----------------------------------------------------
    # DEFAULT THRESHOLD
    # -----------------------------------------------------

    y_pred = (
        y_probability >= 0.50
    ).astype(int)

    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability,
    )

    pr_auc = average_precision_score(
        y_test,
        y_probability,
    )

    # -----------------------------------------------------
    # RESULTS
    # -----------------------------------------------------

    print()
    print(
        "========================================"
    )
    print(
        "BASELINE MODEL RESULTS"
    )
    print(
        "========================================"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall:    {recall:.4f}"
    )

    print(
        f"F1 Score:  {f1:.4f}"
    )

    print(
        f"ROC-AUC:   {roc_auc:.4f}"
    )

    print(
        f"PR-AUC:    {pr_auc:.4f}"
    )

    # -----------------------------------------------------
    # CLASSIFICATION REPORT
    # -----------------------------------------------------

    print()
    print(
        "Classification Report:"
    )

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "Legitimate",
                "Fraud",
            ],
            zero_division=0,
        )
    )

    # -----------------------------------------------------
    # CONFUSION MATRIX
    # -----------------------------------------------------

    print(
        "Confusion Matrix:"
    )

    matrix = confusion_matrix(
        y_test,
        y_pred,
    )

    print(matrix)

    # -----------------------------------------------------
    # SAVE METRICS
    # -----------------------------------------------------

    results = pd.DataFrame(
        {
            "metric": [
                "precision",
                "recall",
                "f1_score",
                "roc_auc",
                "pr_auc",
            ],
            "value": [
                precision,
                recall,
                f1,
                roc_auc,
                pr_auc,
            ],
        }
    )

    results.to_csv(
        MODEL_RESULTS_PATH,
        index=False,
    )

    # -----------------------------------------------------
    # FINAL MESSAGE
    # -----------------------------------------------------

    print()
    print(
        "========================================"
    )
    print(
        "BASELINE MODEL COMPLETE"
    )
    print(
        "========================================"
    )

    print(
        f"Results saved to: "
        f"{MODEL_RESULTS_PATH}"
    )


if __name__ == "__main__":
    main()