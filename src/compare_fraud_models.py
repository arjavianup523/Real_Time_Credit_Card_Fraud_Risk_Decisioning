from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
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

RESULTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_comparison.csv"
)


# ---------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------

def create_preprocessor(
    categorical_features,
    numeric_features,
):
    return ColumnTransformer(
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


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    print("Loading ML dataset...")

    df = pd.read_csv(
        ML_DATA_PATH
    )

    X = df.drop(
        columns=["fraud"]
    )

    y = df["fraud"]

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Fraud rate: {y.mean():.2%}"
    )

    # -----------------------------------------------------
    # FEATURES
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
    # MODELS
    # -----------------------------------------------------

    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=42,
        ),
    }

    results = []

    # -----------------------------------------------------
    # TRAIN MODELS
    # -----------------------------------------------------

    for model_name, model in models.items():

        print()
        print(
            "========================================"
        )
        print(
            f"Training: {model_name}"
        )
        print(
            "========================================"
        )

        if model_name == "HistGradientBoosting":
            # HistGradientBoosting requires numeric
            # input, so encode categorical variables
            # without scaling them.
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
                ],
                remainder="passthrough",
            )
        else:
            preprocessor = create_preprocessor(
                categorical_features,
                numeric_features,
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

        pipeline.fit(
            X_train,
            y_train,
        )

        probabilities = (
            pipeline.predict_proba(
                X_test
            )[:, 1]
        )

        predictions = (
            probabilities >= 0.50
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

        results.append(
            {
                "model": model_name,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
            }
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
    # COMPARISON
    # -----------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        "pr_auc",
        ascending=False,
    )

    results_df.to_csv(
        RESULTS_PATH,
        index=False,
    )

    print()
    print(
        "========================================"
    )
    print(
        "MODEL COMPARISON"
    )
    print(
        "========================================"
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    print()
    print(
        f"Results saved to: {RESULTS_PATH}"
    )


if __name__ == "__main__":
    main()