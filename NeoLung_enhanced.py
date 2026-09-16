"""
NeoLung
ML-Based Lung Cancer Risk Prediction System
============================================

This module provides a clean, reproducible machine-learning pipeline for
the survey-based lung cancer prediction project.

Pipeline:
    Dataset -> Validation -> Preprocessing -> Train/Test Split
            -> Multiple ML Models -> Evaluation -> Comparison
            -> Best Model -> Saved Artifacts

Dataset expected:
    data/survey_lung_cancer.csv

Target:
    LUNG_CANCER (YES/NO or 1/0)

Models:
    Logistic Regression
    K-Nearest Neighbors
    Decision Tree
    Support Vector Machine
    Gaussian Naive Bayes
    Random Forest
    Gradient Boosting
    Neural Network (MLP)
    AdaBoost
    XGBoost (optional)

Outputs:
    results/model_comparison.csv
    results/classification_report_<model>.txt
    results/confusion_matrix_<model>.png
    models/<model>.joblib
    models/feature_columns.joblib

Important:
    This is an academic risk-prediction project. It is not a medical
    diagnosis tool and model scores should not be interpreted as clinical
    probabilities without proper clinical validation.
"""

from __future__ import annotations

from pathlib import Path
import re
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
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

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "survey_lung_cancer.csv"
MODEL_DIR = BASE_DIR / "models"
RESULT_DIR = BASE_DIR / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def print_section(title: str) -> None:
    """Print a readable terminal section heading."""
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def normalise_column_name(name: str) -> str:
    """Make column names consistent without changing their meaning."""
    name = str(name).strip().upper()
    name = re.sub(r"[\s\-]+", "_", name)
    return name


def find_target_column(columns) -> str:
    """Find the lung-cancer target column."""
    normalised = {
        normalise_column_name(column): column
        for column in columns
    }

    candidates = [
        "LUNG_CANCER",
        "LUNG_CANCER_STATUS",
        "LUNG_CANCER_LABEL",
    ]

    for candidate in candidates:
        if candidate in normalised:
            return normalised[candidate]

    raise ValueError(
        "Could not find the target column. Expected a column such as "
        "'LUNG_CANCER'."
    )


def encode_target(series: pd.Series) -> pd.Series:
    """Convert common YES/NO target labels into 0/1."""
    mapping = {
        "YES": 1,
        "Y": 1,
        "TRUE": 1,
        "POSITIVE": 1,
        "1": 1,
        "NO": 0,
        "N": 0,
        "FALSE": 0,
        "NEGATIVE": 0,
        "0": 0,
    }

    result = []

    for value in series:
        if pd.isna(value):
            result.append(np.nan)
            continue

        text = str(value).strip().upper()

        if text in mapping:
            result.append(mapping[text])
            continue

        try:
            numeric = float(text)
            if numeric in (0, 1):
                result.append(int(numeric))
            else:
                result.append(np.nan)
        except ValueError:
            result.append(np.nan)

    return pd.Series(result, index=series.index, dtype="float64")


def safe_filename(name: str) -> str:
    """Convert a model name into a safe filename."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


# ============================================================
# DATA LOADING
# ============================================================

def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the CSV dataset."""
    if not path.exists():
        raise FileNotFoundError(
            f"\nDataset not found:\n{path}\n\n"
            "Expected structure:\n"
            "NeoLung/\n"
            "├── lung_cancer_ml.py\n"
            "└── data/\n"
            "    └── survey_lung_cancer.csv"
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("The dataset is empty.")

    df.columns = [normalise_column_name(c) for c in df.columns]

    print(f"Dataset: {path.name}")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    return df


# ============================================================
# DATA PREPROCESSING
# ============================================================

def preprocess_dataset(df: pd.DataFrame):
    """
    Prepare X and y while preserving categorical information.

    YES/NO values in features are normalised to numeric 1/0.
    Remaining categorical columns are handled by OneHotEncoder.
    Missing numeric values use the median and missing categorical
    values use the most frequent category.
    """
    data = df.copy()

    target_column = find_target_column(data.columns)

    # Remove duplicate rows.
    before = len(data)
    data = data.drop_duplicates().reset_index(drop=True)
    duplicates_removed = before - len(data)

    # Encode the target.
    data[target_column] = encode_target(data[target_column])

    # Remove rows where the target cannot be interpreted.
    before_target_clean = len(data)
    data = data.dropna(subset=[target_column]).reset_index(drop=True)
    target_rows_removed = before_target_clean - len(data)

    X = data.drop(columns=[target_column])
    y = data[target_column].astype(int)

    # Convert common binary strings in feature columns.
    binary_map = {
        "YES": 1,
        "NO": 0,
        "Y": 1,
        "N": 0,
        "TRUE": 1,
        "FALSE": 0,
    }

    for column in X.columns:
        if X[column].dtype == "object":
            cleaned = X[column].astype(str).str.strip().str.upper()
            unique_values = set(cleaned.dropna().unique())

            if unique_values and unique_values.issubset(
                set(binary_map.keys())
            ):
                X[column] = cleaned.map(binary_map)

    numeric_columns = X.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical_columns = [
        column for column in X.columns
        if column not in numeric_columns
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent"),
                    ),
                    (
                        "onehot",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False,
                        ),
                    ),
                ]),
                categorical_columns,
            ),
        ],
        remainder="drop",
    )

    info = {
        "target_column": target_column,
        "duplicates_removed": duplicates_removed,
        "target_rows_removed": target_rows_removed,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
    }

    return X, y, preprocessor, info


# ============================================================
# MODEL DEFINITIONS
# ============================================================

def create_models() -> dict:
    """Create the classification models used in the project."""
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),

        "KNN": KNeighborsClassifier(
            n_neighbors=5,
        ),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=4,
            random_state=RANDOM_STATE,
        ),

        "SVM": SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),

        "Naive Bayes": GaussianNB(),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_split=4,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),

        "Neural Network": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=0.0005,
            learning_rate_init=0.001,
            max_iter=1500,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=RANDOM_STATE,
        ),

        "AdaBoost": AdaBoostClassifier(
            n_estimators=150,
            learning_rate=0.5,
            random_state=RANDOM_STATE,
        ),
    }

    # Optional XGBoost model.
    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    except ImportError:
        print(
            "\n[XGBoost] Package not installed. "
            "The remaining models will still run."
        )
        print("Install it with: pip install xgboost")

    return models


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    predictions,
    probabilities=None,
) -> dict:
    """Calculate the main binary-classification metrics."""
    metrics = {
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "Recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "F1-Score": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
    }

    if probabilities is not None:
        try:
            metrics["ROC-AUC"] = roc_auc_score(
                y_true,
                probabilities,
            )
        except ValueError:
            metrics["ROC-AUC"] = np.nan
    else:
        metrics["ROC-AUC"] = np.nan

    return metrics


# ============================================================
# TRAINING
# ============================================================

def train_and_evaluate(
    models: dict,
    preprocessor,
    X_train,
    X_test,
    y_train,
    y_test,
):
    """Train every model and collect evaluation results."""
    results = []
    trained_models = {}
    predictions_by_model = {}

    for name, estimator in models.items():
        print(f"\nTraining {name}...")

        pipeline = Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("model", estimator),
        ])

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        probabilities = None
        if hasattr(pipeline, "predict_proba"):
            probabilities = pipeline.predict_proba(X_test)[:, 1]

        metrics = calculate_metrics(
            y_test,
            predictions,
            probabilities,
        )

        row = {"Model": name, **metrics}
        results.append(row)

        trained_models[name] = pipeline
        predictions_by_model[name] = predictions

        print(
            f"Accuracy={metrics['Accuracy']:.4f} | "
            f"Precision={metrics['Precision']:.4f} | "
            f"Recall={metrics['Recall']:.4f} | "
            f"F1={metrics['F1-Score']:.4f} | "
            f"ROC-AUC={metrics['ROC-AUC']:.4f}"
        )

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by=["F1-Score", "Recall", "Accuracy"],
        ascending=False,
    ).reset_index(drop=True)

    return results_df, trained_models, predictions_by_model


# ============================================================
# RESULT VISUALISATION
# ============================================================

def save_model_comparison_plot(results_df: pd.DataFrame) -> None:
    """Save a comparison chart for all models."""
    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
    ]

    x = np.arange(len(results_df))
    width = 0.18

    plt.figure(figsize=(14, 7))

    for index, metric in enumerate(metrics):
        plt.bar(
            x + (index - 1.5) * width,
            results_df[metric] * 100,
            width,
            label=metric,
        )

    plt.xticks(
        x,
        results_df["Model"],
        rotation=35,
        ha="right",
    )
    plt.ylabel("Score (%)")
    plt.title("NeoLung - Model Performance Comparison")
    plt.ylim(0, 105)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULT_DIR / "model_comparison.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()


def save_confusion_matrix(
    model,
    model_name: str,
    X_test,
    y_test,
) -> None:
    """Save a confusion-matrix image for one model."""
    predictions = model.predict(X_test)
    matrix = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, interpolation="nearest")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.colorbar()

    labels = ["No Lung Cancer", "Lung Cancer"]
    ticks = np.arange(len(labels))

    plt.xticks(ticks, labels, rotation=15)
    plt.yticks(ticks, labels)

    threshold = matrix.max() / 2

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            plt.text(
                column,
                row,
                matrix[row, column],
                ha="center",
                va="center",
                color=(
                    "white"
                    if matrix[row, column] > threshold
                    else "black"
                ),
                fontsize=12,
            )

    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()

    filename = (
        f"confusion_matrix_{safe_filename(model_name)}.png"
    )

    plt.savefig(
        RESULT_DIR / filename,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()


def save_classification_report(
    model,
    model_name: str,
    X_test,
    y_test,
) -> None:
    """Save the detailed classification report."""
    predictions = model.predict(X_test)

    report = classification_report(
        y_test,
        predictions,
        target_names=[
            "No Lung Cancer",
            "Lung Cancer",
        ],
        zero_division=0,
    )

    filename = (
        f"classification_report_{safe_filename(model_name)}.txt"
    )

    (RESULT_DIR / filename).write_text(
        report,
        encoding="utf-8",
    )


# ============================================================
# SAVE MODELS
# ============================================================

def save_trained_models(trained_models: dict) -> None:
    """Save every trained pipeline for later inference."""
    for name, model in trained_models.items():
        filename = f"{safe_filename(name)}.joblib"

        joblib.dump(
            model,
            MODEL_DIR / filename,
        )


# ============================================================
# SAVE PROJECT METADATA
# ============================================================

def save_metadata(
    results_df: pd.DataFrame,
    dataset_info: dict,
    X_train,
    X_test,
) -> None:
    """Save reproducibility information."""
    metadata = {
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "training_samples": len(X_train),
        "testing_samples": len(X_test),
        "target_column": dataset_info["target_column"],
        "duplicates_removed": dataset_info["duplicates_removed"],
        "target_rows_removed": dataset_info["target_rows_removed"],
        "numeric_features": dataset_info["numeric_columns"],
        "categorical_features": dataset_info["categorical_columns"],
        "models_trained": results_df["Model"].tolist(),
    }

    joblib.dump(
        metadata,
        MODEL_DIR / "training_metadata.joblib",
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print_section(
        "NEOLUNG - ML BASED LUNG CANCER RISK PREDICTION SYSTEM"
    )

    # 1. Load data
    print_section("1. DATA LOADING")
    df = load_dataset()

    # 2. Preprocess
    print_section("2. DATA PREPROCESSING")
    X, y, preprocessor, dataset_info = preprocess_dataset(df)

    print(
        f"Target column: {dataset_info['target_column']}"
    )
    print(
        f"Duplicates removed: "
        f"{dataset_info['duplicates_removed']}"
    )
    print(
        f"Rows removed because target was invalid/missing: "
        f"{dataset_info['target_rows_removed']}"
    )
    print(f"Final samples: {len(X)}")
    print(f"Positive class: {int(y.sum())}")
    print(f"Negative class: {int((y == 0).sum())}")

    # 3. Train/test split
    print_section("3. TRAIN / TEST SPLIT")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")

    # 4. Create models
    print_section("4. MODEL INITIALISATION")
    models = create_models()
    print("Models:", ", ".join(models.keys()))

    # 5. Train and evaluate
    print_section("5. MODEL TRAINING AND EVALUATION")

    results_df, trained_models, predictions_by_model = (
        train_and_evaluate(
            models,
            preprocessor,
            X_train,
            X_test,
            y_train,
            y_test,
        )
    )

    # 6. Save results
    print_section("6. SAVING RESULTS")

    results_df.to_csv(
        RESULT_DIR / "model_comparison.csv",
        index=False,
    )

    save_model_comparison_plot(results_df)

    for name, model in trained_models.items():
        save_confusion_matrix(
            model,
            name,
            X_test,
            y_test,
        )

        save_classification_report(
            model,
            name,
            X_test,
            y_test,
        )

    save_trained_models(trained_models)
    save_metadata(
        results_df,
        dataset_info,
        X_train,
        X_test,
    )

    # 7. Display comparison
    print_section("7. FINAL MODEL COMPARISON")

    display_df = results_df.copy()

    for column in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "ROC-AUC",
    ]:
        display_df[column] = (
            display_df[column] * 100
        ).round(2)

    print(display_df.to_string(index=False))

    # 8. Report model with highest F1 in this particular run.
    # This is descriptive only; it is NOT a universal claim about
    # which algorithm is best for lung-cancer prediction.
    selected_model = results_df.iloc[0]["Model"]

    print_section("8. RUN SUMMARY")
    print(
        f"Highest F1-Score in this train/test run: "
        f"{selected_model}"
    )
    print(
        "Use the saved model_comparison.csv and reports when "
        "documenting the actual results."
    )

    print("\nGenerated files:")
    print(f"  Models  : {MODEL_DIR}")
    print(f"  Results : {RESULT_DIR}")

    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()
