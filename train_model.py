"""Training script for weather-based wind turbine power prediction."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

matplotlib.use("Agg")

np.random.seed(42)


def normalize_column_name(name: str) -> str:
    cleaned = (
        name.strip()
        .lower()
        .replace("(", "")
        .replace(")", "")
        .replace("/", " per ")
        .replace("-", "_")
    )
    return "_".join(part for part in cleaned.split() if part)


def detect_columns(columns: List[str]) -> Dict[str, str]:
    lowered = {col: col.lower() for col in columns}
    target = None
    theoretical = None
    wind_speed = None
    wind_direction = None

    for original, lower in lowered.items():
        if "power" in lower and "theoretical" not in lower:
            target = original
            break

    if target is None:
        for original, lower in lowered.items():
            if "power" in lower:
                target = original
                break

    for original, lower in lowered.items():
        if "theoretical" in lower and "power" in lower:
            theoretical = original
        if "wind" in lower and "speed" in lower and wind_speed is None:
            wind_speed = original
        if "direction" in lower and wind_direction is None:
            wind_direction = original

    if target is None:
        column_hint = ", ".join(columns)
        raise ValueError(
            "Unable to automatically detect the target power column. Columns found: "
            f"{column_hint}"
        )

    return {
        "target": target,
        "theoretical": theoretical,
        "wind_speed": wind_speed,
        "wind_direction": wind_direction,
    }


def ensure_directories(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def generate_plots(df: pd.DataFrame, target_col: str, plot_dir: Path, detection: Dict[str, str]) -> None:
    ensure_directories(plot_dir)
    numeric_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(8, 6))
    corr = numeric_df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap="viridis", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(plot_dir / "correlation_heatmap.png", dpi=160)
    plt.close()

    wind_speed_col = detection.get("wind_speed")
    if wind_speed_col and wind_speed_col in df.columns:
        plt.figure(figsize=(7, 5))
        sns.scatterplot(x=df[wind_speed_col], y=df[target_col])
        plt.xlabel(f"{wind_speed_col} (normalized)")
        plt.ylabel(target_col)
        plt.title("Wind Speed vs Power")
        plt.tight_layout()
        plt.savefig(plot_dir / "wind_speed_vs_power.png", dpi=160)
        plt.close()

    theoretical_col = detection.get("theoretical")
    if theoretical_col and theoretical_col in df.columns:
        plt.figure(figsize=(7, 5))
        sns.scatterplot(x=df[theoretical_col], y=df[target_col])
        plt.xlabel(theoretical_col)
        plt.ylabel(target_col)
        plt.title("Theoretical Curve vs Actual Power")
        plt.tight_layout()
        plt.savefig(plot_dir / "theoretical_vs_power.png", dpi=160)
        plt.close()

    plt.figure(figsize=(7, 5))
    sns.histplot(df[target_col], kde=True, bins=20)
    plt.xlabel(target_col)
    plt.title("Target Power Distribution")
    plt.tight_layout()
    plt.savefig(plot_dir / "power_distribution.png", dpi=160)
    plt.close()



def build_pipeline(feature_names: List[str], df: pd.DataFrame) -> Pipeline:
    numeric_features = df[feature_names].select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [col for col in feature_names if col not in numeric_features]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_transformer, numeric_features))

    if categorical_features:
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        transformers.append(("cat", categorical_transformer, categorical_features))

    if not transformers:
        raise ValueError("No valid features available for preprocessing")

    preprocessor = ColumnTransformer(transformers=transformers)

    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=14,
        min_samples_leaf=2,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(rmse))
    r2 = r2_score(y_true, y_pred)
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train wind turbine power prediction model")
    parser.add_argument(
        "--data",
        type=str,
        default=str(Path("data") / "wind_dataset.csv"),
        help="Path to training CSV file",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(Path("model")),
        help="Directory to store the trained model and metadata",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default=str(Path("static") / "plots"),
        help="Directory to store generated plots",
    )

    args = parser.parse_args()
    data_path = Path(args.data)
    model_dir = Path(args.model_dir)
    plot_dir = Path(args.plot_dir)

    ensure_directories(model_dir, plot_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    original_columns = df.columns.tolist()

    rename_map = {col: normalize_column_name(col) for col in df.columns}
    df.rename(columns=rename_map, inplace=True)

    detection = detect_columns(list(rename_map.values()))
    target_col = detection["target"]

    if target_col not in df.columns:
        raise KeyError(f"Detected target column '{target_col}' is missing after normalization")

    df = df.dropna(how="all")

    feature_columns = [col for col in df.columns if col != target_col]
    if not feature_columns:
        raise ValueError("No feature columns found for training")

    generate_plots(df, target_col, plot_dir, detection)

    X = df[feature_columns]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_pipeline(feature_columns, df)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    metrics = evaluate_model(y_test, y_pred)

    model_path = model_dir / "wind_power_model.sav"
    joblib.dump(pipeline, model_path)

    metadata = {
        "training_data": str(data_path.resolve()),
        "feature_columns": feature_columns,
        "target_column": target_col,
        "metrics": metrics,
        "training_timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "original_columns": original_columns,
    }

    metadata_path = model_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Training complete. Metrics:")
    for key, value in metrics.items():
        print(f"  {key.upper()}: {value:.4f}")
    print(f"Model saved to {model_path}")
    print(f"Metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()
