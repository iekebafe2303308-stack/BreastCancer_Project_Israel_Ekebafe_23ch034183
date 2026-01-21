"""
Train and persist the breast cancer model and scaler.
This script is safe to run multiple times.
"""

import os
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

SELECTED_FEATURES = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
]

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "breast_cancer_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")


def train_and_save():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load dataset
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["diagnosis"] = data.target  # 1 = Malignant, 0 = Benign

    X = df[SELECTED_FEATURES]
    y = df["diagnosis"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train
    model = LogisticRegression(random_state=42, max_iter=1000, solver="lbfgs")
    model.fit(X_train_scaled, y_train)

    # Save artifacts
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print(f"✓ Model saved at: {MODEL_PATH}")
    print(f"✓ Scaler saved at: {SCALER_PATH}")


def main():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        print("Model artifacts already exist. Skipping training.")
        return

    print("Training model artifacts...")
    train_and_save()


if __name__ == "__main__":
    main()
