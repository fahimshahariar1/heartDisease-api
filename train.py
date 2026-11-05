"""Train a heart-disease classifier and save it for the API.

Usage:
    python train.py                       # expects data/heart.csv
    python train.py --data path/to/heart.csv

Outputs:
    model/heart_model.joblib   the trained scikit-learn model
    model/metadata.json        feature order, metrics and versions (served by GET /info)
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

# Column order the model is trained on. The API reads this back from metadata.json,
# so training and serving always use the same order.
FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
TARGET = "target"
MODEL_DIR = Path("model")


def main(data_path: str) -> None:
    df = pd.read_csv(data_path)

    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise SystemExit(f"Dataset is missing columns: {sorted(missing)}")

    # This dataset contains many duplicate rows. Keeping them would put the same
    # patient in both train and test sets and inflate the test score.
    n_before = len(df)
    df = df.drop_duplicates()
    print(f"Rows: {n_before} -> {len(df)} after removing duplicates")

    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    # Random Forest: no feature scaling needed, robust default for tabular data
    model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"Test accuracy: {accuracy:.4f} | ROC-AUC: {roc_auc:.4f}")
    print(classification_report(y_test, y_pred, digits=3))

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "heart_model.joblib")

    metadata = {
        "model_type": type(model).__name__,
        "features": FEATURES,
        "target_description": "1 = heart disease present, 0 = absent",
        "metrics": {"test_accuracy": round(accuracy, 4), "test_roc_auc": round(roc_auc, 4)},
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved model to {MODEL_DIR / 'heart_model.joblib'} and metadata to {MODEL_DIR / 'metadata.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the heart disease model")
    parser.add_argument("--data", default="data/heart.csv", help="Path to heart.csv")
    main(parser.parse_args().data)
