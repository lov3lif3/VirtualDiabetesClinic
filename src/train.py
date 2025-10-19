from pathlib import Path
from datetime import datetime, timezone
import json
import joblib
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import argparse

# Parse arguments for model choice
parser = argparse.ArgumentParser(description="Train a diabetes progression model.")
parser.add_argument("--model", choices=["linear", "rf"], default="linear", help="Choose the model: 'linear' or 'rf' (RandomForest)")
args = parser.parse_args()

# Load data
data = load_diabetes(as_frame=False)
X, y = data.data, data.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define pipeline and model based on input argument
if args.model == "linear":
    model = LinearRegression()
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])
    model_name = "LinearRegression"
    version = "0.1.0"
elif args.model == "rf":
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])
    model_name = "RandomForestRegressor"
    version = "0.2.0"

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate
preds = pipeline.predict(X_test)
rmse = float(mean_squared_error(y_test, preds, squared=False))
print(f"RMSE: {rmse:.2f}")

# Save artifacts
artifacts_dir = Path("artifacts", model_name)
artifacts_dir.mkdir(parents=True, exist_ok=True)

# Save the trained model
joblib.dump(pipeline, artifacts_dir / "model.pkl")

# Log feature importance for RandomForest (optional)
feature_importance = None
if args.model == "rf":
    feature_importance = pipeline.named_steps["model"].feature_importances_

# Save metrics (RMSE, Feature Importance)
metrics = {
    "rmse": rmse,
    "feature_importance": feature_importance.tolist() if feature_importance is not None else [],
    "trained_at": datetime.now(timezone.utc).isoformat()
}

# Save metrics to JSON
(artifacts_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

# Save metadata
meta = {
    "pipeline": model_name,
    "version": version,
    "rmse": rmse,
    "trained_at": datetime.now(timezone.utc).isoformat()
}
(artifacts_dir / "meta.json").write_text(json.dumps(meta, indent=2))

print(json.dumps(meta, indent=2))
