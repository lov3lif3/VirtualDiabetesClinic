# src/train.py — v0.2 (final RandomForest version)

from pathlib import Path
from datetime import datetime, timezone
import json
import joblib
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, precision_score, recall_score

# ------------------------------
# Load data
# ------------------------------
data = load_diabetes(as_frame=False)
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------
# Train RandomForest model
# ------------------------------
model = RandomForestRegressor(
    n_estimators=500,
    max_depth=5,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# ------------------------------
# Evaluate
# ------------------------------
preds = model.predict(X_test)
rmse = float(mean_squared_error(y_test, preds, squared=False))
threshold = np.percentile(y_train, 75)
y_test_bin = (y_test > threshold).astype(int)
y_pred_bin = (preds > threshold).astype(int)
precision = precision_score(y_test_bin, y_pred_bin)
recall = recall_score(y_test_bin, y_pred_bin)

print(f"RMSE: {rmse:.3f}")
print(f"Precision: {precision:.3f}, Recall: {recall:.3f}")

# ------------------------------
# Save artifacts
# ------------------------------
outdir = Path("artifacts") / "0.2.0"
outdir.mkdir(parents=True, exist_ok=True)

joblib.dump(model, outdir / "model.joblib")

meta = {
    "pipeline": "rf",
    "version": "0.2.0",
    "rmse": rmse,
    "precision": precision,
    "recall": recall,
    "trained_at": datetime.now(timezone.utc).isoformat()
}
(outdir / "meta.json").write_text(json.dumps(meta, indent=2))

print("Saved model and metadata to artifacts/0.2.0/")