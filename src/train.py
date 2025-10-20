from pathlib import Path
from datetime import datetime, timezone
import json
import joblib
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load data
data = load_diabetes(as_frame=False)
X, y = data.data, data.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

# Train
pipeline.fit(X_train, y_train)

# Evaluate
preds = pipeline.predict(X_test)
rmse = float(mean_squared_error(y_test, preds, squared=False))
print(f"RMSE: {rmse:.2f}")

# Save artifacts
artifacts_dir = Path("artifacts")
artifacts_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(pipeline, artifacts_dir / "model.joblib")

meta = {
    "pipeline": "baseline",
    "version": "0.1.0",
    "rmse": rmse,
    "trained_at": datetime.now(timezone.utc).isoformat()
}
(artifacts_dir / "meta.json").write_text(json.dumps(meta, indent=2))

print(json.dumps(meta, indent=2))

# Load old model
pipeline = joblib.load("artifacts/model.joblib")

# Example input
x_example = np.array([[0.02, -0.044, 0.06, -0.03, -0.02, 0.03, -0.02, 0.02, 0.02, -0.001]])
pred = pipeline.predict(x_example)
print(pred)