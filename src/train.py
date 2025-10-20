from pathlib import Path
from datetime import datetime, timezone
import json
import joblib
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import argparse

parser = argparse.ArgumentParser(description="Train a diabetes progression model.")
parser.add_argument("--model", choices=["linear", "rf"], default="linear")
args = parser.parse_args()

data = load_diabetes(as_frame=False)
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

if args.model == "linear":
    model = LinearRegression()
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])
    model_name = "LinearRegression"
    version = "0.1"
elif args.model == "rf":
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])
    model_name = "RandomForestRegressor"
    version = "0.2"

pipeline.fit(X_train, y_train)

preds = pipeline.predict(X_test)
rmse = float(root_mean_squared_error(y_test, preds))
print(f"RMSE: {rmse:.2f}")

artifacts_dir = Path("artifacts", model_name)
artifacts_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(pipeline, artifacts_dir / "model.pkl")

feature_importance = None
if args.model == "rf":
    feature_importance = pipeline.named_steps["model"].feature_importances_

metrics = {
    "rmse": rmse,
    "feature_importance": feature_importance.tolist() if feature_importance is not None else [],
    "trained_at": datetime.now(timezone.utc).isoformat()
}

(artifacts_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

meta = {
    "pipeline": model_name,
    "version": version,
    "rmse": rmse,
    "trained_at": datetime.now(timezone.utc).isoformat()
}
(artifacts_dir / "meta.json").write_text(json.dumps(meta, indent=2))

docker_models_dir = Path("models")
docker_models_dir.mkdir(exist_ok=True)

versioned_model_path = docker_models_dir / f"model_v{version}.joblib"
joblib.dump(pipeline, versioned_model_path)

feature_list = data.feature_names
(docker_models_dir / "feature_list.json").write_text(json.dumps(feature_list, indent=2))

print(json.dumps(meta, indent=2))