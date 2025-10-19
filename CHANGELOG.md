## 0.1.0 — Baseline
- Model: StandardScaler + LinearRegression
- Metric (held-out RMSE): recorded in `artifacts/metrics.json`
- Deployed API: `/health`, `/predict`
- Deterministic training via fixed seed and pinned deps

## 0.2.0 — Improvement (planned)
- Option to train Ridge or RandomForest (`--pipeline ridge|rf`)
- Enable calibration with `--calibrate` to output `high_risk` and `risk_probability`
- Record precision/recall at threshold and RMSE deltas in `artifacts/metrics.json`

