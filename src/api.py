from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pathlib import Path
from contextlib import asynccontextmanager
import os, json, joblib, numpy as np

ART_DIR = Path("artifacts")
MODEL_PATH = ART_DIR / "model.joblib"
META_PATH = ART_DIR / "meta.json"

class PredictRequest(BaseModel):
    features: list[float] | None = Field(default=None)
    age: float | None = None
    sex: float | None = None
    bmi: float | None = None
    bp: float | None = None
    s1: float | None = None
    s2: float | None = None
    s3: float | None = None
    s4: float | None = None
    s5: float | None = None
    s6: float | None = None

    def to_feature_list(self) -> list[float]:
        if self.features is not None:
            if len(self.features) != 10:
                raise ValueError(f"Expected 10 features, got {len(self.features)}")
            return [float(x) for x in self.features]
        ordered = [self.age, self.sex, self.bmi, self.bp, self.s1, self.s2, self.s3, self.s4, self.s5, self.s6]
        if any(v is None for v in ordered):
            raise ValueError("Provide all named features (age, sex, bmi, bp, s1..s6) or a 10-length 'features' list.")
        return [float(x) for x in ordered]

class PredictResponse(BaseModel):
    prediction: float
    model_version: str
    model_config = {"protected_namespaces": ()}

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = None
    app.state.meta = {"version": "unknown"}
    try:
        if MODEL_PATH.exists():
            app.state.model = joblib.load(MODEL_PATH)
        if META_PATH.exists():
            app.state.meta = json.loads(META_PATH.read_text())
    except Exception:
        app.state.model = None
        app.state.meta = {"version": "unknown"}
    yield

app = FastAPI(title="Virtual Diabetes Clinic Triage", version="0.1.0", lifespan=lifespan)

def get_model(req: Request):
    mdl = getattr(req.app.state, "model", None)
    if mdl is None:
        if os.getenv("ALLOW_DUMMY_MODEL", "1") == "1":
            return lambda X: np.array([0.0])
        raise HTTPException(status_code=503, detail="Model not loaded")
    return mdl

@app.get("/health")
def health(req: Request):
    return {"status": "ok", "model_version": req.app.state.meta.get("version", "unknown")}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, model=Depends(get_model), app_req: Request = None):
    try:
        vec = req.to_feature_list()
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    X = np.asarray(vec, dtype=float).reshape(1, -1)
    try:
        y = float(model.predict(X)[0])
    except AttributeError:
        y = float(model(X)[0])
    return PredictResponse(prediction=y, model_version=getattr(app_req.app.state, "meta", {}).get("version", "unknown"))

@app.exception_handler(ValueError)
async def _value_error_handler(_, exc: ValueError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})
