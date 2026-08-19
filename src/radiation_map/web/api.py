"""FastAPI server for Radiation Spatial Regression and Anomaly Detection."""

import json
from pathlib import Path
from typing import List, Optional, Union
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from radiation_map.inference.predictor import SpatialRadiationPredictor
from radiation_map.config import REFERENCE_SITES

app = FastAPI(
    title="Global Radiation Anomaly Map & Spatial Regressor API",
    description="Spatial regression and real-time radioactive anomaly detection API trained on historical Safecast data.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance (lazy loaded)
_predictor: Optional[SpatialRadiationPredictor] = None


def get_predictor() -> SpatialRadiationPredictor:
    global _predictor
    if _predictor is None:
        model_path = Path("checkpoints/best_spatial_harmonic_net.pt")
        config_path = Path("hf_export/config.json")
        _predictor = SpatialRadiationPredictor(
            model_path=model_path if model_path.exists() else None,
            config_path=config_path if config_path.exists() else None,
        )
    return _predictor


# Pydantic Schemas
class PointPredictionRequest(BaseModel):
    latitude: float = Field(..., ge=-90.0, le=90.0, description="Latitude in degrees")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="Longitude in degrees")
    mc_samples: int = Field(1, ge=1, le=50, description="Monte Carlo samples for uncertainty")


class PointPredictionResponse(BaseModel):
    latitude: float
    longitude: float
    radiation_usv: float
    uncertainty_usv: float
    ci_lower_usv: float
    ci_upper_usv: float


class AnomalyDetectionRequest(BaseModel):
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    value: float = Field(..., gt=0.0, description="Observed radiation measurement")
    unit: str = Field("usv", description="Measurement unit: 'usv' or 'cpm'")


class AnomalyDetectionResponse(BaseModel):
    latitude: float
    longitude: float
    observed_usv: float
    expected_baseline_usv: float
    uncertainty_usv: float
    ci_lower_usv: float
    ci_upper_usv: float
    z_score: float
    p_value: float
    fold_increase: float
    severity: str
    is_anomaly: bool
    description: str


class BatchPredictionRequest(BaseModel):
    coordinates: List[dict] = Field(..., description="List of dicts with 'latitude' and 'longitude'")


class RasterGridRequest(BaseModel):
    min_lat: float = Field(-60.0, ge=-90.0, le=90.0)
    max_lat: float = Field(70.0, ge=-90.0, le=90.0)
    min_lon: float = Field(-180.0, ge=-180.0, le=180.0)
    max_lon: float = Field(180.0, ge=-180.0, le=180.0)
    resolution: int = Field(40, ge=10, le=100)


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    static_file = Path(__file__).parent / "static" / "index.html"
    if static_file.exists():
        with open(static_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Radiation Map API is running</h1>")


@app.post("/api/predict", response_model=PointPredictionResponse)
async def predict_point(req: PointPredictionRequest):
    try:
        predictor = get_predictor()
        res = predictor.predict_point(req.latitude, req.longitude, num_mc_samples=req.mc_samples)
        return PointPredictionResponse(**res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/detect", response_model=AnomalyDetectionResponse)
async def detect_anomaly(req: AnomalyDetectionRequest):
    try:
        predictor = get_predictor()
        res = predictor.detect_anomaly(
            latitude=req.latitude,
            longitude=req.longitude,
            observed_value=req.value,
            unit=req.unit,
        )
        return AnomalyDetectionResponse(**res.to_dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/batch_predict")
async def batch_predict(req: BatchPredictionRequest):
    try:
        predictor = get_predictor()
        lats = [c["latitude"] for c in req.coordinates]
        lons = [c["longitude"] for c in req.coordinates]
        results = predictor.predict_batch(lats, lons)
        return {"count": len(results), "predictions": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/hotspots")
async def get_hotspots():
    try:
        predictor = get_predictor()
        sites = predictor.get_reference_landmarks()
        return {"hotspots": sites}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/raster")
async def get_raster(req: RasterGridRequest):
    try:
        predictor = get_predictor()
        raster_data = predictor.generate_grid_raster(
            min_lat=req.min_lat,
            max_lat=req.max_lat,
            min_lon=req.min_lon,
            max_lon=req.max_lon,
            resolution=req.resolution,
        )
        return {
            "lats": raster_data["lats"].tolist(),
            "lons": raster_data["lons"].tolist(),
            "radiation_grid": raster_data["radiation_grid"].tolist(),
            "uncertainty_grid": raster_data["uncertainty_grid"].tolist(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sensor_samples")
async def get_sensor_samples():
    """Return authentic sampled measurements from the Safecast holdout dataset."""
    test_path = Path("data/processed/radiation_test.parquet")
    if not test_path.exists():
        return {"samples": []}
    try:
        import pandas as pd
        df = pd.read_parquet(test_path)
        # Sample 600 points across the dataset with diverse radiation levels
        sample_df = df.sample(n=min(600, len(df)), random_state=42)
        samples = []
        for _, row in sample_df.iterrows():
            samples.append({
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
                "measured_usv": round(float(row["radiation_usv"]), 4),
                "location_name": str(row.get("location_name") or ""),
            })
        return {"samples": samples}
    except Exception as e:
        return {"samples": [], "error": str(e)}


@app.get("/api/world_choropleth")
async def get_world_choropleth():
    """Return high-resolution subnational and regional boundaries with model-predicted baseline radiation."""
    regional_path = Path(__file__).parent / "static" / "regional_radiation.geojson"
    if regional_path.exists():
        from fastapi.responses import FileResponse
        return FileResponse(regional_path, media_type="application/json")
    
    fallback_path = Path(__file__).parent / "static" / "world_radiation.geojson"
    if fallback_path.exists():
        from fastapi.responses import FileResponse
        return FileResponse(fallback_path, media_type="application/json")
    return {"type": "FeatureCollection", "features": []}


@app.get("/api/info")
async def get_info():
    eval_file = Path("checkpoints/evaluation_results.json")
    eval_summary = {}
    if eval_file.exists():
        with open(eval_file, "r", encoding="utf-8") as f:
            eval_summary = json.load(f)
            
    return {
        "model_name": "SpatialHarmonicNet",
        "dataset": "Safecast Historical Radiation Measurements (265M+ records)",
        "task": "tabular-regression / spatial-ml",
        "unit": "uSv/h (microsieverts per hour)",
        "calibration_factor": "1 CPM = 0.002994 uSv/h (Cs-137 / LND 7317)",
        "evaluation_summary": eval_summary,
    }
