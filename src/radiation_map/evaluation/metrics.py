"""Evaluation metrics for spatial regression and anomaly detection."""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from radiation_map.config import REFERENCE_SITES
from radiation_map.data.transforms import (
    log1p_target_transform,
    inverse_log1p_target_transform,
)


@dataclass
class SpatialEvaluationReport:
    r2: float
    rmse_usv: float
    mae_usv: float
    medae_usv: float
    rmse_log: float
    mae_log: float
    picp_95: float
    mpiw_usv: float
    sample_count: int

    def to_dict(self) -> dict:
        return {
            "r2": round(self.r2, 4),
            "rmse_usv": round(self.rmse_usv, 5),
            "mae_usv": round(self.mae_usv, 5),
            "medae_usv": round(self.medae_usv, 5),
            "rmse_log": round(self.rmse_log, 5),
            "mae_log": round(self.mae_log, 5),
            "picp_95": round(self.picp_95, 4),
            "mpiw_usv": round(self.mpiw_usv, 5),
            "sample_count": self.sample_count,
        }


def evaluate_spatial_regression(
    y_true_usv: np.ndarray,
    y_pred_usv: np.ndarray,
    ci_lower_usv: Optional[np.ndarray] = None,
    ci_upper_usv: Optional[np.ndarray] = None,
) -> SpatialEvaluationReport:
    """Calculate comprehensive spatial regression and uncertainty calibration metrics."""
    y_true = np.asarray(y_true_usv, dtype=np.float64)
    y_pred = np.asarray(y_pred_usv, dtype=np.float64)
    
    # Standard physical metrics
    rmse_usv = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae_usv = float(mean_absolute_error(y_true, y_pred))
    medae_usv = float(np.median(np.abs(y_true - y_pred)))
    
    # Log metrics
    y_true_log = log1p_target_transform(y_true)
    y_pred_log = log1p_target_transform(y_pred)
    
    r2 = float(r2_score(y_true_log, y_pred_log))
    rmse_log = float(np.sqrt(mean_squared_error(y_true_log, y_pred_log)))
    mae_log = float(mean_absolute_error(y_true_log, y_pred_log))
    
    # Uncertainty calibration
    if ci_lower_usv is not None and ci_upper_usv is not None:
        ci_low = np.asarray(ci_lower_usv, dtype=np.float64)
        ci_high = np.asarray(ci_upper_usv, dtype=np.float64)
        
        # Prediction Interval Coverage Probability (PICP)
        covered = (y_true >= ci_low) & (y_true <= ci_high)
        picp_95 = float(np.mean(covered))
        
        # Mean Prediction Interval Width (MPIW)
        mpiw_usv = float(np.mean(ci_high - ci_low))
    else:
        picp_95 = 0.0
        mpiw_usv = 0.0
        
    return SpatialEvaluationReport(
        r2=r2,
        rmse_usv=rmse_usv,
        mae_usv=mae_usv,
        medae_usv=medae_usv,
        rmse_log=rmse_log,
        mae_log=mae_log,
        picp_95=picp_95,
        mpiw_usv=mpiw_usv,
        sample_count=len(y_true),
    )


def evaluate_hotspot_benchmarks(
    model,
    detector=None,
    reference_sites: Optional[Dict[str, dict]] = None,
) -> Dict[str, dict]:
    """Evaluate predictions against known global landmark sites and hotspots."""
    sites = reference_sites or REFERENCE_SITES
    results = {}
    
    for site_name, site_info in sites.items():
        lat = site_info["latitude"]
        lon = site_info["longitude"]
        cat = site_info.get("category", "unknown")
        
        if hasattr(model, "predict_radiation"):
            pred = model.predict_radiation(lat, lon)
            pred_usv = float(pred["radiation_usv"])
            unc_usv = float(pred["uncertainty_usv"])
            ci_low = float(pred["ci_lower_usv"])
            ci_high = float(pred["ci_upper_usv"])
        elif hasattr(model, "predict"):
            pred = model.predict(lat, lon)
            pred_usv = float(pred["radiation_usv"])
            ci_low = float(pred["ci_lower_usv"])
            ci_high = float(pred["ci_upper_usv"])
            unc_usv = (ci_high - ci_low) / 3.92
        else:
            pred_usv = 0.08
            unc_usv = 0.02
            ci_low = 0.04
            ci_high = 0.12
            
        site_res = {
            "name": site_name,
            "category": cat,
            "latitude": lat,
            "longitude": lon,
            "predicted_usv": round(pred_usv, 5),
            "uncertainty_usv": round(unc_usv, 5),
            "ci_lower_usv": round(ci_low, 5),
            "ci_upper_usv": round(ci_high, 5),
            "description": site_info.get("description", ""),
        }
        
        if detector is not None:
            # Test anomaly detection with a hypothetical hotspot surge (5.0 uSv/h)
            test_surge = detector.detect(lat, lon, observed_value=5.0, unit="usv")
            # Test anomaly detection with normal background (0.08 uSv/h)
            test_normal = detector.detect(lat, lon, observed_value=0.08, unit="usv")
            site_res["test_surge_severity"] = test_surge.severity.value
            site_res["test_surge_z"] = round(test_surge.z_score, 2)
            site_res["test_normal_severity"] = test_normal.severity.value
            
        results[site_name] = site_res
        
    return results
