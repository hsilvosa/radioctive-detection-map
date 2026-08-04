"""Real-time radioactive anomaly detection and uncertainty-aware statistical testing."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Union, List
import numpy as np
from scipy.stats import norm
from radiation_map.config import (
    CPM_TO_USV_CONVERSION_FACTOR,
    GLOBAL_BACKGROUND_MEAN_USV,
    AnomalyThresholds,
)
from radiation_map.data.transforms import (
    log1p_target_transform,
    inverse_log1p_target_transform,
)


class AnomalySeverity(str, Enum):
    NORMAL = "NORMAL"
    ELEVATED = "ELEVATED"
    ANOMALY_HIGH = "ANOMALY_HIGH"
    ANOMALY_CRITICAL = "ANOMALY_CRITICAL"


@dataclass
class AnomalyDetectionResult:
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
    severity: AnomalySeverity
    is_anomaly: bool
    description: str

    def to_dict(self) -> dict:
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "observed_usv": round(self.observed_usv, 5),
            "expected_baseline_usv": round(self.expected_baseline_usv, 5),
            "uncertainty_usv": round(self.uncertainty_usv, 5),
            "ci_lower_usv": round(self.ci_lower_usv, 5),
            "ci_upper_usv": round(self.ci_upper_usv, 5),
            "z_score": round(self.z_score, 3),
            "p_value": float(f"{self.p_value:.3e}"),
            "fold_increase": round(self.fold_increase, 2),
            "severity": self.severity.value,
            "is_anomaly": self.is_anomaly,
            "description": self.description,
        }


class RadiationAnomalyDetector:
    """Real-time radioactive anomaly detector comparing sensor observations against learned spatial baselines."""

    def __init__(
        self,
        model,
        thresholds: Optional[AnomalyThresholds] = None,
        device=None,
    ):
        self.model = model
        self.thresholds = thresholds or AnomalyThresholds()
        self.device = device

    def detect(
        self,
        latitude: float,
        longitude: float,
        observed_value: float,
        unit: str = "usv",
        num_mc_samples: int = 1,
    ) -> AnomalyDetectionResult:
        """Evaluate a measurement at a given coordinate for radioactive anomalies."""
        # Standardize unit to uSv/h
        unit_clean = unit.lower().strip()
        if unit_clean in ["cpm", "counts per minute"]:
            observed_usv = float(observed_value) * CPM_TO_USV_CONVERSION_FACTOR
        else:
            observed_usv = float(observed_value)

        # Get spatial prediction from model
        if hasattr(self.model, "predict_radiation"):
            pred = self.model.predict_radiation(
                latitude, longitude, num_mc_samples=num_mc_samples, device=self.device
            )
            expected_usv = float(pred["radiation_usv"])
            uncertainty_usv = float(pred["uncertainty_usv"])
            ci_low = float(pred["ci_lower_usv"])
            ci_high = float(pred["ci_upper_usv"])
            mu_log = float(pred["mu_log"])
            sigma_log = float(pred["sigma_log"])
        elif hasattr(self.model, "predict"):
            pred = self.model.predict(latitude, longitude)
            expected_usv = float(pred["radiation_usv"])
            ci_low = float(pred["ci_lower_usv"])
            ci_high = float(pred["ci_upper_usv"])
            mu_log = float(pred["mu_log"])
            sigma_log = 0.15
            uncertainty_usv = max((ci_high - ci_low) / 3.92, 0.01)
        else:
            # Fallback baseline
            expected_usv = GLOBAL_BACKGROUND_MEAN_USV
            ci_low = 0.04
            ci_high = 0.16
            mu_log = float(log1p_target_transform(expected_usv))
            sigma_log = 0.2
            uncertainty_usv = 0.03

        # Compute log-space observed value
        obs_log = float(log1p_target_transform(observed_usv))
        
        # Calculate Z-score
        sigma_safe = max(sigma_log, 1e-4)
        z_score = float((obs_log - mu_log) / sigma_safe)
        
        # Upper-tail p-value for radiation surge
        p_value = float(1.0 - norm.cdf(z_score))
        
        # Physical ratio to baseline
        safe_expected = max(expected_usv, 1e-4)
        fold_increase = float(observed_usv / safe_expected)

        # Determine severity classification
        if z_score < self.thresholds.normal_z_max:
            severity = AnomalySeverity.NORMAL
            is_anomaly = False
            desc = "Radiation level is within standard local environmental background."
        elif z_score < self.thresholds.elevated_z_max:
            severity = AnomalySeverity.ELEVATED
            is_anomaly = True
            desc = f"Slightly elevated radiation detected ({fold_increase:.1f}x baseline). Possible altitude or radon variation."
        elif z_score < self.thresholds.anomaly_high_z_max:
            severity = AnomalySeverity.ANOMALY_HIGH
            is_anomaly = True
            desc = f"Significant radioactive anomaly ({fold_increase:.1f}x baseline, Z={z_score:.1f}). Localized contamination or source."
        else:
            severity = AnomalySeverity.ANOMALY_CRITICAL
            is_anomaly = True
            desc = f"Critical radiological alert ({fold_increase:.1f}x baseline, Z={z_score:.1f}). Dangerous radioactive intensity."

        return AnomalyDetectionResult(
            latitude=latitude,
            longitude=longitude,
            observed_usv=observed_usv,
            expected_baseline_usv=expected_usv,
            uncertainty_usv=uncertainty_usv,
            ci_lower_usv=ci_low,
            ci_upper_usv=ci_high,
            z_score=z_score,
            p_value=p_value,
            fold_increase=fold_increase,
            severity=severity,
            is_anomaly=is_anomaly,
            description=desc,
        )

    def detect_batch(
        self,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
        observed_values: np.ndarray,
        units: Union[str, List[str]] = "usv",
    ) -> List[AnomalyDetectionResult]:
        """Vectorized evaluation over multiple coordinate points."""
        results = []
        lats = np.asarray(latitudes)
        lons = np.asarray(longitudes)
        obs = np.asarray(observed_values)
        
        if isinstance(units, str):
            unit_list = [units] * len(lats)
        else:
            unit_list = units
            
        for i in range(len(lats)):
            res = self.detect(
                latitude=float(lats[i]),
                longitude=float(lons[i]),
                observed_value=float(obs[i]),
                unit=unit_list[i],
            )
            results.append(res)
        return results
