"""Unified predictor for Spatial Radiation Regressors and Anomaly Detection."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from radiation_map.models.spatial_net import SpatialHarmonicNet
from radiation_map.models.anomaly_detector import RadiationAnomalyDetector, AnomalyDetectionResult
from radiation_map.config import REFERENCE_SITES


class SpatialRadiationPredictor:
    """Production predictor serving spatial radiation estimates and anomaly alerts."""

    def __init__(
        self,
        model: Optional[SpatialHarmonicNet] = None,
        model_path: Optional[Union[str, Path]] = None,
        config_path: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        if model is not None:
            self.model = model.to(self.device)
        elif model_path is not None:
            self.model = self._load_model(model_path, config_path)
        else:
            # Initialize default architecture
            self.model = SpatialHarmonicNet().to(self.device)

        self.model.eval()
        self.detector = RadiationAnomalyDetector(self.model, device=self.device)

    def _load_model(self, model_path: Union[str, Path], config_path: Optional[Union[str, Path]]) -> SpatialHarmonicNet:
        model_path = Path(model_path)
        
        # Load config if available
        if config_path and Path(config_path).exists():
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            num_freq = cfg.get("num_frequencies", 32)
            max_freq = cfg.get("max_frequency_log", 4.5)
            hidden_dims = tuple(cfg.get("hidden_dims", [256, 256, 128, 64]))
            model = SpatialHarmonicNet(
                num_frequencies=num_freq,
                max_frequency_log=max_freq,
                hidden_dims=hidden_dims,
            )
        else:
            model = SpatialHarmonicNet()

        # Load weights
        if model_path.suffix == ".safetensors":
            from safetensors.torch import load_file
            state_dict = load_file(str(model_path))
            model.load_state_dict(state_dict)
        else:
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            elif isinstance(checkpoint, dict):
                model.load_state_dict(checkpoint)
            else:
                model = checkpoint

        return model.to(self.device)

    def predict_point(
        self,
        latitude: float,
        longitude: float,
        num_mc_samples: int = 1,
    ) -> Dict[str, float]:
        """Predict radiation baseline and confidence interval at a single coordinate."""
        pred = self.model.predict_radiation(
            latitude, longitude, num_mc_samples=num_mc_samples, device=self.device
        )
        return {
            "latitude": float(latitude),
            "longitude": float(longitude),
            "radiation_usv": round(float(pred["radiation_usv"]), 5),
            "uncertainty_usv": round(float(pred["uncertainty_usv"]), 5),
            "ci_lower_usv": round(float(pred["ci_lower_usv"]), 5),
            "ci_upper_usv": round(float(pred["ci_upper_usv"]), 5),
        }

    def predict_batch(
        self,
        latitudes: Union[List[float], np.ndarray],
        longitudes: Union[List[float], np.ndarray],
    ) -> List[Dict[str, float]]:
        """Vectorized batch prediction."""
        lats = np.asarray(latitudes, dtype=np.float64)
        lons = np.asarray(longitudes, dtype=np.float64)
        pred = self.model.predict_radiation(lats, lons, device=self.device)
        
        results = []
        for i in range(len(lats)):
            results.append({
                "latitude": float(lats[i]),
                "longitude": float(lons[i]),
                "radiation_usv": round(float(pred["radiation_usv"][i]), 5),
                "uncertainty_usv": round(float(pred["uncertainty_usv"][i]), 5),
                "ci_lower_usv": round(float(pred["ci_lower_usv"][i]), 5),
                "ci_upper_usv": round(float(pred["ci_upper_usv"][i]), 5),
            })
        return results

    def detect_anomaly(
        self,
        latitude: float,
        longitude: float,
        observed_value: float,
        unit: str = "usv",
    ) -> AnomalyDetectionResult:
        """Run real-time anomaly check."""
        return self.detector.detect(
            latitude=latitude,
            longitude=longitude,
            observed_value=observed_value,
            unit=unit,
        )

    def generate_grid_raster(
        self,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float,
        resolution: int = 50,
    ) -> Dict[str, np.ndarray]:
        """Generate 2D spatial grid predictions for heatmaps and isorad contouring."""
        lats = np.linspace(min_lat, max_lat, resolution)
        lons = np.linspace(min_lon, max_lon, resolution)
        grid_lon, grid_lat = np.meshgrid(lons, lats)
        
        flat_lats = grid_lat.ravel()
        flat_lons = grid_lon.ravel()
        
        pred = self.model.predict_radiation(flat_lats, flat_lons, device=self.device)
        
        grid_rad = pred["radiation_usv"].reshape(resolution, resolution)
        grid_unc = pred["uncertainty_usv"].reshape(resolution, resolution)
        
        return {
            "lats": lats,
            "lons": lons,
            "radiation_grid": grid_rad,
            "uncertainty_grid": grid_unc,
        }

    def get_reference_landmarks(self) -> List[dict]:
        """Return list of reference sites with current model predictions."""
        sites = []
        for name, info in REFERENCE_SITES.items():
            pred = self.predict_point(info["latitude"], info["longitude"])
            sites.append({
                "id": name,
                "name": name.replace("_", " "),
                "category": info.get("category", "landmark"),
                "description": info.get("description", ""),
                "latitude": info["latitude"],
                "longitude": info["longitude"],
                "predicted_usv": pred["radiation_usv"],
                "ci_lower_usv": pred["ci_lower_usv"],
                "ci_upper_usv": pred["ci_upper_usv"],
            })
        return sites
