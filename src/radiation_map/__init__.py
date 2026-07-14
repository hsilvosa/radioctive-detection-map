"""Global Radiation Anomaly Map & Spatial Regressor.

A spatial regression and real-time radioactive anomaly detection system
trained on global historical Safecast radiation measurements.
"""

__version__ = "0.1.0"

from radiation_map.models.spatial_net import SpatialHarmonicNet
from radiation_map.models.anomaly_detector import RadiationAnomalyDetector
from radiation_map.inference.predictor import SpatialRadiationPredictor

__all__ = [
    "SpatialHarmonicNet",
    "RadiationAnomalyDetector",
    "SpatialRadiationPredictor",
]
