"""Unit tests for predictor and batch inference."""

import numpy as np
import pytest

from radiation_map.inference.predictor import SpatialRadiationPredictor
from radiation_map.models.spatial_net import SpatialHarmonicNet


def test_predictor_point_and_batch():
    model = SpatialHarmonicNet(num_frequencies=8, hidden_dims=(32, 16))
    predictor = SpatialRadiationPredictor(model=model)
    
    # Point prediction
    pt = predictor.predict_point(latitude=35.6895, longitude=139.6917)
    assert "radiation_usv" in pt
    assert "uncertainty_usv" in pt
    assert "ci_lower_usv" in pt
    assert "ci_upper_usv" in pt
    assert pt["ci_lower_usv"] <= pt["radiation_usv"] <= pt["ci_upper_usv"]

    # Batch prediction
    lats = [35.6895, 51.5074, 40.7128]
    lons = [139.6917, -0.1278, -74.0060]
    batch_res = predictor.predict_batch(lats, lons)
    assert len(batch_res) == 3
    for item in batch_res:
        assert item["radiation_usv"] >= 0.0


def test_predictor_anomaly_detection():
    model = SpatialHarmonicNet(num_frequencies=8, hidden_dims=(32, 16))
    predictor = SpatialRadiationPredictor(model=model)
    
    result = predictor.detect_anomaly(
        latitude=51.3896,
        longitude=30.0999,
        observed_value=12.5,
        unit="usv"
    )
    assert result.observed_usv == 12.5
    assert result.is_anomaly is True
    assert result.fold_increase > 1.0
