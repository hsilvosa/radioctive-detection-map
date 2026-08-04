"""Unit tests for model forward passes, loss, and anomaly detection."""

import numpy as np
import pytest
import torch

from radiation_map.models.spatial_net import SpatialHarmonicNet
from radiation_map.models.spatial_ensemble import SpatialEnsembleRegressor
from radiation_map.models.anomaly_detector import RadiationAnomalyDetector, AnomalySeverity
from radiation_map.training.loss import HeteroscedasticGaussianLoss


def test_spatial_harmonic_net_forward():
    model = SpatialHarmonicNet(
        in_dim=3,
        num_frequencies=16,
        hidden_dims=(64, 32),
        dropout=0.0,
    )
    model.eval()
    
    # 3D Cartesian coordinates on sphere
    coords = torch.randn(8, 3)
    coords = coords / torch.norm(coords, dim=-1, keepdim=True)
    
    mu, log_var = model(coords)
    assert mu.shape == (8, 1)
    assert log_var.shape == (8, 1)
    assert not torch.isnan(mu).any()
    assert not torch.isnan(log_var).any()


def test_heteroscedastic_loss():
    loss_fn = HeteroscedasticGaussianLoss()
    mu = torch.tensor([[0.5], [1.0]], requires_grad=True)
    log_var = torch.tensor([[0.0], [-1.0]], requires_grad=True)
    target = torch.tensor([[0.5], [1.2]])
    
    loss = loss_fn(mu, log_var, target)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert mu.grad is not None
    assert log_var.grad is not None


def test_anomaly_detector_severities():
    model = SpatialHarmonicNet(num_frequencies=8, hidden_dims=(32, 16))
    detector = RadiationAnomalyDetector(model)
    
    # Normal background reading
    res_normal = detector.detect(35.68, 139.69, observed_value=0.08, unit="usv")
    assert isinstance(res_normal.severity, AnomalySeverity)
    
    # Extreme radioactive spike (50.0 uSv/h)
    res_critical = detector.detect(35.68, 139.69, observed_value=50.0, unit="usv")
    assert res_critical.is_anomaly is True
    assert res_critical.severity in [AnomalySeverity.ANOMALY_HIGH, AnomalySeverity.ANOMALY_CRITICAL]
    assert res_critical.z_score > 4.0


def test_spatial_ensemble_regressor():
    ensemble = SpatialEnsembleRegressor(num_frequencies=8, max_iter=20)
    lats = np.array([35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0])
    lons = np.array([139.0, 140.0, 141.0, 142.0, 143.0, 144.0, 145.0, 146.0])
    targets = np.array([0.08, 0.09, 0.50, 0.12, 0.07, 0.08, 0.09, 0.10])
    
    ensemble.fit(lats, lons, targets)
    pred = ensemble.predict(37.0, 141.0)
    assert "radiation_usv" in pred
    assert pred["radiation_usv"] > 0
    assert pred["ci_lower_usv"] <= pred["radiation_usv"] <= pred["ci_upper_usv"]
