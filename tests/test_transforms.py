"""Unit tests for spatial coordinate and unit transformations."""

import numpy as np
import pandas as pd
import pytest

from radiation_map.data.transforms import (
    latlon_to_cartesian,
    cartesian_to_latlon,
    normalize_measurement_units,
    log1p_target_transform,
    inverse_log1p_target_transform,
    SpatialFeatureTransformer,
)
from radiation_map.config import CPM_TO_USV_CONVERSION_FACTOR


def test_latlon_to_cartesian_and_inverse():
    # Test equator and prime meridian
    xyz = latlon_to_cartesian(0.0, 0.0)
    assert np.allclose(xyz, [1.0, 0.0, 0.0], atol=1e-5)
    lat, lon = cartesian_to_latlon(xyz)
    assert np.isclose(lat, 0.0, atol=1e-4)
    assert np.isclose(lon, 0.0, atol=1e-4)

    # Test North Pole
    xyz_pole = latlon_to_cartesian(90.0, 0.0)
    assert np.isclose(xyz_pole[2], 1.0, atol=1e-5)
    lat, _ = cartesian_to_latlon(xyz_pole)
    assert np.isclose(lat, 90.0, atol=1e-4)

    # Vectorized test
    lats = np.array([35.6895, 48.8566, -33.8688])
    lons = np.array([139.6917, 2.3522, 151.2093])
    xyz_vec = latlon_to_cartesian(lats, lons)
    assert xyz_vec.shape == (3, 3)
    rec_lats, rec_lons = cartesian_to_latlon(xyz_vec)
    assert np.allclose(lats, rec_lats, atol=1e-3)
    assert np.allclose(lons, rec_lons, atol=1e-3)


def test_log1p_target_transform_invertibility():
    values = np.array([0.05, 0.08, 0.15, 1.25, 10.0, 50.0])
    transformed = log1p_target_transform(values)
    recovered = inverse_log1p_target_transform(transformed)
    assert np.allclose(values, recovered, atol=1e-6)


def test_normalize_measurement_units():
    raw_data = pd.DataFrame({
        "latitude": [35.6, 51.5, 95.0, 40.7, 37.4],  # 95.0 is invalid lat
        "longitude": [139.7, -0.1, 10.0, -74.0, 141.0],
        "value": [33.4, 0.08, 10.0, -5.0, 1000.0],   # -5.0 is invalid value
        "unit": ["cpm", "usv", "cpm", "usv", "cpm"],
    })
    
    clean_df = normalize_measurement_units(raw_data)
    assert len(clean_df) == 3
    assert "radiation_usv" in clean_df.columns
    # Check CPM conversion for 33.4 CPM: approx 33.4 * 0.002994 ~ 0.100 uSv/h
    cpm_row = clean_df[clean_df["latitude"] == 35.6].iloc[0]
    assert np.isclose(cpm_row["radiation_usv"], 33.4 * CPM_TO_USV_CONVERSION_FACTOR, atol=1e-4)


def test_spatial_feature_transformer():
    transformer = SpatialFeatureTransformer(num_fourier_frequencies=16, fourier_scale=8.0)
    assert transformer.output_dim == 3 + 2 * 16
    
    features = transformer.transform(35.6895, 139.6917)
    assert features.shape == (1, 35)
