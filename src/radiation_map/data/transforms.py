"""Spatial feature engineering, coordinate transforms, and unit normalization."""

import numpy as np
import pandas as pd
from typing import Tuple, Union, Optional
from radiation_map.config import (
    CPM_TO_USV_CONVERSION_FACTOR,
    LAT_RANGE,
    LON_RANGE,
)


def latlon_to_cartesian(
    lat: Union[float, np.ndarray, pd.Series],
    lon: Union[float, np.ndarray, pd.Series]
) -> np.ndarray:
    """Project latitude/longitude (degrees) to 3D unit sphere Cartesian coordinates (x, y, z).
    
    This avoids the boundary discontinuity at longitude +/-180 and singularities at poles.
    
    x = cos(lat) * cos(lon)
    y = cos(lat) * sin(lon)
    z = sin(lat)
    """
    lat_rad = np.radians(np.asarray(lat, dtype=np.float64))
    lon_rad = np.radians(np.asarray(lon, dtype=np.float64))
    
    cos_lat = np.cos(lat_rad)
    x = cos_lat * np.cos(lon_rad)
    y = cos_lat * np.sin(lon_rad)
    z = np.sin(lat_rad)
    
    if np.isscalar(lat) and np.isscalar(lon):
        return np.array([x.item(), y.item(), z.item()], dtype=np.float32)
    
    return np.column_stack([x, y, z]).astype(np.float32)


def cartesian_to_latlon(
    xyz: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Inverse mapping from 3D unit sphere coordinates (x, y, z) to latitude and longitude."""
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim == 1:
        x, y, z = xyz[0], xyz[1], xyz[2]
        r = np.sqrt(x**2 + y**2 + z**2) + 1e-12
        lat = np.degrees(np.arcsin(np.clip(z / r, -1.0, 1.0)))
        lon = np.degrees(np.arctan2(y, x))
        return float(lat), float(lon)
    else:
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        r = np.sqrt(x**2 + y**2 + z**2) + 1e-12
        lat = np.degrees(np.arcsin(np.clip(z / r, -1.0, 1.0)))
        lon = np.degrees(np.arctan2(y, x))
        return lat.astype(np.float32), lon.astype(np.float32)


def normalize_measurement_units(
    df: pd.DataFrame,
    min_cpm: float = 1.0,
    max_cpm: float = 50000.0,
    min_usv: float = 0.003,
    max_usv: float = 250.0,
) -> pd.DataFrame:
    """Clean records and convert all radiation measurements to standard uSv/h."""
    # Filter valid coordinates
    mask = (
        df["latitude"].notna()
        & df["longitude"].notna()
        & (df["latitude"] >= LAT_RANGE[0])
        & (df["latitude"] <= LAT_RANGE[1])
        & (df["longitude"] >= LON_RANGE[0])
        & (df["longitude"] <= LON_RANGE[1])
        & df["value"].notna()
        & (df["value"] > 0)
    )
    df_clean = df[mask].copy()
    
    # Standardize unit column
    unit_lower = df_clean["unit"].astype(str).str.lower().str.strip()
    
    # Mask for CPM
    is_cpm = unit_lower.isin(["cpm", "counts per minute"]) & (df_clean["value"] >= min_cpm) & (df_clean["value"] <= max_cpm)
    
    # Mask for uSv/h
    is_usv = unit_lower.isin(["usv", "usv/h", "µsv/h", "microsievert/h"]) & (df_clean["value"] >= min_usv) & (df_clean["value"] <= max_usv)
    
    valid_radiation_mask = is_cpm | is_usv
    df_clean = df_clean[valid_radiation_mask].copy()
    
    # Calculate radiation in uSv/h
    cpm_mask = unit_lower[valid_radiation_mask].isin(["cpm", "counts per minute"])
    
    df_clean["radiation_usv"] = df_clean["value"].astype(np.float64)
    df_clean.loc[cpm_mask, "radiation_usv"] = df_clean.loc[cpm_mask, "value"] * CPM_TO_USV_CONVERSION_FACTOR
    
    # Filter out physical outliers
    df_clean = df_clean[
        (df_clean["radiation_usv"] >= min_usv) & (df_clean["radiation_usv"] <= max_usv)
    ].copy()
    
    return df_clean


def log1p_target_transform(y: Union[float, np.ndarray, pd.Series]) -> Union[float, np.ndarray]:
    """Transform radiation uSv/h to log(1 + 10 * y) for stabilized variance and homoscedastic regression."""
    # Scale factor 10.0 provides better numerical gradients in the 0.03 - 0.20 uSv/h range
    return np.log1p(10.0 * np.asarray(y, dtype=np.float64))


def inverse_log1p_target_transform(y_log: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Inverse transform from log space back to original physical uSv/h scale."""
    y_scaled = np.expm1(np.asarray(y_log, dtype=np.float64))
    return np.maximum(y_scaled / 10.0, 0.0)


class SpatialFeatureTransformer:
    """Transforms raw spatial coordinates (lat, lon) into rich spherical multi-scale embeddings."""
    
    def __init__(self, num_fourier_frequencies: int = 32, fourier_scale: float = 12.0, seed: int = 42):
        self.num_fourier_frequencies = num_fourier_frequencies
        self.fourier_scale = fourier_scale
        self.seed = seed
        rng = np.random.RandomState(seed)
        # Random Fourier Feature projection matrix for 3D sphere (3, num_fourier_frequencies)
        self.b_matrix = rng.normal(scale=fourier_scale, size=(3, num_fourier_frequencies)).astype(np.float32)
        
    def transform(self, lat: Union[float, np.ndarray], lon: Union[float, np.ndarray]) -> np.ndarray:
        """Project (lat, lon) to [x, y, z, sin(2*pi*B^T x), cos(2*pi*B^T x)]."""
        xyz = latlon_to_cartesian(lat, lon)
        if xyz.ndim == 1:
            xyz = xyz.reshape(1, 3)
            
        # Compute multi-frequency Fourier projection
        proj = 2.0 * np.pi * np.matmul(xyz, self.b_matrix)
        sin_feat = np.sin(proj)
        cos_feat = np.cos(proj)
        
        # Concatenate 3D sphere coordinate + Fourier sine + Fourier cosine
        features = np.hstack([xyz, sin_feat, cos_feat]).astype(np.float32)
        return features
        
    @property
    def output_dim(self) -> int:
        return 3 + 2 * self.num_fourier_frequencies
