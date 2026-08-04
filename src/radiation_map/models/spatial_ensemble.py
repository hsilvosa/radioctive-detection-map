"""Spatial ensemble regressor with tree-based gradient boosting and spatial KNN."""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Union
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neighbors import NearestNeighbors
from radiation_map.data.transforms import (
    latlon_to_cartesian,
    SpatialFeatureTransformer,
    log1p_target_transform,
    inverse_log1p_target_transform,
)


class SpatialEnsembleRegressor:
    """Spatial Regressor combining Fourier features, HistGradientBoosting, and local KNN."""
    
    def __init__(
        self,
        num_frequencies: int = 24,
        fourier_scale: float = 10.0,
        max_iter: int = 150,
        learning_rate: float = 0.08,
        random_state: int = 42,
    ):
        self.transformer = SpatialFeatureTransformer(
            num_fourier_frequencies=num_frequencies,
            fourier_scale=fourier_scale,
            seed=random_state,
        )
        self.gbr_model = HistGradientBoostingRegressor(
            max_iter=max_iter,
            learning_rate=learning_rate,
            min_samples_leaf=20,
            random_state=random_state,
            loss="squared_error",
        )
        self.knn = NearestNeighbors(n_neighbors=5, algorithm="ball_tree", metric="haversine")
        self.fitted = False
        self.conformal_q95 = 0.15
        
    def fit(
        self,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
        targets_usv: np.ndarray,
    ):
        """Fit spatial transformer and gradient boosting regressor on log1p targets."""
        lat = np.asarray(latitudes, dtype=np.float64)
        lon = np.asarray(longitudes, dtype=np.float64)
        
        # Transform spatial features
        X_feat = self.transformer.transform(lat, lon)
        y_log = log1p_target_transform(targets_usv)
        
        # Fit gradient booster
        self.gbr_model.fit(X_feat, y_log)
        
        # Fit haversine KNN on radians
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        self.knn.fit(np.column_stack([lat_rad, lon_rad]))
        
        # Compute empirical residual quantile for conformal prediction intervals
        y_pred_log = self.gbr_model.predict(X_feat)
        residuals = np.abs(y_log - y_pred_log)
        self.conformal_q95 = float(np.quantile(residuals, 0.95))
        self.fitted = True
        return self
        
    def predict(
        self,
        latitudes: Union[float, np.ndarray],
        longitudes: Union[float, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Predict expected radiation in physical uSv/h scale and conformal bounds."""
        if not self.fitted:
            raise ValueError("Model must be fitted before calling predict.")
            
        lat = np.asarray(latitudes, dtype=np.float64)
        lon = np.asarray(longitudes, dtype=np.float64)
        
        is_scalar = np.isscalar(latitudes) and np.isscalar(longitudes)
        if is_scalar:
            lat = np.array([lat])
            lon = np.array([lon])
            
        X_feat = self.transformer.transform(lat, lon)
        mu_log = self.gbr_model.predict(X_feat)
        
        radiation_usv = inverse_log1p_target_transform(mu_log)
        lower_log = np.maximum(mu_log - self.conformal_q95, 0.0)
        upper_log = mu_log + self.conformal_q95
        
        radiation_usv_low = inverse_log1p_target_transform(lower_log)
        radiation_usv_high = inverse_log1p_target_transform(upper_log)
        
        if is_scalar:
            return {
                "radiation_usv": float(radiation_usv[0]),
                "ci_lower_usv": float(radiation_usv_low[0]),
                "ci_upper_usv": float(radiation_usv_high[0]),
                "mu_log": float(mu_log[0]),
            }
            
        return {
            "radiation_usv": radiation_usv,
            "ci_lower_usv": radiation_usv_low,
            "ci_upper_usv": radiation_usv_high,
            "mu_log": mu_log,
        }
