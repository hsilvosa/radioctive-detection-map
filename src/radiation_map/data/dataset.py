"""PyTorch Dataset and spatial batching for radiation regression."""

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from radiation_map.data.transforms import (
    latlon_to_cartesian,
    log1p_target_transform,
)


class SpatialRadiationDataset(Dataset):
    """PyTorch Dataset for spatial coordinates and radiation target values."""
    
    def __init__(
        self,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
        targets_usv: Optional[np.ndarray] = None,
        transform_target: bool = True
    ):
        self.coords_cartesian = torch.tensor(
            latlon_to_cartesian(latitudes, longitudes),
            dtype=torch.float32
        )
        self.raw_coords = torch.tensor(
            np.column_stack([latitudes, longitudes]),
            dtype=torch.float32
        )
        self.transform_target = transform_target
        
        if targets_usv is not None:
            if transform_target:
                targets_log = log1p_target_transform(targets_usv)
                self.targets = torch.tensor(targets_log, dtype=torch.float32).unsqueeze(-1)
            else:
                self.targets = torch.tensor(targets_usv, dtype=torch.float32).unsqueeze(-1)
            self.raw_targets = torch.tensor(targets_usv, dtype=torch.float32).unsqueeze(-1)
        else:
            self.targets = None
            self.raw_targets = None
            
    def __len__(self) -> int:
        return len(self.coords_cartesian)
        
    def __getitem__(self, idx: int) -> dict:
        item = {
            "coords_cartesian": self.coords_cartesian[idx],
            "raw_coords": self.raw_coords[idx]
        }
        if self.targets is not None:
            item["target"] = self.targets[idx]
            item["raw_target"] = self.raw_targets[idx]
        return item
