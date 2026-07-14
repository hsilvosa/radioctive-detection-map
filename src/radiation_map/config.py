"""Configuration and physical constants for radiation mapping and modeling."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


# Standard Safecast bGeigie calibration: 1 CPM ~ 0.002994 uSv/h (334 CPM per 1 uSv/h)
CPM_TO_USV_CONVERSION_FACTOR: float = 0.002994

# Normal environmental background radiation range (uSv/h)
GLOBAL_BACKGROUND_MIN_USV: float = 0.03
GLOBAL_BACKGROUND_MEAN_USV: float = 0.08
GLOBAL_BACKGROUND_MAX_USV: float = 0.25

# Coordinate bounds
LAT_RANGE: Tuple[float, float] = (-90.0, 90.0)
LON_RANGE: Tuple[float, float] = (-180.0, 180.0)

# Reference known radiation hotspots and background sites for validation
REFERENCE_SITES: Dict[str, Dict[str, float]] = {
    "Chernobyl_Reactor_4": {
        "latitude": 51.3896,
        "longitude": 30.0999,
        "expected_min_usv": 1.5,
        "category": "hotspot",
        "description": "Chernobyl Nuclear Power Plant Shelter"
    },
    "Pripyat_Red_Forest": {
        "latitude": 51.3833,
        "longitude": 30.0500,
        "expected_min_usv": 1.0,
        "category": "hotspot",
        "description": "Pripyat Red Forest Exclusion Zone"
    },
    "Fukushima_Daiichi": {
        "latitude": 37.4211,
        "longitude": 141.0328,
        "expected_min_usv": 0.8,
        "category": "hotspot",
        "description": "Fukushima Daiichi Nuclear Power Station"
    },
    "Futaba_Town_Fukushima": {
        "latitude": 37.4549,
        "longitude": 141.0101,
        "expected_min_usv": 0.4,
        "category": "hotspot",
        "description": "Futaba Exclusion Zone, Fukushima Prefecture"
    },
    "Ramsar_Talesh_Mahalleh": {
        "latitude": 36.9011,
        "longitude": 50.6558,
        "expected_min_usv": 0.5,
        "category": "natural_high",
        "description": "Ramsar, Iran (high natural radioactivity from radium-rich hot springs)"
    },
    "Guarapari_Beach": {
        "latitude": -20.6714,
        "longitude": -40.4981,
        "expected_min_usv": 0.3,
        "category": "natural_high",
        "description": "Guarapari, Brazil (monazite sand background)"
    },
    "Tokyo_Shinjuku": {
        "latitude": 35.6938,
        "longitude": 139.7034,
        "expected_min_usv": 0.05,
        "category": "background",
        "description": "Tokyo Metropolitan Area"
    },
    "Paris_Eiffel": {
        "latitude": 48.8584,
        "longitude": 2.2945,
        "expected_min_usv": 0.06,
        "category": "background",
        "description": "Paris, France"
    },
    "New_York_Manhattan": {
        "latitude": 40.7831,
        "longitude": -73.9712,
        "expected_min_usv": 0.07,
        "category": "background",
        "description": "New York City, USA"
    },
    "London_Trafalgar": {
        "latitude": 51.5080,
        "longitude": -0.1281,
        "expected_min_usv": 0.06,
        "category": "background",
        "description": "London, UK"
    },
    "Denver_High_Altitude": {
        "latitude": 39.7392,
        "longitude": -104.9903,
        "expected_min_usv": 0.12,
        "category": "background_elevated",
        "description": "Denver, Colorado (mile-high altitude cosmic ray contribution)"
    }
}


@dataclass
class DataConfig:
    raw_safecast_dir: Path = Path(r"D:\datasets\safecast\data\processed\measurements")
    processed_dir: Path = Path("data/processed")
    sample_size: int = 1_000_000
    test_size: float = 0.15
    val_size: float = 0.15
    random_seed: int = 42
    min_cpm: float = 1.0
    max_cpm: float = 50_000.0
    min_usv: float = 0.003
    max_usv: float = 200.0


@dataclass
class ModelConfig:
    name: str = "SpatialHarmonicNet"
    in_dim: int = 3  # (x, y, z) 3D spherical coordinate
    num_fourier_frequencies: int = 32
    fourier_scale: float = 12.0
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256, 128, 64])
    dropout: float = 0.1
    use_residual: bool = True
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 2048
    epochs: int = 30
    patience: int = 7


@dataclass
class AnomalyThresholds:
    normal_z_max: float = 2.0
    elevated_z_max: float = 4.0
    anomaly_high_z_max: float = 8.0
    # Values above 8.0 are considered ANOMALY_CRITICAL
