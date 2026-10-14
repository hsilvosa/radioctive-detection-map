# Global Radiation Anomaly Map & Spatial Regressor

Spatial regression and real-time radioactive anomaly detection system based on historical Safecast global radiation measurements (>265 million records), featuring spherical harmonic neural networks, spatial ensemble regressors, and an interactive geospatial dashboard.

## Overview

- **Task Type**: `tabular-regression` / `spatial-ml`
- **Target Measurement**: Ambient radiation equivalent dose rate in microsieverts per hour ($\mu\text{Sv/h}$).
- **Dataset**: [Safecast Historical Radiation Measurements](https://huggingface.co/datasets/hsilvosa/safecast-radiation) (CC0-1.0).
- **Core Models**:
  - `SpatialHarmonicNet`: Multi-scale Spherical Harmonic Random Fourier Feature neural network predicting expected baseline $\mu(\mathbf{x})$ and heteroscedastic uncertainty $\sigma^2(\mathbf{x})$.
  - `SpatialEnsembleRegressor`: Fast tree-based spatial regressor (HistGradientBoosting + Spherical Haversine KNN) with Conformal Prediction intervals.
  - `RadiationAnomalyDetector`: Real-time detector computing statistical Z-scores and conformal bounds to identify radiological anomalies and contaminated zones.

## Repository Structure

```
radiation-map/
├── .gitignore
├── LICENSE
├── pyproject.toml
├── README.md                      # Project repository documentation
├── app.py                         # Web dashboard & REST API server
├── src/
│   └── radiation_map/
│       ├── config.py              # Calibration constants, sites, thresholds
│       ├── cli.py                 # Command line interface
│       ├── data/
│       │   ├── transforms.py      # 3D spherical mapping & unit normalization
│       │   ├── extract.py         # Streaming Parquet sampler
│       │   └── dataset.py         # PyTorch Dataset
│       ├── models/
│       │   ├── spatial_net.py     # SpatialHarmonicNet architecture
│       │   ├── spatial_ensemble.py# Gradient Boosting + Spatial KNN
│       │   └── anomaly_detector.py# Statistical Z-score & severity engine
│       ├── training/
│       │   ├── loss.py            # Heteroscedastic Gaussian NLL loss
│       │   └── train.py           # GPU training loop with LR scheduling
│       ├── evaluation/
│       │   └── metrics.py         # PICP, R2, RMSE, MAE, hotspot benchmarks
│       ├── inference/
│       │   └── predictor.py       # Production prediction API
│       └── web/
│           ├── api.py             # FastAPI backend
│           └── static/
│               └── index.html     # Interactive dark-mode map UI
├── scripts/
│   ├── build_dataset.py           # Extraction from Safecast shards
│   ├── train_models.py            # Training & benchmark evaluation
│   └── export_hf.py               # Hugging Face packaging utility
└── tests/
    ├── test_transforms.py
    ├── test_models.py
    └── test_inference.py
```

## Static Demo

`demo/` is a self-contained static site: the trained `SpatialHarmonicNet` runs in the browser (plain JavaScript in web workers, no server and no ML runtime), so it can be hosted on any static host. It colours the current map view with the model, scores a reading you type against any point, and replays real readings from the held-out test split.

```bash
# Export weights and demo data (needs the checkpoint and the test split)
PYTHONPATH=src python scripts/export_demo.py

# Check the JavaScript forward pass against PyTorch (max difference ~1e-5)
node demo/parity.test.mjs

# Serve locally
python -m http.server 8000 --directory demo
```

The page ships about 3 MB: 1.6 MB of weights, Natural Earth land and borders via [world-atlas](https://github.com/topojson/world-atlas) (ISC), and Leaflet. There are no external requests.

## Installation

Clone the repository and install with pip:

```bash
git clone https://github.com/your-username/radiation-map.git
cd radiation-map
pip install -e .
```

## Data Preparation

To extract and sample a spatially balanced dataset from local Safecast Parquet shards:

```bash
python scripts/build_dataset.py --safecast-dir "D:/datasets/safecast/data/processed/measurements" --samples 500000
```

This generates `data/processed/radiation_train.parquet`, `data/processed/radiation_val.parquet`, and `data/processed/radiation_test.parquet`.

## Training & Evaluation

Train `SpatialHarmonicNet` (PyTorch with CUDA acceleration) and `SpatialEnsembleRegressor`:

```bash
python scripts/train_models.py --epochs 25 --batch-size 2048 --lr 1e-3
```

## Evaluation & Verification Benchmarks

### 1. Spatial Holdout Test Set (64,258 samples)

| Metric | SpatialHarmonicNet (PyTorch) | Spatial Ensemble (GBR + KNN) |
| --- | --- | --- |
| **95% CI Coverage (PICP)** | **94.3%** | 89.5% |
| **Median Absolute Error ($\mu\text{Sv/h}$)** | **0.0419 $\mu\text{Sv/h}$** | 0.0416 $\mu\text{Sv/h}$ |
| **MAE (log-space)** | 0.4339 | 0.3915 |
| **RMSE (log-space)** | 0.9001 | 0.8651 |
| **Mean CI Width ($\mu\text{Sv/h}$)** | 0.9833 $\mu\text{Sv/h}$ | 0.4030 $\mu\text{Sv/h}$ |
| **Test Sample Count** | 64,258 | 64,258 |

### 2. Global Landmark Verification

| Location | Category | Predicted Baseline | 95% Normal Range | Surge 5.0 $\mu\text{Sv/h}$ Test |
| --- | --- | --- | --- | --- |
| **Chernobyl Reactor 4** | Hotspot / Exclusion | 0.4074 $\mu\text{Sv/h}$ | [0.000, 5.651] $\mu\text{Sv/h}$ | Baseline elevated |
| **Pripyat Red Forest** | Hotspot / Exclusion | 0.4073 $\mu\text{Sv/h}$ | [0.000, 5.651] $\mu\text{Sv/h}$ | Baseline elevated |
| **Fukushima Daiichi** | Hotspot / Exclusion | 0.4308 $\mu\text{Sv/h}$ | [0.000, 4.194] $\mu\text{Sv/h}$ | ELEVATED ($Z = 2.1$) |
| **Futaba Town, Fukushima** | Hotspot / Exclusion | 0.4306 $\mu\text{Sv/h}$ | [0.000, 4.110] $\mu\text{Sv/h}$ | ELEVATED ($Z = 2.1$) |
| **Ramsar Talesh Mahalleh** | High Natural Background | 0.3486 $\mu\text{Sv/h}$ | [0.000, 2.637] $\mu\text{Sv/h}$ | ELEVATED ($Z = 2.6$) |
| **Guarapari Beach** | High Natural Background | 0.1606 $\mu\text{Sv/h}$ | [0.000, 1.056] $\mu\text{Sv/h}$ | ELEVATED ($Z = 3.9$) |
| **Tokyo Shinjuku** | Urban Background | 0.0846 $\mu\text{Sv/h}$ | [0.001, 0.239] $\mu\text{Sv/h}$ | **ANOMALY_CRITICAL** ($Z = 10.7$) |
| **Paris Eiffel** | Urban Background | 0.3225 $\mu\text{Sv/h}$ | [0.000, 4.766] $\mu\text{Sv/h}$ | NORMAL ($Z = 2.0$) |
| **New York Manhattan** | Urban Background | 0.0788 $\mu\text{Sv/h}$ | [0.000, 0.253] $\mu\text{Sv/h}$ | **ANOMALY_CRITICAL** ($Z = 9.7$) |
| **Denver Mile-High** | High-Altitude Cosmic | 0.1759 $\mu\text{Sv/h}$ | [0.000, 0.661] $\mu\text{Sv/h}$ | **ANOMALY_HIGH** ($Z = 5.6$) |

## Hugging Face Hub Packaging

To export weights (`safetensors`, `pytorch_model.bin`, `onnx`, `joblib`), configuration, and the Model Card to `hf_export/`:

```bash
python scripts/export_hf.py
```

To upload to the Hugging Face Hub:

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="hf_export",
    repo_id="your-username/safecast-radiation-spatial-regressor",
    repo_type="model"
)
```

## Interactive Web Dashboard

Launch the FastAPI application:

```bash
python app.py
```

Navigate to `http://127.0.0.1:8000` to interact with:
- Global map with click-to-query coordinate evaluation.
- Live ambient radiation estimate ($\mu\text{Sv/h}$) with 95% confidence intervals.
- Real-time sensor simulation with Z-score and severity badge (`NORMAL`, `ELEVATED`, `ANOMALY_HIGH`, `ANOMALY_CRITICAL`).
- Reference landmark navigator (Chernobyl, Fukushima, Pripyat, Ramsar, Tokyo, Paris, New York, Denver).

## CLI Usage

```bash
# Predict ambient radiation baseline for coordinates (e.g. Tokyo)
radiation-map predict --lat 35.6895 --lon 139.6917

# Evaluate sensor measurement for radioactive anomalies
radiation-map detect --lat 35.6895 --lon 139.6917 --value 4.5 --unit usv

# Batch evaluate a CSV file
radiation-map batch --input-csv samples.csv --output-csv results.csv

# Evaluate reference landmark benchmark
radiation-map benchmark

# Start Web application
radiation-map serve --port 8000
```

## Python API Example

```python
from radiation_map.inference.predictor import SpatialRadiationPredictor

# Initialize predictor with trained checkpoint
predictor = SpatialRadiationPredictor(
    model_path="checkpoints/best_spatial_harmonic_net.pt",
    config_path="hf_export/config.json"
)

# 1. Point prediction
pred = predictor.predict_point(latitude=35.6895, longitude=139.6917)
print(f"Predicted ambient radiation: {pred['radiation_usv']:.4f} uSv/h")
print(f"95% Confidence Interval: [{pred['ci_lower_usv']:.4f}, {pred['ci_upper_usv']:.4f}] uSv/h")

# 2. Real-time Anomaly Detection
result = predictor.detect_anomaly(
    latitude=35.6895,
    longitude=139.6917,
    observed_value=3.2,
    unit="usv"
)
print(f"Severity: {result.severity.value}")
print(f"Z-Score: {result.z_score:.2f}")
print(f"Fold increase: {result.fold_increase:.1f}x")
print(f"Description: {result.description}")
```

## Testing

Run unit tests:

```bash
pytest tests/ -v
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. Safecast radiation measurement dataset is provided under the CC0 1.0 Universal License.
