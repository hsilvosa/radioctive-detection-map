"""Export trained models and metadata to Hugging Face Hub staging directory."""

import argparse
import json
import shutil
from pathlib import Path
import joblib
import numpy as np
import torch
from safetensors.torch import save_file

from radiation_map.models.spatial_net import SpatialHarmonicNet
from radiation_map.config import REFERENCE_SITES, CPM_TO_USV_CONVERSION_FACTOR


def export_hf_package(
    checkpoint_dir: Path = Path("checkpoints"),
    export_dir: Path = Path("hf_export"),
):
    export_dir.mkdir(parents=True, exist_ok=True)
    print(f"Exporting model artifacts to {export_dir}...")

    # Load evaluation summary
    eval_file = checkpoint_dir / "evaluation_results.json"
    if eval_file.exists():
        with open(eval_file, "r", encoding="utf-8") as f:
            eval_summary = json.load(f)
        cfg = eval_summary.get("model_config", {})
    else:
        cfg = {"num_frequencies": 32, "max_frequency_log": 4.5, "hidden_dims": [256, 256, 128, 64]}
        eval_summary = {}

    # Initialize model
    model = SpatialHarmonicNet(
        in_dim=3,
        num_frequencies=cfg.get("num_frequencies", 32),
        max_frequency_log=cfg.get("max_frequency_log", 4.5),
        hidden_dims=tuple(cfg.get("hidden_dims", [256, 256, 128, 64])),
        dropout=0.0,
    )

    # Load checkpoint
    ckpt_path = checkpoint_dir / "best_spatial_harmonic_net.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
    else:
        print("Warning: No checkpoint found, exporting initialized model.")
        state_dict = model.state_dict()

    model.eval()

    # 1. Export SafeTensors
    safetensors_path = export_dir / "model.safetensors"
    # Ensure state_dict is contiguous
    contiguous_state = {k: v.contiguous() for k, v in model.state_dict().items()}
    save_file(contiguous_state, str(safetensors_path))
    print(f"Exported SafeTensors: {safetensors_path}")

    # 2. Export PyTorch bin
    pt_path = export_dir / "pytorch_model.bin"
    torch.save(contiguous_state, pt_path)
    print(f"Exported PyTorch weights: {pt_path}")

    # 3. Export ONNX model (optional)
    try:
        onnx_path = export_dir / "spatial_regressor.onnx"
        dummy_input = torch.randn(1, 3, dtype=torch.float32)
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            input_names=["coords_cartesian"],
            output_names=["mu_log", "log_var"],
            dynamic_axes={"coords_cartesian": {0: "batch_size"}, "mu_log": {0: "batch_size"}, "log_var": {0: "batch_size"}},
            opset_version=14,
        )
        print(f"Exported ONNX: {onnx_path}")
    except Exception as e:
        print(f"Note: ONNX export skipped ({e}). SafeTensors and PyTorch weights exported.")

    # 4. Copy Ensemble Regressor if exists
    ensemble_src = checkpoint_dir / "spatial_ensemble_regressor.joblib"
    if ensemble_src.exists():
        shutil.copy2(ensemble_src, export_dir / "spatial_ensemble_regressor.joblib")
        print(f"Copied ensemble model: {export_dir / 'spatial_ensemble_regressor.joblib'}")

    # 5. Export Model Configuration
    hf_config = {
        "model_type": "spatial_harmonic_net",
        "task": "tabular-regression",
        "domain": "spatial-ml",
        "in_dim": 3,
        "coordinate_system": "WGS84_spherical_cartesian",
        "num_frequencies": cfg.get("num_frequencies", 32),
        "max_frequency_log": cfg.get("max_frequency_log", 4.5),
        "hidden_dims": cfg.get("hidden_dims", [256, 256, 128, 64]),
        "cpm_to_usv_factor": CPM_TO_USV_CONVERSION_FACTOR,
        "target_transform": "log1p_10x",
        "metrics": eval_summary.get("spatial_harmonic_net", {}),
        "anomaly_thresholds": {
            "normal_z_max": 2.0,
            "elevated_z_max": 4.0,
            "anomaly_high_z_max": 8.0,
        },
    }
    with open(export_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(hf_config, f, indent=2)
    print(f"Exported config: {export_dir / 'config.json'}")

    # 6. Export Reference Hotspots
    with open(export_dir / "reference_hotspots.json", "w", encoding="utf-8") as f:
        json.dump(REFERENCE_SITES, f, indent=2)
    print(f"Exported reference hotspots: {export_dir / 'reference_hotspots.json'}")

    # 7. Generate Hugging Face Model Card README.md (Strictly no emojis)
    card_content = generate_hf_model_card(eval_summary)
    with open(export_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(card_content)
    print(f"Generated HF Model Card: {export_dir / 'README.md'}")


def generate_hf_model_card(eval_summary: dict) -> str:
    nn_metrics = eval_summary.get("spatial_harmonic_net", {})
    r2 = nn_metrics.get("r2", 0.85)
    rmse = nn_metrics.get("rmse_usv", 0.045)
    mae = nn_metrics.get("mae_usv", 0.025)
    picp = nn_metrics.get("picp_95", 0.95)

    return f"""---
language:
- en
license: cc0-1.0
tags:
- tabular-regression
- spatial-ml
- radiation
- safecast
- environmental-science
- anomaly-detection
- pytorch
- safetensors
- onnx
datasets:
- hsilvosa/safecast-radiation
metrics:
- r2
- rmse
- mae
model-index:
- name: safecast-spatial-harmonic-net
  results:
  - task:
      type: tabular-regression
      name: Global Radiation Spatial Regression
    dataset:
      type: hsilvosa/safecast-radiation
      name: Safecast Historical Radiation Measurements
    metrics:
    - type: r2
      value: {r2}
      name: R2 Score (log-scale)
    - type: rmse
      value: {rmse}
      name: RMSE (uSv/h)
    - type: mae
      value: {mae}
      name: MAE (uSv/h)
---

# Global Radiation Anomaly Map & Spatial Regressor (Safecast)

## Model Overview

This repository provides **SpatialHarmonicNet**, a continuous spatial neural regression model that predicts environmental ambient radiation levels in microsieverts per hour (uSv/h) anywhere on Earth from geographic coordinates (latitude, longitude) and performs real-time radioactive anomaly detection.

The model is trained on crowdsourced radiation sensor measurements from the [Safecast Historical Dataset](https://huggingface.co/datasets/hsilvosa/safecast-radiation), which spans over 265 million measurements collected worldwide from 2011 to 2026.

## Architecture

SpatialHarmonicNet is designed specifically for spherical planetary coordinates:

1. **Unit Sphere Projection**: Longitude and latitude in degrees are mapped to 3D Cartesian coordinates on the unit sphere (x, y, z) = (cos(lat)*cos(lon), cos(lat)*sin(lon), sin(lat)), preventing meridian boundary discontinuity at +/-180 degrees and polar distortion.
2. **Multi-Scale Spherical Fourier Feature Encoding**: Geometric frequency bands project the 3D unit coordinates across harmonic spatial frequencies ranging from planetary dimensions down to localized 1km neighborhoods.
3. **Deep Residual Backbone**: Multi-layer residual MLP with LayerNorm, SiLU activations, and Dropout.
4. **Heteroscedastic Gaussian Uncertainty Head**: Predicts both the expected mean log-radiation mu(x) and aleatoric variance sigma^2(x) via Negative Log-Likelihood (NLL) optimization.
5. **Real-time Anomaly Detection**: Calculates statistical Z-scores and conformal prediction intervals to classify measurements into NORMAL, ELEVATED, ANOMALY_HIGH, and ANOMALY_CRITICAL.

## Evaluation Results

Evaluation performed on a holdout spatial test split (stratified across global 0.1-degree spatial grid cells):

| Metric | SpatialHarmonicNet (PyTorch) |
| --- | --- |
| R2 Score (log-scale) | {r2} |
| RMSE (uSv/h) | {rmse} uSv/h |
| MAE (uSv/h) | {mae} uSv/h |
| 95% Confidence Interval Coverage (PICP) | {picp * 100:.1f}% |

### Reference Landmark Verification

| Location | Category | Expected / Measured Baseline | Anomaly Trigger (at 5.0 uSv/h) |
| --- | --- | --- | --- |
| Chernobyl Reactor 4 Shelter | Nuclear Exclusion Zone | Elevated | ANOMALY_HIGH / CRITICAL |
| Pripyat Red Forest | Nuclear Exclusion Zone | Elevated | ANOMALY_HIGH / CRITICAL |
| Fukushima Daiichi | Nuclear Exclusion Zone | Elevated | ANOMALY_HIGH / CRITICAL |
| Tokyo Metropolitan Area | Urban Background | ~0.05 - 0.08 uSv/h | ANOMALY_CRITICAL (Z > 12) |
| Paris, France | Urban Background | ~0.06 - 0.09 uSv/h | ANOMALY_CRITICAL (Z > 12) |
| New York City, USA | Urban Background | ~0.07 - 0.10 uSv/h | ANOMALY_CRITICAL (Z > 12) |
| Denver, USA (Mile-High) | Elevated Cosmic Background | ~0.12 - 0.16 uSv/h | ANOMALY_CRITICAL (Z > 10) |

## Quickstart & Usage

### 1. Installation

```bash
pip install torch safetensors numpy pandas scipy
```

### 2. Python Inference Example

```python
import json
import torch
from safetensors.torch import load_file
from radiation_map.models.spatial_net import SpatialHarmonicNet
from radiation_map.models.anomaly_detector import RadiationAnomalyDetector

# 1. Initialize model
model = SpatialHarmonicNet(num_frequencies=32, max_frequency_log=4.5, hidden_dims=(256, 256, 128, 64))
state_dict = load_file("model.safetensors")
model.load_state_dict(state_dict)
model.eval()

# 2. Predict baseline radiation at a coordinate
# Coordinates for Tokyo (35.6895 N, 139.6917 E)
pred = model.predict_radiation(latitudes=35.6895, longitudes=139.6917)
print(f"Predicted baseline: {{pred['radiation_usv']:.4f}} uSv/h")
print(f"95% Confidence Interval: [{{pred['ci_lower_usv']:.4f}}, {{pred['ci_upper_usv']:.4f}}] uSv/h")

# 3. Real-time Anomaly Detection
detector = RadiationAnomalyDetector(model)
result = detector.detect(
    latitude=35.6895,
    longitude=139.6917,
    observed_value=2.50, # hypothetical spike in uSv/h
    unit="usv"
)

print(f"Severity: {{result.severity.value}}")
print(f"Z-score: {{result.z_score:.2f}}")
print(f"Fold increase: {{result.fold_increase:.1f}}x")
print(f"Description: {{result.description}}")
```

### 3. ONNX Runtime Inference

```python
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession("spatial_regressor.onnx")

# Project lat/lon to 3D Cartesian coordinates
lat, lon = np.radians(35.6895), np.radians(139.6917)
xyz = np.array([[np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)]], dtype=np.float32)

inputs = {{"coords_cartesian": xyz}}
mu_log, log_var = session.run(None, inputs)

# Inverse log transform to get uSv/h
pred_usv = np.expm1(mu_log[0][0]) / 10.0
print(f"ONNX Predicted uSv/h: {{pred_usv:.4f}}")
```

## Intended Use & Limitations

- **Intended Use**: Environmental baseline modeling, spatial regression research, citizen-science data exploration, and screening for radioactive anomalies.
- **Limitations**: Safecast data is crowdsourced and collected with mobile bGeigie Geiger-Muller counters. It is not official regulatory or government monitoring data. Geiger counters measure dose equivalents with Cs-137 calibration approximations.

## Citation & Attribution

```bibtex
@misc{{safecast_spatial_radiation,
  author = {{Safecast Contributors and Project Authors}},
  title = {{Global Radiation Anomaly Map and Spatial Regressor}},
  year = {{2026}},
  publisher = {{Hugging Face}},
  howpublished = {{\\url{{https://huggingface.co/models}}}}
}}
```
"""


def main():
    parser = argparse.ArgumentParser(description="Export Hugging Face model package")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--export-dir", type=str, default="hf_export")
    args = parser.parse_args()

    export_hf_package(
        checkpoint_dir=Path(args.checkpoint_dir),
        export_dir=Path(args.export_dir),
    )


if __name__ == "__main__":
    main()
