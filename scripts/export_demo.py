"""Export the trained SpatialHarmonicNet and demo data for the static browser demo.

Writes into demo/:
  model/weights.bin     flat float32 weights
  model/model.json      op list (offsets into weights.bin), constants, metrics
  data/parity.json      reference outputs from PyTorch, used to check the JS port
  data/landmarks.json   reference sites
  data/samples.json     measured readings sampled from the held-out test split
  data/coverage.json    sample counts per 2 degree cell (all splits), base64 uint8
"""

import base64

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from radiation_map.config import CPM_TO_USV_CONVERSION_FACTOR, REFERENCE_SITES, AnomalyThresholds
from radiation_map.data.transforms import latlon_to_cartesian
from radiation_map.inference.predictor import SpatialRadiationPredictor
from radiation_map.models.spatial_net import ResidualBlock


class Packer:
    def __init__(self):
        self.chunks = []
        self.size = 0

    def add(self, tensor):
        arr = tensor.detach().cpu().numpy().astype("<f4").ravel()
        offset = self.size
        self.chunks.append(arr)
        self.size += arr.size
        return offset

    def linear(self, layer):
        return {
            "op": "linear",
            "in": layer.in_features,
            "out": layer.out_features,
            "w": self.add(layer.weight),
            "b": self.add(layer.bias),
        }

    def norm(self, layer):
        return {
            "op": "layernorm",
            "dim": layer.normalized_shape[0],
            "eps": layer.eps,
            "g": self.add(layer.weight),
            "b": self.add(layer.bias),
        }


def build_ops(model, packer):
    ops = [packer.linear(model.input_layer[0]), packer.norm(model.input_layer[1]), {"op": "silu"}]
    for block in model.backbone:
        if isinstance(block, ResidualBlock):
            ops.append(
                {
                    "op": "residual",
                    "fc1": packer.linear(block.fc1),
                    "norm1": packer.norm(block.norm1),
                    "fc2": packer.linear(block.fc2),
                    "norm2": packer.norm(block.norm2),
                }
            )
        else:
            ops += [packer.linear(block[0]), packer.norm(block[1]), {"op": "silu"}]
    heads = {}
    for name, head in (("mu", model.head_mu), ("logvar", model.head_logvar)):
        heads[name] = [packer.linear(head[0]), packer.linear(head[2])]
    return ops, heads


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best_spatial_harmonic_net.pt")
    parser.add_argument("--config", default="hf_export/config.json")
    parser.add_argument("--eval", default="checkpoints/evaluation_results.json")
    parser.add_argument("--test", default="data/processed/radiation_test.parquet")
    parser.add_argument("--out", default="demo")
    parser.add_argument("--samples", type=int, default=1500)
    args = parser.parse_args()

    out = Path(args.out)
    (out / "model").mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(parents=True, exist_ok=True)

    predictor = SpatialRadiationPredictor(
        model_path=args.checkpoint, config_path=args.config, device=torch.device("cpu")
    )
    model = predictor.model.eval()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    evaluation = json.loads(Path(args.eval).read_text(encoding="utf-8"))

    packer = Packer()
    ops, heads = build_ops(model, packer)
    np.concatenate(packer.chunks).tofile(out / "model" / "weights.bin")

    thresholds = AnomalyThresholds()
    manifest = {
        "num_frequencies": cfg["num_frequencies"],
        "max_frequency_log": cfg["max_frequency_log"],
        "embed_dim": model.embedding.output_dim,
        "ops": ops,
        "heads": heads,
        "logvar_clamp": [-7.0, 3.0],
        "target_scale": 10.0,
        "cpm_to_usv": CPM_TO_USV_CONVERSION_FACTOR,
        "thresholds": {
            "normal_z_max": thresholds.normal_z_max,
            "elevated_z_max": thresholds.elevated_z_max,
            "anomaly_high_z_max": thresholds.anomaly_high_z_max,
        },
        "weights_floats": packer.size,
        "metrics": {
            "neural": evaluation["spatial_harmonic_net"],
            "ensemble": evaluation["spatial_ensemble"],
        },
    }
    (out / "model" / "model.json").write_text(json.dumps(manifest), encoding="utf-8")

    # Parity fixtures: PyTorch reference outputs at random points plus the landmarks.
    rng = np.random.default_rng(7)
    lats = np.concatenate([rng.uniform(-80, 80, 120), [v["latitude"] for v in REFERENCE_SITES.values()]])
    lons = np.concatenate([rng.uniform(-180, 180, 120), [v["longitude"] for v in REFERENCE_SITES.values()]])
    xyz = torch.tensor(latlon_to_cartesian(lats, lons), dtype=torch.float32)
    with torch.no_grad():
        mu, logvar = model(xyz)
    parity = [
        {"lat": round(float(a), 5), "lon": round(float(b), 5), "mu": float(m), "logvar": float(v)}
        for a, b, m, v in zip(lats, lons, mu.squeeze(-1), logvar.squeeze(-1))
    ]
    (out / "data" / "parity.json").write_text(json.dumps(parity), encoding="utf-8")

    sites = [
        {
            "id": key,
            "name": key.replace("_", " "),
            "category": info.get("category", "landmark"),
            "description": info.get("description", ""),
            "latitude": info["latitude"],
            "longitude": info["longitude"],
        }
        for key, info in REFERENCE_SITES.items()
    ]
    (out / "data" / "landmarks.json").write_text(json.dumps(sites), encoding="utf-8")

    test_path = Path(args.test)
    if test_path.exists():
        df = pd.read_parquet(test_path)
        sample = df.sample(n=min(args.samples, len(df)), random_state=42)
        rows = [
            [round(float(r.latitude), 4), round(float(r.longitude), 4), round(float(r.radiation_usv), 4)]
            for r in sample.itertuples()
        ]
        (out / "data" / "samples.json").write_text(json.dumps(rows), encoding="utf-8")
        print(f"samples: {len(rows)} of {len(df)} test rows")

    parts = [Path(args.test).with_name(f"radiation_{n}.parquet") for n in ("train", "val", "test")]
    frames = [pd.read_parquet(f, columns=["latitude", "longitude"]) for f in parts if f.exists()]
    if frames:
        pts = pd.concat(frames)
        cell = 2
        rows, cols = 180 // cell, 360 // cell
        iy = ((pts.latitude.clip(-89.999, 89.999) + 90) // cell).astype(int)
        ix = ((pts.longitude.clip(-179.999, 179.999) + 180) // cell).astype(int)
        counts = np.zeros((rows, cols), dtype=np.int64)
        np.add.at(counts, (iy.to_numpy(), ix.to_numpy()), 1)
        # Counts saturate at 255 after a log squeeze; the page only needs relative support.
        packed = np.minimum(255, np.round(np.log1p(counts) / np.log1p(counts.max()) * 255)).astype(np.uint8)
        (out / "data" / "coverage.json").write_text(
            json.dumps(
                {
                    "cell": cell,
                    "rows": rows,
                    "cols": cols,
                    "max_count": int(counts.max()),
                    "data": base64.b64encode(packed.tobytes()).decode("ascii"),
                }
            ),
            encoding="utf-8",
        )
        print(f"coverage: {len(pts)} points, {(counts > 0).sum()} of {rows * cols} cells occupied")

    print(f"weights: {packer.size} floats ({packer.size * 4 / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
