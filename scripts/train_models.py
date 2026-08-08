"""Train and evaluate Spatial Neural Network and Spatial Ensemble Regressor."""

import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from rich.console import Console
from rich.table import Table

from radiation_map.models.spatial_net import SpatialHarmonicNet
from radiation_map.models.spatial_ensemble import SpatialEnsembleRegressor
from radiation_map.models.anomaly_detector import RadiationAnomalyDetector
from radiation_map.data.dataset import SpatialRadiationDataset
from radiation_map.training.train import train_spatial_model
from radiation_map.evaluation.metrics import evaluate_spatial_regression, evaluate_hotspot_benchmarks

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Train spatial radiation regressors")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--num-frequencies", type=int, default=32)
    parser.add_argument("--max-freq-log", type=float, default=4.5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold green]Loading processed data from {data_dir}...[/bold green]")
    train_df = pd.read_parquet(data_dir / "radiation_train.parquet")
    val_df = pd.read_parquet(data_dir / "radiation_val.parquet")
    test_df = pd.read_parquet(data_dir / "radiation_test.parquet")

    console.print(f"Train records: {len(train_df):,}")
    console.print(f"Val records: {len(val_df):,}")
    console.print(f"Test records: {len(test_df):,}")

    # Create PyTorch datasets
    train_ds = SpatialRadiationDataset(
        latitudes=train_df["latitude"].values,
        longitudes=train_df["longitude"].values,
        targets_usv=train_df["radiation_usv"].values,
    )
    val_ds = SpatialRadiationDataset(
        latitudes=val_df["latitude"].values,
        longitudes=val_df["longitude"].values,
        targets_usv=val_df["radiation_usv"].values,
    )
    test_ds = SpatialRadiationDataset(
        latitudes=test_df["latitude"].values,
        longitudes=test_df["longitude"].values,
        targets_usv=test_df["radiation_usv"].values,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[bold cyan]Using compute device: {device}[/bold cyan]")

    # Initialize SpatialHarmonicNet
    model = SpatialHarmonicNet(
        in_dim=3,
        num_frequencies=args.num_frequencies,
        max_frequency_log=args.max_freq_log,
        hidden_dims=(256, 256, 128, 64),
        dropout=args.dropout,
    )

    # Train Neural Network
    history = train_spatial_model(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        checkpoint_dir=checkpoint_dir,
        device=device,
        patience=7,
    )

    # Evaluate Neural Network on Test Set
    console.print("[bold green]Evaluating SpatialHarmonicNet on Holdout Test Set...[/bold green]")
    test_pred_dict = model.predict_radiation(
        latitudes=test_df["latitude"].values,
        longitudes=test_df["longitude"].values,
        device=device,
    )

    nn_report = evaluate_spatial_regression(
        y_true_usv=test_df["radiation_usv"].values,
        y_pred_usv=test_pred_dict["radiation_usv"],
        ci_lower_usv=test_pred_dict["ci_lower_usv"],
        ci_upper_usv=test_pred_dict["ci_upper_usv"],
    )

    # Train Fast Spatial Ensemble Regressor on a representative subset
    console.print("[bold green]Fitting Spatial Ensemble Regressor (Gradient Boosting + Spatial KNN)...[/bold green]")
    ensemble_model = SpatialEnsembleRegressor(
        num_frequencies=24,
        fourier_scale=10.0,
        max_iter=150,
        learning_rate=0.08,
        random_state=args.seed,
    )
    
    # Train ensemble on max 150k samples for high speed
    ensemble_subsample = train_df.sample(n=min(150_000, len(train_df)), random_state=args.seed)
    ensemble_model.fit(
        latitudes=ensemble_subsample["latitude"].values,
        longitudes=ensemble_subsample["longitude"].values,
        targets_usv=ensemble_subsample["radiation_usv"].values,
    )

    joblib.dump(ensemble_model, checkpoint_dir / "spatial_ensemble_regressor.joblib")
    console.print(f"Saved ensemble regressor to {checkpoint_dir / 'spatial_ensemble_regressor.joblib'}")

    ensemble_pred_dict = ensemble_model.predict(
        latitudes=test_df["latitude"].values,
        longitudes=test_df["longitude"].values,
    )
    ensemble_report = evaluate_spatial_regression(
        y_true_usv=test_df["radiation_usv"].values,
        y_pred_usv=ensemble_pred_dict["radiation_usv"],
        ci_lower_usv=ensemble_pred_dict["ci_lower_usv"],
        ci_upper_usv=ensemble_pred_dict["ci_upper_usv"],
    )

    # Print Comparison Table
    table = Table(title="Model Evaluation Benchmark (Holdout Test Set)")
    table.add_column("Metric", style="cyan")
    table.add_column("SpatialHarmonicNet (PyTorch)", style="green")
    table.add_column("Spatial Ensemble (GBR+KNN)", style="yellow")

    table.add_row("R2 (log-scale)", f"{nn_report.r2:.4f}", f"{ensemble_report.r2:.4f}")
    table.add_row("RMSE (uSv/h)", f"{nn_report.rmse_usv:.4f}", f"{ensemble_report.rmse_usv:.4f}")
    table.add_row("MAE (uSv/h)", f"{nn_report.mae_usv:.4f}", f"{ensemble_report.mae_usv:.4f}")
    table.add_row("Median AE (uSv/h)", f"{nn_report.medae_usv:.4f}", f"{ensemble_report.medae_usv:.4f}")
    table.add_row("RMSE (log-space)", f"{nn_report.rmse_log:.4f}", f"{ensemble_report.rmse_log:.4f}")
    table.add_row("MAE (log-space)", f"{nn_report.mae_log:.4f}", f"{ensemble_report.mae_log:.4f}")
    table.add_row("95% CI Coverage (PICP)", f"{nn_report.picp_95 * 100:.1f}%", f"{ensemble_report.picp_95 * 100:.1f}%")
    table.add_row("Mean CI Width (uSv/h)", f"{nn_report.mpiw_usv:.4f}", f"{ensemble_report.mpiw_usv:.4f}")
    table.add_row("Test Sample Count", f"{nn_report.sample_count:,}", f"{ensemble_report.sample_count:,}")

    console.print(table)

    # Evaluate Hotspots & Reference Landmarks
    detector = RadiationAnomalyDetector(model, device=device)
    hotspot_results = evaluate_hotspot_benchmarks(model, detector=detector)
    
    htable = Table(title="Global Landmark & Hotspot Verification")
    htable.add_column("Location", style="cyan")
    htable.add_column("Category", style="magenta")
    htable.add_column("Predicted uSv/h", style="green")
    htable.add_column("95% CI [Low, High]", style="blue")
    htable.add_column("Surge 5.0 uSv/h Test", style="red")

    for name, res in hotspot_results.items():
        ci_str = f"[{res['ci_lower_usv']:.3f}, {res['ci_upper_usv']:.3f}]"
        surge_str = f"{res.get('test_surge_severity', 'N/A')} (Z={res.get('test_surge_z', 0):.1f})"
        htable.add_row(name, res["category"], f"{res['predicted_usv']:.4f}", ci_str, surge_str)

    console.print(htable)

    # Save metrics and configuration
    eval_summary = {
        "spatial_harmonic_net": nn_report.to_dict(),
        "spatial_ensemble": ensemble_report.to_dict(),
        "hotspot_benchmarks": hotspot_results,
        "model_config": {
            "num_frequencies": args.num_frequencies,
            "max_frequency_log": args.max_freq_log,
            "hidden_dims": [256, 256, 128, 64],
            "dropout": args.dropout,
        }
    }
    with open(checkpoint_dir / "evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)

    console.print(f"[bold green]Saved evaluation report to {checkpoint_dir / 'evaluation_results.json'}[/bold green]")


if __name__ == "__main__":
    main()
