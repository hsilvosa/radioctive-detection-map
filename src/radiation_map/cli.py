"""Command Line Interface for Radiation Map & Spatial Regressor."""

import argparse
import sys
from pathlib import Path
import pandas as pd
from rich.console import Console
from rich.table import Table
import uvicorn

from radiation_map.inference.predictor import SpatialRadiationPredictor
from radiation_map.config import REFERENCE_SITES

console = Console()


def predict_command(args):
    """Predict expected radiation at a given location."""
    predictor = SpatialRadiationPredictor(
        model_path=args.model_path if Path(args.model_path).exists() else None,
        config_path=args.config_path if Path(args.config_path).exists() else None,
    )
    res = predictor.predict_point(args.lat, args.lon, num_mc_samples=args.mc_samples)
    
    table = Table(title=f"Radiation Prediction at ({args.lat:.4f}, {args.lon:.4f})")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Latitude", f"{res['latitude']:.4f}")
    table.add_row("Longitude", f"{res['longitude']:.4f}")
    table.add_row("Predicted Ambient Radiation", f"{res['radiation_usv']:.4f} uSv/h")
    table.add_row("Standard Uncertainty", f"+/- {res['uncertainty_usv']:.4f} uSv/h")
    table.add_row("95% Confidence Interval", f"[{res['ci_lower_usv']:.4f}, {res['ci_upper_usv']:.4f}] uSv/h")
    console.print(table)


def detect_command(args):
    """Detect if an observed measurement is an anomaly."""
    predictor = SpatialRadiationPredictor(
        model_path=args.model_path if Path(args.model_path).exists() else None,
        config_path=args.config_path if Path(args.config_path).exists() else None,
    )
    res = predictor.detect_anomaly(
        latitude=args.lat,
        longitude=args.lon,
        observed_value=args.value,
        unit=args.unit,
    )
    
    severity_colors = {
        "NORMAL": "green",
        "ELEVATED": "yellow",
        "ANOMALY_HIGH": "bold red",
        "ANOMALY_CRITICAL": "bold white on red",
    }
    color = severity_colors.get(res.severity.value, "white")
    
    table = Table(title=f"Radioactive Anomaly Assessment at ({args.lat:.4f}, {args.lon:.4f})")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Observed Value", f"{res.observed_usv:.4f} uSv/h (from {args.value} {args.unit})")
    table.add_row("Expected Baseline", f"{res.expected_baseline_usv:.4f} uSv/h")
    table.add_row("95% Normal Band", f"[{res.ci_lower_usv:.4f}, {res.ci_upper_usv:.4f}] uSv/h")
    table.add_row("Statistical Z-Score", f"{res.z_score:.2f}")
    table.add_row("P-Value", f"{res.p_value:.2e}")
    table.add_row("Fold Increase", f"{res.fold_increase:.1f}x baseline")
    table.add_row("Severity", f"[{color}]{res.severity.value}[/{color}]")
    table.add_row("Description", res.description)
    console.print(table)


def batch_command(args):
    """Process a CSV of coordinates and measurements."""
    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)
    
    if not input_path.exists():
        console.print(f"[bold red]Error: Input file {input_path} does not exist.[/bold red]")
        sys.exit(1)
        
    df = pd.read_csv(input_path)
    if "latitude" not in df.columns or "longitude" not in df.columns:
        console.print("[bold red]Error: CSV must contain 'latitude' and 'longitude' columns.[/bold red]")
        sys.exit(1)
        
    predictor = SpatialRadiationPredictor(
        model_path=args.model_path if Path(args.model_path).exists() else None,
        config_path=args.config_path if Path(args.config_path).exists() else None,
    )
    
    console.print(f"[bold green]Running spatial inference on {len(df):,} records...[/bold green]")
    preds = predictor.predict_batch(df["latitude"].values, df["longitude"].values)
    
    df["predicted_usv"] = [p["radiation_usv"] for p in preds]
    df["uncertainty_usv"] = [p["uncertainty_usv"] for p in preds]
    df["ci_lower_usv"] = [p["ci_lower_usv"] for p in preds]
    df["ci_upper_usv"] = [p["ci_upper_usv"] for p in preds]
    
    if "value" in df.columns:
        unit = args.unit
        anomalies = []
        z_scores = []
        severities = []
        for _, row in df.iterrows():
            res = predictor.detect_anomaly(
                latitude=row["latitude"],
                longitude=row["longitude"],
                observed_value=row["value"],
                unit=row.get("unit", unit),
            )
            anomalies.append(res.is_anomaly)
            z_scores.append(res.z_score)
            severities.append(res.severity.value)
            
        df["is_anomaly"] = anomalies
        df["z_score"] = z_scores
        df["severity"] = severities
        
    df.to_csv(output_path, index=False)
    console.print(f"[bold green]Batch results saved to {output_path}[/bold green]")


def benchmark_command(args):
    """Run benchmark against known global sites."""
    predictor = SpatialRadiationPredictor(
        model_path=args.model_path if Path(args.model_path).exists() else None,
        config_path=args.config_path if Path(args.config_path).exists() else None,
    )
    sites = predictor.get_reference_landmarks()
    
    table = Table(title="Global Landmark & Hotspot Benchmark")
    table.add_column("Location", style="cyan")
    table.add_column("Category", style="magenta")
    table.add_column("Coordinates", style="blue")
    table.add_column("Predicted Baseline", style="green")
    table.add_column("95% Normal Range", style="yellow")
    
    for s in sites:
        coords_str = f"{s['latitude']:.2f}, {s['longitude']:.2f}"
        ci_str = f"[{s['ci_lower_usv']:.3f}, {s['ci_upper_usv']:.3f}] uSv/h"
        table.add_row(s["name"], s["category"], coords_str, f"{s['predicted_usv']:.4f} uSv/h", ci_str)
        
    console.print(table)


def serve_command(args):
    """Launch the Web Application and REST API."""
    console.print(f"[bold green]Starting Radiation Map server on http://{args.host}:{args.port}[/bold green]")
    uvicorn.run("radiation_map.web.api:app", host=args.host, port=args.port, reload=args.reload)


def main():
    parser = argparse.ArgumentParser(
        description="Global Radiation Anomaly Map & Spatial Regressor CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-path", type=str, default="checkpoints/best_spatial_harmonic_net.pt")
    parser.add_argument("--config-path", type=str, default="hf_export/config.json")
    
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")
    
    # Predict
    p_parser = subparsers.add_parser("predict", help="Predict radiation baseline at a coordinate")
    p_parser.add_argument("--lat", type=float, required=True, help="Latitude in degrees (-90 to 90)")
    p_parser.add_argument("--lon", type=float, required=True, help="Longitude in degrees (-180 to 180)")
    p_parser.add_argument("--mc-samples", type=int, default=1, help="Monte Carlo samples for epistemic uncertainty")
    
    # Detect
    d_parser = subparsers.add_parser("detect", help="Detect radiation anomaly for a given measurement")
    d_parser.add_argument("--lat", type=float, required=True, help="Latitude in degrees")
    d_parser.add_argument("--lon", type=float, required=True, help="Longitude in degrees")
    d_parser.add_argument("--value", type=float, required=True, help="Observed radiation value")
    d_parser.add_argument("--unit", type=str, default="usv", choices=["usv", "cpm"], help="Measurement unit")
    
    # Batch
    b_parser = subparsers.add_parser("batch", help="Batch predict and detect anomalies on a CSV file")
    b_parser.add_argument("--input-csv", type=str, required=True, help="Input CSV path")
    b_parser.add_argument("--output-csv", type=str, required=True, help="Output CSV path")
    b_parser.add_argument("--unit", type=str, default="usv", help="Default unit if not present in CSV")
    
    # Benchmark
    subparsers.add_parser("benchmark", help="Evaluate model predictions against global landmarks")
    
    # Serve
    s_parser = subparsers.add_parser("serve", help="Launch interactive web dashboard and REST API")
    s_parser.add_argument("--host", type=str, default="127.0.0.1", help="Host IP")
    s_parser.add_argument("--port", type=int, default=8000, help="Port number")
    s_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)
        
    if args.command == "predict":
        predict_command(args)
    elif args.command == "detect":
        detect_command(args)
    elif args.command == "batch":
        batch_command(args)
    elif args.command == "benchmark":
        benchmark_command(args)
    elif args.command == "serve":
        serve_command(args)


if __name__ == "__main__":
    main()
