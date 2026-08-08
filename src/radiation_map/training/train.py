"""Training loop and optimization for SpatialHarmonicNet."""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from rich.console import Console
from radiation_map.models.spatial_net import SpatialHarmonicNet
from radiation_map.training.loss import HeteroscedasticGaussianLoss
from radiation_map.data.dataset import SpatialRadiationDataset

console = Console()


def train_spatial_model(
    model: SpatialHarmonicNet,
    train_dataset: SpatialRadiationDataset,
    val_dataset: SpatialRadiationDataset,
    epochs: int = 25,
    batch_size: int = 2048,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    checkpoint_dir: Path = Path("checkpoints"),
    device: Optional[torch.device] = None,
    patience: int = 7,
) -> Dict[str, list]:
    """Train SpatialHarmonicNet with GPU acceleration and validation tracking."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(device)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    
    criterion = HeteroscedasticGaussianLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )
    
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_mae": [],
        "val_rmse": [],
    }
    
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = checkpoint_dir / "best_spatial_harmonic_net.pt"
    
    console.print(f"[bold green]Starting training on {device} ({len(train_dataset):,} train, {len(val_dataset):,} val)[/bold green]")
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        
        for batch in train_loader:
            coords = batch["coords_cartesian"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                mu, log_var = model(coords)
                loss = criterion(mu, log_var, targets)
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_losses.append(loss.item())
            
        epoch_train_loss = float(np.mean(train_losses))
        
        # Validation pass
        model.eval()
        val_losses = []
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                coords = batch["coords_cartesian"].to(device)
                targets = batch["target"].to(device)
                
                mu, log_var = model(coords)
                loss = criterion(mu, log_var, targets)
                
                val_losses.append(loss.item())
                val_preds.append(mu.cpu().numpy())
                val_targets.append(targets.cpu().numpy())
                
        epoch_val_loss = float(np.mean(val_losses))
        val_preds_arr = np.vstack(val_preds).flatten()
        val_targets_arr = np.vstack(val_targets).flatten()
        
        val_mae = float(np.mean(np.abs(val_preds_arr - val_targets_arr)))
        val_rmse = float(np.sqrt(np.mean((val_preds_arr - val_targets_arr) ** 2)))
        
        scheduler.step(epoch_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["val_mae"].append(val_mae)
        history["val_rmse"].append(val_rmse)
        
        console.print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | "
            f"Val MAE: {val_mae:.4f} | "
            f"Val RMSE: {val_rmse:.4f} | "
            f"LR: {current_lr:.1e}"
        )
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": epoch_val_loss,
                    "val_mae": val_mae,
                    "val_rmse": val_rmse,
                },
                best_model_path,
            )
            console.print(f"  [bold cyan]Saved new best checkpoint (Val Loss: {best_val_loss:.4f})[/bold cyan]")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                console.print(f"[bold yellow]Early stopping triggered after {epoch} epochs.[/bold yellow]")
                break
                
    # Load best weights before returning
    if best_model_path.exists():
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        console.print(f"[bold green]Restored best model from epoch {checkpoint['epoch']} with Val Loss: {checkpoint['val_loss']:.4f}[/bold green]")
        
    return history
