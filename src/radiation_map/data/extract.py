"""Streaming extractor and spatial sampler for Safecast Parquet shards."""

import glob
import os
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rich.progress import track
from radiation_map.config import DataConfig
from radiation_map.data.transforms import normalize_measurement_units


def extract_and_sample_dataset(
    safecast_dir: Path,
    output_dir: Path,
    target_samples: int = 600_000,
    max_shards: Optional[int] = None,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stream Safecast Parquet shards and create spatial-balanced train/val/test splits."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    files = sorted(glob.glob(str(safecast_dir / "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {safecast_dir}")
        
    if max_shards is not None:
        files = files[:max_shards]
        
    print(f"Found {len(files)} Safecast parquet shards. Beginning streaming extraction...")
    
    # We will sample from across all shards to capture the entire temporal span
    samples_per_shard = max(1000, int(target_samples * 1.5 / len(files)))
    
    rng = np.random.RandomState(seed)
    collected_dfs = []
    total_rows_scanned = 0
    
    # Stratified regions for spatial balancing:
    # 1: Japan / Asia-Pacific (lat 20 to 50, lon 120 to 150)
    # 2: Europe & Ukraine (lat 35 to 70, lon -15 to 45)
    # 3: North America (lat 24 to 72, lon -170 to -50)
    # 4: Rest of World (South America, Africa, Oceania, Central/South Asia)
    
    for shard_idx, file_path in enumerate(files):
        try:
            # Read columns needed
            table = pq.read_table(
                file_path,
                columns=["captured_at", "latitude", "longitude", "value", "unit", "location_name", "device_id"]
            )
            df = table.to_pandas()
            total_rows_scanned += len(df)
            
            # Clean and normalize units
            df_clean = normalize_measurement_units(df)
            if len(df_clean) == 0:
                continue
                
            # Sample subset from this shard
            if len(df_clean) > samples_per_shard:
                # Prioritize elevated readings/hotspots slightly (20% of sample) and normal readings (80%)
                is_elevated = df_clean["radiation_usv"] >= 0.20
                elevated_df = df_clean[is_elevated]
                normal_df = df_clean[~is_elevated]
                
                n_elevated = min(len(elevated_df), int(samples_per_shard * 0.25))
                n_normal = min(len(normal_df), samples_per_shard - n_elevated)
                
                sampled_parts = []
                if n_elevated > 0:
                    sampled_parts.append(elevated_df.sample(n=n_elevated, random_state=seed + shard_idx))
                if n_normal > 0:
                    sampled_parts.append(normal_df.sample(n=n_normal, random_state=seed + shard_idx))
                    
                if sampled_parts:
                    sampled_df = pd.concat(sampled_parts, ignore_index=True)
                else:
                    sampled_df = df_clean.sample(n=samples_per_shard, random_state=seed + shard_idx)
            else:
                sampled_df = df_clean
                
            collected_dfs.append(sampled_df)
            
            if (shard_idx + 1) % 25 == 0 or (shard_idx + 1) == len(files):
                print(f"Processed {shard_idx + 1}/{len(files)} shards ({total_rows_scanned:,} total rows scanned)")
                
        except Exception as e:
            print(f"Warning: Failed to read shard {file_path}: {e}")
            continue
            
    if not collected_dfs:
        raise ValueError("No valid records extracted from dataset shards.")
        
    full_df = pd.concat(collected_dfs, ignore_index=True)
    print(f"Total extracted candidates: {len(full_df):,}")
    
    # Add spatial grid hashing (0.1 degree ~ 11km) for spatial deduplication and stratified splitting
    full_df["grid_lat"] = np.round(full_df["latitude"], decimals=2)
    full_df["grid_lon"] = np.round(full_df["longitude"], decimals=2)
    full_df["spatial_cell"] = full_df["grid_lat"].astype(str) + "_" + full_df["grid_lon"].astype(str)
    
    # Shuffle full dataset
    full_df = full_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    
    # Limit to target sample size if exceeds
    if len(full_df) > target_samples:
        # Spatial stratified sampling
        unique_cells = full_df["spatial_cell"].unique()
        print(f"Unique 0.01-deg spatial cells captured: {len(unique_cells):,}")
        full_df = full_df.iloc[:target_samples].copy()
        
    # Create spatial block train / val / test split by spatial cells to prevent leakage
    unique_cells = full_df["spatial_cell"].unique()
    rng.shuffle(unique_cells)
    
    n_cells = len(unique_cells)
    n_test_cells = int(n_cells * 0.15)
    n_val_cells = int(n_cells * 0.15)
    
    test_cells = set(unique_cells[:n_test_cells])
    val_cells = set(unique_cells[n_test_cells:n_test_cells + n_val_cells])
    train_cells = set(unique_cells[n_test_cells + n_val_cells:])
    
    train_df = full_df[full_df["spatial_cell"].isin(train_cells)].copy()
    val_df = full_df[full_df["spatial_cell"].isin(val_cells)].copy()
    test_df = full_df[full_df["spatial_cell"].isin(test_cells)].copy()
    
    print(f"Dataset splits created:")
    print(f"  Train: {len(train_df):,} samples ({len(train_cells):,} spatial cells)")
    print(f"  Validation: {len(val_df):,} samples ({len(val_cells):,} spatial cells)")
    print(f"  Test: {len(test_df):,} samples ({len(test_cells):,} spatial cells)")
    
    # Save to Parquet
    train_path = output_dir / "radiation_train.parquet"
    val_path = output_dir / "radiation_val.parquet"
    test_path = output_dir / "radiation_test.parquet"
    
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)
    
    print(f"Saved processed dataset splits to {output_dir}")
    return train_df, val_df, test_df
