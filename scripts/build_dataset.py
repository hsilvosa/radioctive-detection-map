"""Extract and prepare spatial training dataset from Safecast Parquet shards."""

import argparse
from pathlib import Path
from radiation_map.config import DataConfig
from radiation_map.data.extract import extract_and_sample_dataset


def main():
    parser = argparse.ArgumentParser(description="Extract spatial dataset from Safecast Parquet shards")
    parser.add_argument("--safecast-dir", type=str, default=r"D:\datasets\safecast\data\processed\measurements")
    parser.add_argument("--output-dir", type=str, default="data/processed")
    parser.add_argument("--samples", type=int, default=600_000)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    safecast_path = Path(args.safecast_dir)
    output_path = Path(args.output_dir)

    print(f"Extracting from: {safecast_path}")
    print(f"Output to: {output_path}")
    print(f"Target sample size: {args.samples:,}")

    train_df, val_df, test_df = extract_and_sample_dataset(
        safecast_dir=safecast_path,
        output_dir=output_path,
        target_samples=args.samples,
        max_shards=args.max_shards,
        seed=args.seed,
    )
    print(f"Extraction complete! Total processed records: {len(train_df) + len(val_df) + len(test_df):,}")


if __name__ == "__main__":
    main()
