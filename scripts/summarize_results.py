#!/usr/bin/env python3
"""
Generate summary CSV from all experiment results.

This script consolidates results from all experiments into a single CSV file
for easy analysis in spreadsheet software or further processing.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def collect_all_results(results_dir: str) -> pd.DataFrame:
    """Collect all experiment results into a single DataFrame."""
    results_path = Path(results_dir)
    all_data = []
    
    # Scan for all JSON result files
    patterns = [
        "local_benchmarks/**/benchmark_results.json",
        "local_benchmarks/**/batch_*.json",
        "experiments/**/batch_*.json",
        "experiments/**/all_results.json",
        "endpoint_tests/**/batch_*.json",
    ]
    
    for pattern in patterns:
        for file in results_path.glob(pattern):
            try:
                with open(file) as f:
                    data = json.load(f)
                    
                    # Handle both single dict and list of dicts
                    if isinstance(data, list):
                        all_data.extend(data)
                    else:
                        all_data.append(data)
                        
            except Exception as e:
                print(f"⚠ Could not load {file}: {str(e)}")
    
    if not all_data:
        print(f"✗ No results found in {results_dir}")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_data)
    
    # Remove duplicates
    before_count = len(df)
    df = df.drop_duplicates()
    after_count = len(df)
    
    if before_count > after_count:
        print(f"  Removed {before_count - after_count} duplicate entries")
    
    return df


def create_summary_csv(df: pd.DataFrame, output_file: str):
    """Create summary CSV with key metrics."""
    # Select relevant columns
    columns = [
        'quantization',
        'batch_size',
        'total_samples',
        'avg_latency_ms',
        'p50_latency_ms',
        'p95_latency_ms',
        'p99_latency_ms',
        'throughput',
        'source',  # local or endpoint
    ]
    
    # Keep only columns that exist
    available_columns = [col for col in columns if col in df.columns]
    summary = df[available_columns].copy()
    
    # Sort by quantization and batch size
    if 'quantization' in summary.columns and 'batch_size' in summary.columns:
        summary = summary.sort_values(['quantization', 'batch_size'])
    
    # Save to CSV
    summary.to_csv(output_file, index=False)
    
    print(f"✓ Summary CSV created: {output_file}")
    print(f"  Rows: {len(summary)}")
    print(f"  Columns: {len(summary.columns)}")


def create_pivot_tables(df: pd.DataFrame, output_dir: str):
    """Create pivot tables for different metrics."""
    output_path = Path(output_dir)
    
    metrics = ['avg_latency_ms', 'p95_latency_ms', 'throughput']
    
    for metric in metrics:
        if metric not in df.columns:
            continue
        
        try:
            pivot = df.pivot_table(
                values=metric,
                index='quantization',
                columns='batch_size',
                aggfunc='mean'
            )
            
            # Round to 2 decimal places
            pivot = pivot.round(2)
            
            # Save pivot table
            pivot_file = output_path / f"pivot_{metric}.csv"
            pivot.to_csv(pivot_file)
            
            print(f"✓ Pivot table created: {pivot_file}")
            
        except Exception as e:
            print(f"⚠ Could not create pivot for {metric}: {str(e)}")


def create_statistics_summary(df: pd.DataFrame, output_file: str):
    """Create detailed statistics summary."""
    stats = []
    
    # Group by quantization method
    for quant in df['quantization'].unique():
        quant_data = df[df['quantization'] == quant]
        
        stat = {
            'quantization': quant,
            'num_experiments': len(quant_data),
            'avg_latency_mean': quant_data['avg_latency_ms'].mean(),
            'avg_latency_std': quant_data['avg_latency_ms'].std(),
            'p95_latency_mean': quant_data['p95_latency_ms'].mean(),
            'p95_latency_std': quant_data['p95_latency_ms'].std(),
            'throughput_mean': quant_data['throughput'].mean(),
            'throughput_std': quant_data['throughput'].std(),
            'best_throughput': quant_data['throughput'].max(),
            'best_latency': quant_data['p95_latency_ms'].min(),
        }
        
        stats.append(stat)
    
    stats_df = pd.DataFrame(stats)
    stats_df = stats_df.round(2)
    
    stats_df.to_csv(output_file, index=False)
    print(f"✓ Statistics summary created: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate summary CSV from experiment results"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./results",
        help="Directory containing experiment results"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./results/summary.csv",
        help="Output CSV file path"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Generating Results Summary")
    print("=" * 70)
    
    # Collect all results
    print("\nCollecting experiment results...")
    df = collect_all_results(args.results_dir)
    
    if df.empty:
        print("✗ No results found to summarize")
        sys.exit(1)
    
    print(f"✓ Collected {len(df)} result entries")
    
    # Create output directory
    output_path = Path(args.output).parent
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate summary CSV
    print("\nGenerating summary CSV...")
    create_summary_csv(df, args.output)
    
    # Generate pivot tables
    print("\nGenerating pivot tables...")
    create_pivot_tables(df, output_path)
    
    # Generate statistics summary
    print("\nGenerating statistics summary...")
    stats_file = output_path / "statistics.csv"
    create_statistics_summary(df, stats_file)
    
    print("\n" + "=" * 70)
    print("✓ Summary generation complete!")
    print("=" * 70)
    print(f"\nFiles created:")
    print(f"  - {args.output}")
    print(f"  - {output_path}/pivot_*.csv")
    print(f"  - {output_path}/statistics.csv")


if __name__ == "__main__":
    main()

