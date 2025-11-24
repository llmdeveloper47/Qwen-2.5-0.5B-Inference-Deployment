#!/usr/bin/env python3
"""
Analyze experiment results and generate comparison visualizations.

This script loads all experiment results and creates:
- Comparison tables
- Latency vs batch size plots
- Throughput comparisons
- Memory usage analysis
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_results(results_dir: str) -> pd.DataFrame:
    """Load all experiment results into a DataFrame."""
    results_path = Path(results_dir)
    
    # Check for both local and endpoint results
    all_results = []
    
    # Load local benchmark results
    local_dir = results_path / "local_benchmarks"
    if local_dir.exists():
        for quant_dir in local_dir.iterdir():
            if quant_dir.is_dir():
                result_file = quant_dir / "benchmark_results.json"
                if result_file.exists():
                    with open(result_file) as f:
                        data = json.load(f)
                        for entry in data:
                            entry['source'] = 'local'
                            all_results.append(entry)
    
    # Load endpoint test results
    endpoint_dir = results_path / "experiments"
    if endpoint_dir.exists():
        for quant_dir in endpoint_dir.iterdir():
            if quant_dir.is_dir():
                for batch_file in quant_dir.glob("batch_*.json"):
                    with open(batch_file) as f:
                        data = json.load(f)
                        data['source'] = 'endpoint'
                        all_results.append(data)
    
    if not all_results:
        print(f"✗ No results found in {results_dir}")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_results)
    
    print(f"✓ Loaded {len(df)} experiment results")
    print(f"  Sources: {df['source'].unique().tolist()}")
    print(f"  Quantization methods: {df['quantization'].unique().tolist()}")
    print(f"  Batch sizes: {sorted(df['batch_size'].unique().tolist())}")
    
    return df


def create_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create summary comparison table."""
    # Pivot table with quantization and batch size
    summary = df.groupby(['quantization', 'batch_size']).agg({
        'avg_latency_ms': 'mean',
        'p95_latency_ms': 'mean',
        'p99_latency_ms': 'mean',
        'throughput': 'mean',
    }).reset_index()
    
    # Round to 2 decimal places
    for col in ['avg_latency_ms', 'p95_latency_ms', 'p99_latency_ms', 'throughput']:
        summary[col] = summary[col].round(2)
    
    return summary


def plot_latency_comparison(df: pd.DataFrame, output_dir: str):
    """Create latency comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Average latency vs batch size
    ax = axes[0, 0]
    for quant in df['quantization'].unique():
        data = df[df['quantization'] == quant]
        ax.plot(data['batch_size'], data['avg_latency_ms'], 
               marker='o', label=quant, linewidth=2)
    ax.set_xlabel('Batch Size', fontsize=12)
    ax.set_ylabel('Average Latency (ms)', fontsize=12)
    ax.set_title('Average Latency vs Batch Size', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: P95 latency vs batch size
    ax = axes[0, 1]
    for quant in df['quantization'].unique():
        data = df[df['quantization'] == quant]
        ax.plot(data['batch_size'], data['p95_latency_ms'], 
               marker='s', label=quant, linewidth=2)
    ax.set_xlabel('Batch Size', fontsize=12)
    ax.set_ylabel('P95 Latency (ms)', fontsize=12)
    ax.set_title('P95 Latency vs Batch Size', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Throughput vs batch size
    ax = axes[1, 0]
    for quant in df['quantization'].unique():
        data = df[df['quantization'] == quant]
        ax.plot(data['batch_size'], data['throughput'], 
               marker='^', label=quant, linewidth=2)
    ax.set_xlabel('Batch Size', fontsize=12)
    ax.set_ylabel('Throughput (samples/s)', fontsize=12)
    ax.set_title('Throughput vs Batch Size', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Heatmap of latencies
    ax = axes[1, 1]
    pivot = df.pivot_table(
        values='p95_latency_ms',
        index='quantization',
        columns='batch_size',
        aggfunc='mean'
    )
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'P95 Latency (ms)'})
    ax.set_title('P95 Latency Heatmap', fontsize=14, fontweight='bold')
    ax.set_xlabel('Batch Size', fontsize=12)
    ax.set_ylabel('Quantization Method', fontsize=12)
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir) / "latency_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Latency comparison plot saved to: {output_path}")
    
    plt.close()


def plot_efficiency_analysis(df: pd.DataFrame, output_dir: str):
    """Create efficiency analysis plots."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Latency per sample vs batch size
    ax = axes[0]
    for quant in df['quantization'].unique():
        data = df[df['quantization'] == quant].copy()
        data['latency_per_sample'] = data['avg_latency_ms'] / data['batch_size']
        ax.plot(data['batch_size'], data['latency_per_sample'], 
               marker='o', label=quant, linewidth=2)
    ax.set_xlabel('Batch Size', fontsize=12)
    ax.set_ylabel('Latency per Sample (ms)', fontsize=12)
    ax.set_title('Per-Sample Latency vs Batch Size', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Efficiency (throughput / latency)
    ax = axes[1]
    for quant in df['quantization'].unique():
        data = df[df['quantization'] == quant].copy()
        data['efficiency'] = data['throughput'] / data['avg_latency_ms'] * 1000
        ax.plot(data['batch_size'], data['efficiency'], 
               marker='s', label=quant, linewidth=2)
    ax.set_xlabel('Batch Size', fontsize=12)
    ax.set_ylabel('Efficiency (samples/s per ms)', fontsize=12)
    ax.set_title('Inference Efficiency vs Batch Size', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir) / "efficiency_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Efficiency analysis plot saved to: {output_path}")
    
    plt.close()


def generate_recommendations(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate recommendations based on results."""
    recommendations = {
        "best_latency": {},
        "best_throughput": {},
        "best_balance": {},
    }
    
    # Find best configuration for latency (p95)
    best_latency_row = df.loc[df['p95_latency_ms'].idxmin()]
    recommendations['best_latency'] = {
        "quantization": best_latency_row['quantization'],
        "batch_size": int(best_latency_row['batch_size']),
        "p95_latency_ms": round(best_latency_row['p95_latency_ms'], 2),
        "throughput": round(best_latency_row['throughput'], 2),
    }
    
    # Find best configuration for throughput
    best_throughput_row = df.loc[df['throughput'].idxmax()]
    recommendations['best_throughput'] = {
        "quantization": best_throughput_row['quantization'],
        "batch_size": int(best_throughput_row['batch_size']),
        "p95_latency_ms": round(best_throughput_row['p95_latency_ms'], 2),
        "throughput": round(best_throughput_row['throughput'], 2),
    }
    
    # Find best balanced configuration (high throughput, low latency)
    # Use normalized scores
    df_norm = df.copy()
    df_norm['throughput_norm'] = (df_norm['throughput'] - df_norm['throughput'].min()) / (df_norm['throughput'].max() - df_norm['throughput'].min())
    df_norm['latency_norm'] = 1 - (df_norm['p95_latency_ms'] - df_norm['p95_latency_ms'].min()) / (df_norm['p95_latency_ms'].max() - df_norm['p95_latency_ms'].min())
    df_norm['balance_score'] = (df_norm['throughput_norm'] + df_norm['latency_norm']) / 2
    
    best_balance_row = df_norm.loc[df_norm['balance_score'].idxmax()]
    recommendations['best_balance'] = {
        "quantization": best_balance_row['quantization'],
        "batch_size": int(best_balance_row['batch_size']),
        "p95_latency_ms": round(best_balance_row['p95_latency_ms'], 2),
        "throughput": round(best_balance_row['throughput'], 2),
        "balance_score": round(best_balance_row['balance_score'], 3),
    }
    
    return recommendations


def main():
    parser = argparse.ArgumentParser(
        description="Analyze experiment results and generate visualizations"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./results",
        help="Directory containing experiment results"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/analysis",
        help="Output directory for analysis results"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Analyzing Experiment Results")
    print("=" * 70)
    
    # Load results
    df = load_results(args.results_dir)
    
    if df.empty:
        print("✗ No results to analyze")
        sys.exit(1)
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create comparison table
    print("\nGenerating comparison table...")
    summary = create_comparison_table(df)
    
    # Save summary CSV
    summary_file = output_path / "comparison_table.csv"
    summary.to_csv(summary_file, index=False)
    print(f"  ✓ Saved to: {summary_file}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("Performance Summary")
    print("=" * 70)
    print(summary.to_string(index=False))
    
    # Create visualizations
    print("\nGenerating visualizations...")
    plot_latency_comparison(df, args.output_dir)
    plot_efficiency_analysis(df, args.output_dir)
    
    # Generate recommendations
    print("\nGenerating recommendations...")
    recommendations = generate_recommendations(df)
    
    # Save recommendations
    rec_file = output_path / "recommendations.json"
    with open(rec_file, "w") as f:
        json.dump(recommendations, f, indent=2)
    
    print("\n" + "=" * 70)
    print("Recommendations")
    print("=" * 70)
    
    print("\n[Best for Low Latency]")
    rec = recommendations['best_latency']
    print(f"  Quantization: {rec['quantization']}")
    print(f"  Batch Size: {rec['batch_size']}")
    print(f"  P95 Latency: {rec['p95_latency_ms']}ms")
    print(f"  Throughput: {rec['throughput']} samples/s")
    
    print("\n[Best for High Throughput]")
    rec = recommendations['best_throughput']
    print(f"  Quantization: {rec['quantization']}")
    print(f"  Batch Size: {rec['batch_size']}")
    print(f"  P95 Latency: {rec['p95_latency_ms']}ms")
    print(f"  Throughput: {rec['throughput']} samples/s")
    
    print("\n[Best Balanced]")
    rec = recommendations['best_balance']
    print(f"  Quantization: {rec['quantization']}")
    print(f"  Batch Size: {rec['batch_size']}")
    print(f"  P95 Latency: {rec['p95_latency_ms']}ms")
    print(f"  Throughput: {rec['throughput']} samples/s")
    print(f"  Balance Score: {rec['balance_score']}")
    
    print("\n" + "=" * 70)
    print(f"✓ Analysis complete!")
    print("=" * 70)
    print(f"\nResults saved to: {args.output_dir}")
    print(f"  - comparison_table.csv")
    print(f"  - latency_comparison.png")
    print(f"  - efficiency_analysis.png")
    print(f"  - recommendations.json")


if __name__ == "__main__":
    main()

