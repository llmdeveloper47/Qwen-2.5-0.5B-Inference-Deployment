#!/usr/bin/env python3
"""
Display a terminal dashboard showing current experiment status and results.
"""

import json
import sys
from pathlib import Path
from datetime import datetime


def load_results_summary(results_dir: str = "./results"):
    """Load and summarize all results."""
    results_path = Path(results_dir)
    
    summary = {
        "quantization_methods": {},
        "total_experiments": 0,
        "total_requests": 0,
        "total_failures": 0,
    }
    
    # Scan experiments directory
    exp_dir = results_path / "experiments"
    
    if not exp_dir.exists():
        return summary
    
    for method_dir in exp_dir.iterdir():
        if not method_dir.is_dir():
            continue
        
        method = method_dir.name
        summary["quantization_methods"][method] = {
            "batch_sizes": {},
            "best_throughput": 0,
            "best_latency": float('inf')
        }
        
        for batch_file in method_dir.glob("batch_*.json"):
            try:
                with open(batch_file) as f:
                    data = json.load(f)
                
                bs = data['batch_size']
                summary["quantization_methods"][method]["batch_sizes"][bs] = data
                summary["total_experiments"] += 1
                summary["total_requests"] += data.get('successful', 0)
                summary["total_failures"] += data.get('failed', 0)
                
                # Track bests
                if data['throughput'] > summary["quantization_methods"][method]["best_throughput"]:
                    summary["quantization_methods"][method]["best_throughput"] = data['throughput']
                
                if data['p95_latency_ms'] < summary["quantization_methods"][method]["best_latency"]:
                    summary["quantization_methods"][method]["best_latency"] = data['p95_latency_ms']
                
            except Exception as e:
                print(f"Warning: Could not load {batch_file}: {e}")
    
    return summary


def display_dashboard(summary: dict):
    """Display formatted dashboard."""
    print("\n" + "=" * 80)
    print(" " * 20 + "🚀 INTENT CLASSIFICATION EXPERIMENT DASHBOARD 🚀")
    print("=" * 80)
    
    print(f"\n📊 Overall Statistics")
    print("-" * 80)
    print(f"  Total Experiments Run: {summary['total_experiments']}")
    print(f"  Total Requests: {summary['total_requests']:,}")
    print(f"  Total Failures: {summary['total_failures']:,}")
    
    if summary['total_requests'] > 0:
        success_rate = 100 * (1 - summary['total_failures'] / summary['total_requests'])
        print(f"  Success Rate: {success_rate:.2f}%")
    
    # Quantization methods overview
    print(f"\n📦 Quantization Methods Tested: {len(summary['quantization_methods'])}")
    print("-" * 80)
    
    for method, data in summary['quantization_methods'].items():
        num_batch_sizes = len(data['batch_sizes'])
        print(f"\n  [{method.upper()}]")
        print(f"    Batch sizes tested: {num_batch_sizes}/5")
        print(f"    Best throughput: {data['best_throughput']:.2f} samples/s")
        print(f"    Best P95 latency: {data['best_latency']:.2f} ms")
        
        # Show batch size breakdown
        if data['batch_sizes']:
            print(f"    Tested batch sizes: {sorted(data['batch_sizes'].keys())}")
    
    # Find overall bests
    if summary['quantization_methods']:
        print(f"\n🏆 Best Configurations")
        print("-" * 80)
        
        # Best latency
        best_latency_method = None
        best_latency_value = float('inf')
        
        for method, data in summary['quantization_methods'].items():
            if data['best_latency'] < best_latency_value:
                best_latency_value = data['best_latency']
                best_latency_method = method
        
        print(f"  Lowest Latency: {best_latency_method} @ {best_latency_value:.2f}ms (P95)")
        
        # Best throughput
        best_throughput_method = None
        best_throughput_value = 0
        
        for method, data in summary['quantization_methods'].items():
            if data['best_throughput'] > best_throughput_value:
                best_throughput_value = data['best_throughput']
                best_throughput_method = method
        
        print(f"  Highest Throughput: {best_throughput_method} @ {best_throughput_value:.2f} samples/s")
    
    # Progress tracking
    total_possible = len(summary['quantization_methods']) * 5 if summary['quantization_methods'] else 20
    progress = 100 * summary['total_experiments'] / total_possible if total_possible > 0 else 0
    
    print(f"\n📈 Experiment Progress")
    print("-" * 80)
    print(f"  Completed: {summary['total_experiments']}/20 configurations")
    print(f"  Progress: {progress:.1f}%")
    
    # Show progress bar
    bar_length = 50
    filled = int(bar_length * progress / 100)
    bar = "█" * filled + "░" * (bar_length - filled)
    print(f"  [{bar}] {progress:.1f}%")
    
    # Recommendations
    if summary['total_experiments'] >= 10:
        print(f"\n💡 Recommendations")
        print("-" * 80)
        print(f"  ✓ Sufficient data collected for initial analysis")
        print(f"  → Run: python scripts/analyze_results.py")
        print(f"  → Generate report: python scripts/generate_report.py")
    elif summary['total_experiments'] > 0:
        print(f"\n💡 Next Steps")
        print("-" * 80)
        print(f"  → Continue running experiments")
        print(f"  → Target: 20 configurations (4 methods × 5 batch sizes)")
    else:
        print(f"\n💡 Getting Started")
        print("-" * 80)
        print(f"  → No results found yet")
        print(f"  → Start with: python scripts/test_endpoint.py --latency-test")
    
    print("\n" + "=" * 80)
    
    return summary['total_experiments'], total_possible, []


def display_recent_results(results_dir: str = "./results/experiments", num_recent: int = 5):
    """Display most recent experiment results."""
    results_path = Path(results_dir)
    
    # Find all result files with timestamps
    all_files = []
    
    for file in results_path.rglob("batch_*.json"):
        try:
            with open(file) as f:
                data = json.load(f)
            
            if 'timestamp' in data:
                all_files.append((file, data['timestamp'], data))
        except:
            continue
    
    if not all_files:
        return
    
    # Sort by timestamp (most recent first)
    all_files.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n📅 Recent Results (Last {num_recent})")
    print("-" * 80)
    
    for i, (file, timestamp, data) in enumerate(all_files[:num_recent]):
        print(f"\n  [{i+1}] {file.parent.name}/batch_{data['batch_size']}.json")
        print(f"      Time: {timestamp}")
        print(f"      P95: {data['p95_latency_ms']:.2f}ms | Throughput: {data['throughput']:.2f} samples/s")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Display experiment dashboard"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./results",
        help="Results directory"
    )
    parser.add_argument(
        "--show-recent",
        type=int,
        default=5,
        help="Number of recent results to show"
    )
    
    args = parser.parse_args()
    
    # Load and display summary
    summary = load_results_summary(args.results_dir)
    
    # Display recent results
    if args.show_recent > 0:
        display_recent_results(args.results_dir + "/experiments", args.show_recent)
    
    print()  # Final newline


if __name__ == "__main__":
    main()

