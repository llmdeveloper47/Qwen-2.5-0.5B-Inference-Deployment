#!/usr/bin/env python3
"""
Parameterized load testing script using Locust programmatically.

This script runs Locust tests with different configurations and saves results.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def run_locust_test(endpoint_id: str,
                   api_key: str,
                   batch_size: int,
                   num_users: int = 10,
                   spawn_rate: int = 2,
                   duration: str = "5m",
                   output_dir: str = "./results/load_tests") -> dict:
    """
    Run a Locust load test with specified parameters.
    
    Args:
        endpoint_id: RunPod endpoint ID
        api_key: RunPod API key
        batch_size: Batch size for classification requests
        num_users: Number of simulated users
        spawn_rate: User spawn rate (users/second)
        duration: Test duration (e.g., "5m", "1h")
        output_dir: Directory to save results
        
    Returns:
        Dictionary with test results
    """
    print("\n" + "=" * 70)
    print(f"Load Test Configuration")
    print("=" * 70)
    print(f"  Endpoint ID: {endpoint_id}")
    print(f"  Batch size: {batch_size}")
    print(f"  Users: {num_users}")
    print(f"  Spawn rate: {spawn_rate}")
    print(f"  Duration: {duration}")
    
    # Create output directory
    output_path = Path(output_dir) / f"batch_{batch_size}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # HTML report path
    html_report = output_path / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    
    # CSV stats path
    csv_stats = output_path / f"stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    # Build Locust command
    cmd = [
        "locust",
        "-f", "locustfile.py",
        "--headless",
        "-u", str(num_users),
        "-r", str(spawn_rate),
        "--run-time", duration,
        "--html", str(html_report),
        "--csv", str(csv_stats.with_suffix("")),  # Locust adds .csv extensions
        "--host", f"https://api.runpod.ai/v2/{endpoint_id}",
    ]
    
    # Set environment variables for Locust
    env = {
        **dict(os.environ),
        "RUNPOD_API_KEY": api_key,
        "RUNPOD_ENDPOINT_ID": endpoint_id,
        "BATCH_SIZE": str(batch_size),
    }
    
    print(f"\nStarting Locust test...")
    print(f"  Command: {' '.join(cmd)}")
    
    try:
        # Run Locust
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            check=True
        )
        
        print(result.stdout)
        
        # Parse results from CSV (Locust generates _stats.csv, _failures.csv, etc.)
        stats_file = csv_stats.parent / f"{csv_stats.stem}_stats.csv"
        
        if stats_file.exists():
            print(f"\n✓ Test complete. Results saved to:")
            print(f"    HTML: {html_report}")
            print(f"    CSV: {stats_file}")
            
            # Parse key metrics from CSV
            metrics = parse_locust_stats(stats_file)
            
            # Save summary
            summary = {
                "endpoint_id": endpoint_id,
                "batch_size": batch_size,
                "num_users": num_users,
                "spawn_rate": spawn_rate,
                "duration": duration,
                "timestamp": datetime.now().isoformat(),
                "metrics": metrics,
            }
            
            summary_file = output_path / "summary.json"
            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2)
            
            return summary
        else:
            print(f"⚠ Stats file not found: {stats_file}")
            return {}
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Locust test failed:")
        print(e.stderr)
        return {}
    except Exception as e:
        print(f"✗ Error running test: {str(e)}")
        return {}


def parse_locust_stats(csv_file: Path) -> dict:
    """Parse Locust stats CSV file."""
    import csv
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Get aggregate row (usually last row with name="Aggregated")
    agg_row = None
    for row in rows:
        if row.get("Type") == "Aggregated" or row.get("Name") == "Aggregated":
            agg_row = row
            break
    
    if not agg_row:
        # If no aggregate, use first row
        agg_row = rows[0] if rows else {}
    
    return {
        "request_count": int(agg_row.get("Request Count", 0)),
        "failure_count": int(agg_row.get("Failure Count", 0)),
        "avg_response_time": float(agg_row.get("Average Response Time", 0)),
        "min_response_time": float(agg_row.get("Min Response Time", 0)),
        "max_response_time": float(agg_row.get("Max Response Time", 0)),
        "requests_per_sec": float(agg_row.get("Requests/s", 0)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run parameterized load tests with Locust"
    )
    parser.add_argument(
        "--endpoint-id",
        type=str,
        required=True,
        help="RunPod endpoint ID"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        required=True,
        help="RunPod API key"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for requests"
    )
    parser.add_argument(
        "--users",
        type=int,
        default=10,
        help="Number of simulated users"
    )
    parser.add_argument(
        "--spawn-rate",
        type=int,
        default=2,
        help="User spawn rate (users/second)"
    )
    parser.add_argument(
        "--duration",
        type=str,
        default="5m",
        help="Test duration (e.g., 5m, 1h)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/load_tests",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Run load test
    summary = run_locust_test(
        endpoint_id=args.endpoint_id,
        api_key=args.api_key,
        batch_size=args.batch_size,
        num_users=args.users,
        spawn_rate=args.spawn_rate,
        duration=args.duration,
        output_dir=args.output_dir
    )
    
    if summary:
        print("\n✓ Load test completed successfully")
    else:
        print("\n✗ Load test failed")
        sys.exit(1)


if __name__ == "__main__":
    main()

