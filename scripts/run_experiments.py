#!/usr/bin/env python3
"""
Automated experiment runner for quantization and batch size testing.

This script systematically tests all combinations of:
- Quantization methods: none, bitsandbytes, awq, gptq
- Batch sizes: 1, 4, 8, 16, 32

For each configuration, it:
1. Updates the RunPod endpoint environment variables
2. Waits for deployment to complete
3. Runs latency tests
4. Saves results to structured output directory
"""

import argparse
import json
import time
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

import requests
import numpy as np
from tqdm import tqdm


class ExperimentRunner:
    """Manages the full experiment pipeline."""
    
    def __init__(self, api_key: str, endpoint_id: str):
        self.api_key = api_key
        self.endpoint_id = endpoint_id
        self.base_url = f"https://api.runpod.ai/v2/{endpoint_id}"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def update_endpoint_config(self, quantization: str, max_num_seqs: int) -> bool:
        """Update endpoint environment variables."""
        print(f"\nUpdating endpoint configuration...")
        print(f"  Quantization: {quantization}")
        print(f"  MAX_NUM_SEQS: {max_num_seqs}")
        
        # Note: RunPod GraphQL API required for updating environment variables
        # This is a simplified example; in practice you may need to use the GraphQL endpoint
        
        query = """
        mutation updateEndpoint($input: UpdateEndpointInput!) {
          updateEndpoint(input: $input) {
            id
            name
          }
        }
        """
        
        variables = {
            "input": {
                "endpointId": self.endpoint_id,
                "environment": {
                    "QUANTIZATION": quantization,
                    "MAX_NUM_SEQS": str(max_num_seqs)
                }
            }
        }
        
        try:
            # For simplicity, we'll assume manual configuration
            # In production, use the RunPod GraphQL API
            print(f"  ⚠ Note: Please manually update the endpoint configuration in RunPod console")
            print(f"     QUANTIZATION={quantization}, MAX_NUM_SEQS={max_num_seqs}")
            print(f"  Waiting 60s for configuration update...")
            time.sleep(60)
            return True
            
        except Exception as e:
            print(f"  ✗ Failed to update endpoint: {str(e)}")
            return False
    
    def wait_for_endpoint_ready(self, timeout: int = 300) -> bool:
        """Wait for endpoint to be ready after configuration change."""
        print(f"  Waiting for endpoint to be ready (timeout={timeout}s)...")
        
        start = time.time()
        
        while time.time() - start < timeout:
            try:
                # Send a health check request
                health_check = {
                    "input": {
                        "prompts": ["test"]
                    }
                }
                
                response = requests.post(
                    f"{self.base_url}/run",
                    headers=self.headers,
                    json=health_check,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Check if output exists and is valid
                    if "output" in result or "result" in result:
                        print(f"  ✓ Endpoint is ready")
                        return True
                
                time.sleep(10)
                
            except Exception as e:
                time.sleep(10)
                continue
        
        print(f"  ✗ Endpoint not ready after {timeout}s")
        return False
    
    def run_latency_test(self, 
                        prompts: List[str], 
                        batch_size: int, 
                        num_iterations: int = 20) -> Dict[str, Any]:
        """Run latency test with specified batch size."""
        print(f"\n  Testing batch_size={batch_size} ({num_iterations} iterations)...")
        
        # Prepare batches
        batch = prompts[:batch_size]
        
        latencies = []
        successful = 0
        failed = 0
        
        for i in tqdm(range(num_iterations), desc=f"    Batch {batch_size}", leave=False):
            payload = {
                "input": {
                    "prompts": batch
                }
            }
            
            start = time.time()
            
            try:
                response = requests.post(
                    f"{self.base_url}/run",
                    headers=self.headers,
                    json=payload,
                    timeout=120
                )
                
                elapsed = (time.time() - start) * 1000  # Convert to ms
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Extract output
                    if "output" in result:
                        output = result["output"]
                    elif "result" in result:
                        output = result["result"]
                    else:
                        output = result
                    
                    # Validate response
                    if "results" in output and len(output["results"]) == batch_size:
                        latencies.append(elapsed)
                        successful += 1
                    else:
                        failed += 1
                else:
                    failed += 1
                    
            except Exception as e:
                failed += 1
        
        if not latencies:
            print(f"    ✗ All requests failed")
            return None
        
        # Calculate statistics
        stats = {
            "batch_size": batch_size,
            "num_iterations": num_iterations,
            "successful": successful,
            "failed": failed,
            "avg_latency_ms": round(np.mean(latencies), 2),
            "min_latency_ms": round(np.min(latencies), 2),
            "max_latency_ms": round(np.max(latencies), 2),
            "p50_latency_ms": round(np.percentile(latencies, 50), 2),
            "p90_latency_ms": round(np.percentile(latencies, 90), 2),
            "p95_latency_ms": round(np.percentile(latencies, 95), 2),
            "p99_latency_ms": round(np.percentile(latencies, 99), 2),
            "std_latency_ms": round(np.std(latencies), 2),
            "throughput": round(batch_size / (np.mean(latencies) / 1000), 2),
        }
        
        print(f"    ✓ Avg: {stats['avg_latency_ms']:.2f}ms, "
              f"P95: {stats['p95_latency_ms']:.2f}ms, "
              f"Throughput: {stats['throughput']:.2f} samples/s")
        
        return stats
    
    def run_experiment(self, 
                      quantization: str, 
                      batch_sizes: List[int],
                      test_prompts: List[str],
                      num_iterations: int = 20) -> List[Dict]:
        """Run full experiment for one quantization method."""
        print("\n" + "=" * 70)
        print(f"Experiment: Quantization={quantization}")
        print("=" * 70)
        
        # Update endpoint configuration
        # Note: MAX_NUM_SEQS set to max batch size for flexibility
        max_batch = max(batch_sizes)
        success = self.update_endpoint_config(quantization, max_batch)
        
        if not success:
            print(f"✗ Failed to configure endpoint for {quantization}")
            return []
        
        # Wait for endpoint to be ready
        if not self.wait_for_endpoint_ready():
            print(f"✗ Endpoint not ready for {quantization}")
            return []
        
        # Run tests for each batch size
        results = []
        
        for batch_size in batch_sizes:
            stats = self.run_latency_test(test_prompts, batch_size, num_iterations)
            
            if stats:
                stats['quantization'] = quantization
                stats['timestamp'] = datetime.now().isoformat()
                results.append(stats)
        
        return results


def load_test_prompts(num_samples: int = 1000) -> List[str]:
    """Load test prompts from dataset."""
    try:
        from datasets import load_dataset
        
        print("Loading test dataset...")
        dataset = load_dataset("codefactory4791/amazon_test")
        df = dataset["test"].to_pandas()
        
        if 'query' in df.columns:
            df = df.rename(columns={'query': 'text'})
        
        prompts = df['text'].tolist()[:num_samples]
        print(f"  ✓ Loaded {len(prompts)} test prompts")
        
        return prompts
        
    except Exception as e:
        print(f"  ⚠ Could not load dataset: {str(e)}")
        print(f"  Using synthetic test prompts")
        
        # Fallback to synthetic prompts
        base_prompts = [
            "Book a flight to New York",
            "Order pizza for delivery",
            "Play some jazz music",
            "Set an alarm for 7 AM",
            "What's the weather today?",
            "Send an email to John",
            "Schedule a meeting tomorrow",
            "Find a nearby restaurant",
        ]
        
        return base_prompts * (num_samples // len(base_prompts))


def save_results(results: List[Dict], output_dir: str):
    """Save experiment results."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save all results
    all_results_file = output_path / "all_results.json"
    with open(all_results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ All results saved to: {all_results_file}")
    
    # Save by quantization method
    by_quant = {}
    for result in results:
        quant = result['quantization']
        if quant not in by_quant:
            by_quant[quant] = []
        by_quant[quant].append(result)
    
    for quant, quant_results in by_quant.items():
        quant_dir = output_path / quant
        quant_dir.mkdir(exist_ok=True)
        
        for result in quant_results:
            batch_size = result['batch_size']
            result_file = quant_dir / f"batch_{batch_size}.json"
            
            with open(result_file, "w") as f:
                json.dump(result, f, indent=2)
    
    print(f"✓ Individual results saved to: {output_path}")
    
    # Generate summary CSV
    summary_file = output_path / "summary.csv"
    
    with open(summary_file, "w") as f:
        # Header
        f.write("quantization,batch_size,avg_latency_ms,p95_latency_ms,p99_latency_ms,throughput\n")
        
        # Data
        for result in results:
            f.write(
                f"{result['quantization']},"
                f"{result['batch_size']},"
                f"{result['avg_latency_ms']},"
                f"{result['p95_latency_ms']},"
                f"{result['p99_latency_ms']},"
                f"{result['throughput']}\n"
            )
    
    print(f"✓ Summary CSV saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run systematic quantization and batch size experiments"
    )
    parser.add_argument(
        "--endpoint-id",
        type=str,
        required=True,
        help="RunPod endpoint ID (or base ID if using multiple endpoints)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        required=True,
        help="RunPod API key"
    )
    parser.add_argument(
        "--quantization-methods",
        nargs="+",
        default=["none", "bitsandbytes"],
        help="Quantization methods to test"
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,4,8,16,32",
        help="Comma-separated batch sizes to test"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="Number of iterations per configuration"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of samples from dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/experiments",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Parse batch sizes
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]
    
    print("=" * 70)
    print("Quantization & Batch Size Experiment Runner")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Endpoint ID: {args.endpoint_id}")
    print(f"  Quantization methods: {args.quantization_methods}")
    print(f"  Batch sizes: {batch_sizes}")
    print(f"  Iterations per config: {args.iterations}")
    print(f"  Output directory: {args.output_dir}")
    
    # Load test prompts
    test_prompts = load_test_prompts(args.num_samples)
    
    # Initialize experiment runner
    runner = ExperimentRunner(args.api_key, args.endpoint_id)
    
    # Run experiments for each quantization method
    all_results = []
    
    total_experiments = len(args.quantization_methods) * len(batch_sizes)
    current_experiment = 0
    
    print(f"\nTotal experiments to run: {total_experiments}")
    print("=" * 70)
    
    for quantization in args.quantization_methods:
        results = runner.run_experiment(
            quantization=quantization,
            batch_sizes=batch_sizes,
            test_prompts=test_prompts,
            num_iterations=args.iterations
        )
        
        all_results.extend(results)
        current_experiment += len(batch_sizes)
        
        print(f"\nProgress: {current_experiment}/{total_experiments} experiments complete")
    
    # Save all results
    save_results(all_results, args.output_dir)
    
    # Print final summary
    print("\n" + "=" * 70)
    print("Experiment Summary")
    print("=" * 70)
    
    print(f"\n{'Quantization':<15} {'Batch':<7} {'Avg (ms)':<10} {'P95 (ms)':<10} {'Throughput':<12}")
    print("-" * 70)
    
    for result in all_results:
        print(
            f"{result['quantization']:<15} "
            f"{result['batch_size']:<7} "
            f"{result['avg_latency_ms']:<10.2f} "
            f"{result['p95_latency_ms']:<10.2f} "
            f"{result['throughput']:<12.2f}"
        )
    
    print("\n" + "=" * 70)
    print("✓ All experiments complete!")
    print("=" * 70)
    
    print(f"\nResults saved to: {args.output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Analyze results: python scripts/analyze_results.py --results-dir {args.output_dir}")
    print(f"  2. Generate report: python scripts/generate_report.py --results-dir {args.output_dir}")


if __name__ == "__main__":
    main()

