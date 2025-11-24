#!/usr/bin/env python3
"""
Test deployed RunPod endpoint for intent classification.
Sends test requests and validates responses.
"""

import argparse
import json
import time
import sys
from typing import List, Dict, Any

import requests


def test_endpoint(endpoint_id: str, 
                 api_key: str,
                 prompts: List[str],
                 timeout: int = 120) -> Dict[str, Any]:
    """
    Send test request to RunPod endpoint.
    
    Args:
        endpoint_id: RunPod endpoint ID
        api_key: RunPod API key
        prompts: List of prompts to classify
        timeout: Request timeout in seconds
        
    Returns:
        Response dictionary
    """
    url = f"https://api.runpod.ai/v2/{endpoint_id}/run"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "input": {
            "prompts": prompts
        }
    }
    
    print(f"Sending request to: {url}")
    print(f"Number of prompts: {len(prompts)}")
    
    start_time = time.time()
    
    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=timeout
        )
        
        elapsed = time.time() - start_time
        
        print(f"Response received in {elapsed:.3f}s")
        print(f"Status code: {response.status_code}")
        
        if response.status_code != 200:
            print(f"✗ Error: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            return None
        
        result = response.json()
        
        # RunPod wraps the response in a job structure
        # Extract the actual output
        if "output" in result:
            output = result["output"]
        elif "result" in result:
            output = result["result"]
        else:
            output = result
        
        return {
            "output": output,
            "elapsed_time": elapsed,
            "status_code": response.status_code,
            "job_id": result.get("id"),
            "status": result.get("status")
        }
        
    except requests.Timeout:
        print(f"✗ Request timed out after {timeout}s")
        return None
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def validate_response(response: Dict[str, Any], expected_num_prompts: int) -> bool:
    """Validate response structure and content."""
    if not response:
        return False
    
    output = response.get("output", {})
    
    # Check for errors
    if "error" in output:
        print(f"✗ API returned error: {output['error']}")
        return False
    
    # Check for results
    if "results" not in output:
        print(f"✗ Missing 'results' in output")
        return False
    
    results = output["results"]
    
    if len(results) != expected_num_prompts:
        print(f"✗ Expected {expected_num_prompts} results, got {len(results)}")
        return False
    
    # Validate each result
    for idx, result in enumerate(results):
        required_fields = ["prompt", "predicted_class", "confidence", "probabilities"]
        
        for field in required_fields:
            if field not in result:
                print(f"✗ Missing '{field}' in result {idx}")
                return False
        
        # Validate probability distribution
        probs = result["probabilities"]
        if not isinstance(probs, list):
            print(f"✗ Probabilities should be a list")
            return False
        
        prob_sum = sum(probs)
        if abs(prob_sum - 1.0) > 0.01:
            print(f"✗ Probabilities don't sum to 1.0 (sum={prob_sum:.4f})")
            return False
    
    print(f"✓ Response validation passed")
    return True


def display_results(response: Dict[str, Any]):
    """Display formatted results."""
    output = response.get("output", {})
    
    print("\n" + "=" * 70)
    print("Classification Results")
    print("=" * 70)
    
    if "results" in output:
        for idx, result in enumerate(output["results"]):
            print(f"\n[Result {idx + 1}]")
            print(f"  Prompt: {result['prompt'][:60]}...")
            print(f"  Predicted Class: {result['predicted_class']}")
            print(f"  Confidence: {result['confidence']:.4f}")
            
            # Show top 3 predictions
            probs = result['probabilities']
            top_indices = sorted(
                range(len(probs)), 
                key=lambda i: probs[i], 
                reverse=True
            )[:3]
            
            print(f"  Top 3 Predictions:")
            for i, class_id in enumerate(top_indices):
                print(f"    {i+1}. Class {class_id}: {probs[class_id]:.4f}")
    
    if "metadata" in output:
        meta = output["metadata"]
        print(f"\n{'=' * 70}")
        print(f"Metadata:")
        print(f"  Total time: {meta.get('total_time')}s")
        print(f"  Avg time/prompt: {meta.get('avg_time_per_prompt')}s")
        print(f"  Model: {meta.get('model_name')}")
        print(f"  Quantization: {meta.get('quantization')}")
        print(f"  Max sequences: {meta.get('max_num_seqs')}")
    
    print(f"{'=' * 70}")


def run_latency_test(endpoint_id: str, 
                    api_key: str, 
                    num_iterations: int = 10,
                    batch_size: int = 1) -> Dict[str, float]:
    """
    Run multiple iterations to measure latency statistics.
    
    Args:
        endpoint_id: RunPod endpoint ID
        api_key: RunPod API key
        num_iterations: Number of test iterations
        batch_size: Number of prompts per request
        
    Returns:
        Dictionary with latency statistics
    """
    print("\n" + "=" * 70)
    print(f"Latency Test (batch_size={batch_size}, iterations={num_iterations})")
    print("=" * 70)
    
    # Test prompts
    test_prompts = [
        "Book a flight",
        "Order pizza",
        "Play music",
        "Set alarm",
        "Check weather",
        "Send email",
        "Schedule meeting",
        "Find restaurant",
    ]
    
    # Select prompts for batch
    prompts = test_prompts[:batch_size]
    
    latencies = []
    
    for i in range(num_iterations):
        response = test_endpoint(endpoint_id, api_key, prompts, timeout=120)
        
        if response and response.get("elapsed_time"):
            latency_ms = response["elapsed_time"] * 1000
            latencies.append(latency_ms)
            print(f"  Iteration {i+1}: {latency_ms:.2f}ms")
        else:
            print(f"  Iteration {i+1}: Failed")
    
    if not latencies:
        print("✗ All requests failed")
        return {}
    
    # Calculate statistics
    import numpy as np
    
    stats = {
        "batch_size": batch_size,
        "num_iterations": num_iterations,
        "avg_latency_ms": round(np.mean(latencies), 2),
        "min_latency_ms": round(np.min(latencies), 2),
        "max_latency_ms": round(np.max(latencies), 2),
        "p50_latency_ms": round(np.percentile(latencies, 50), 2),
        "p95_latency_ms": round(np.percentile(latencies, 95), 2),
        "p99_latency_ms": round(np.percentile(latencies, 99), 2),
        "std_latency_ms": round(np.std(latencies), 2),
        "throughput": round(batch_size / (np.mean(latencies) / 1000), 2),
    }
    
    print(f"\n  Latency Statistics:")
    print(f"    Average: {stats['avg_latency_ms']:.2f}ms")
    print(f"    P50: {stats['p50_latency_ms']:.2f}ms")
    print(f"    P95: {stats['p95_latency_ms']:.2f}ms")
    print(f"    P99: {stats['p99_latency_ms']:.2f}ms")
    print(f"    Throughput: {stats['throughput']:.2f} samples/s")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Test deployed RunPod endpoint"
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
        "--prompts",
        nargs="+",
        default=None,
        help="Custom prompts to classify (space-separated)"
    )
    parser.add_argument(
        "--latency-test",
        action="store_true",
        help="Run latency benchmarking"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of iterations for latency test"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for latency test"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (JSON)"
    )
    
    args = parser.parse_args()
    
    # Use custom prompts or defaults
    if args.prompts:
        prompts = args.prompts
    else:
        prompts = [
            "Book me a flight to San Francisco next Tuesday",
            "Add milk and bread to my shopping list",
            "Play some rock music",
        ]
    
    # Test endpoint
    print("=" * 70)
    print("Testing RunPod Endpoint")
    print("=" * 70)
    
    response = test_endpoint(args.endpoint_id, args.api_key, prompts)
    
    if not response:
        print("\n✗ Endpoint test failed")
        sys.exit(1)
    
    # Validate response
    valid = validate_response(response, len(prompts))
    
    if not valid:
        print("\n✗ Response validation failed")
        sys.exit(1)
    
    # Display results
    display_results(response)
    
    # Run latency test if requested
    if args.latency_test:
        stats = run_latency_test(
            args.endpoint_id, 
            args.api_key, 
            args.iterations,
            args.batch_size
        )
        
        # Save results if output path specified
        if args.output and stats:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(stats, f, indent=2)
            print(f"\n✓ Latency stats saved to: {args.output}")
    
    print("\n✓ All endpoint tests passed!")


if __name__ == "__main__":
    main()

