#!/usr/bin/env python3
"""
Test the RunPod handler locally before deployment.
This simulates the RunPod environment and validates the handler logic.
"""

import sys
import os
import json
from pathlib import Path

# Add app directory to path to import handler
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.handler import handler


def test_single_prompt():
    """Test classification with a single prompt."""
    print("\n" + "=" * 70)
    print("Test 1: Single Prompt")
    print("=" * 70)
    
    event = {
        "input": {
            "prompts": "Book me a flight to San Francisco next Tuesday"
        }
    }
    
    print(f"Input: {json.dumps(event, indent=2)}")
    
    response = handler(event)
    
    print(f"\nResponse:")
    print(json.dumps(response, indent=2))
    
    # Validate response
    assert "results" in response, "Missing 'results' in response"
    assert len(response["results"]) == 1, "Expected 1 result"
    assert "predicted_class" in response["results"][0], "Missing 'predicted_class'"
    assert "confidence" in response["results"][0], "Missing 'confidence'"
    assert "probabilities" in response["results"][0], "Missing 'probabilities'"
    
    print("\n✓ Single prompt test passed")
    return response


def test_batch_prompts():
    """Test classification with multiple prompts."""
    print("\n" + "=" * 70)
    print("Test 2: Batch Prompts")
    print("=" * 70)
    
    prompts = [
        "Book me a flight to San Francisco",
        "Add milk and bread to my shopping list",
        "Play some rock music",
        "Set an alarm for 7 AM tomorrow",
        "What's the weather like today?",
    ]
    
    event = {
        "input": {
            "prompts": prompts
        }
    }
    
    print(f"Input: {len(prompts)} prompts")
    
    response = handler(event)
    
    print(f"\nResponse:")
    print(f"  Status: {response.get('status')}")
    print(f"  Number of results: {len(response.get('results', []))}")
    
    # Show first result in detail
    if response.get('results'):
        print(f"\n  First result:")
        first = response['results'][0]
        print(f"    Prompt: {first['prompt']}")
        print(f"    Predicted class: {first['predicted_class']}")
        print(f"    Confidence: {first['confidence']:.4f}")
        print(f"    Top 3 classes:")
        
        probs = first['probabilities']
        top_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:3]
        for idx in top_indices:
            print(f"      Class {idx}: {probs[idx]:.4f}")
    
    # Show metadata
    if 'metadata' in response:
        meta = response['metadata']
        print(f"\n  Metadata:")
        print(f"    Total time: {meta['total_time']}s")
        print(f"    Avg time per prompt: {meta['avg_time_per_prompt']}s")
        print(f"    Model: {meta['model_name']}")
        print(f"    Quantization: {meta['quantization']}")
    
    # Validate response
    assert response.get("status") == "success", f"Expected success, got {response.get('status')}"
    assert len(response["results"]) == len(prompts), f"Expected {len(prompts)} results"
    
    print("\n✓ Batch prompts test passed")
    return response


def test_error_handling():
    """Test error handling with invalid inputs."""
    print("\n" + "=" * 70)
    print("Test 3: Error Handling")
    print("=" * 70)
    
    # Test 1: Missing input
    print("\n  3a. Missing 'input' field")
    event = {}
    response = handler(event)
    assert "error" in response, "Expected error for missing input"
    print(f"    ✓ Correctly returned error: {response['error']}")
    
    # Test 2: Missing prompts
    print("\n  3b. Missing 'prompts' field")
    event = {"input": {}}
    response = handler(event)
    assert "error" in response, "Expected error for missing prompts"
    print(f"    ✓ Correctly returned error: {response['error']}")
    
    # Test 3: Invalid prompt type
    print("\n  3c. Invalid prompt type")
    event = {"input": {"prompts": 12345}}
    response = handler(event)
    assert "error" in response, "Expected error for invalid type"
    print(f"    ✓ Correctly returned error: {response['error']}")
    
    print("\n✓ Error handling tests passed")


def test_performance_metrics():
    """Test and report performance metrics."""
    print("\n" + "=" * 70)
    print("Test 4: Performance Metrics")
    print("=" * 70)
    
    # Use a moderate batch
    prompts = [
        "Book a flight",
        "Order pizza",
        "Play music",
        "Set alarm",
        "Check weather",
        "Send email",
        "Schedule meeting",
        "Find restaurant",
    ]
    
    event = {"input": {"prompts": prompts}}
    
    # Run multiple iterations
    iterations = 5
    times = []
    
    print(f"\nRunning {iterations} iterations with {len(prompts)} prompts each...")
    
    for i in range(iterations):
        start = time.perf_counter()
        response = handler(event)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        
        assert response.get("status") == "success"
        print(f"  Iteration {i+1}: {elapsed:.3f}s ({len(prompts)/elapsed:.2f} samples/s)")
    
    # Calculate statistics
    avg_time = np.mean(times)
    min_time = np.min(times)
    max_time = np.max(times)
    std_time = np.std(times)
    
    print(f"\n  Performance Summary:")
    print(f"    Average time: {avg_time:.3f}s")
    print(f"    Min time: {min_time:.3f}s")
    print(f"    Max time: {max_time:.3f}s")
    print(f"    Std dev: {std_time:.3f}s")
    print(f"    Throughput: {len(prompts)/avg_time:.2f} samples/s")
    
    print("\n✓ Performance metrics test passed")


def main():
    """Run all tests."""
    print("=" * 70)
    print("Testing RunPod Handler Locally")
    print("=" * 70)
    print("\nThis will test the handler logic before deployment to RunPod.")
    print("Ensure you have a GPU available for vLLM initialization.")
    
    try:
        # Run test suite
        test_single_prompt()
        test_batch_prompts()
        test_error_handling()
        test_performance_metrics()
        
        print("\n" + "=" * 70)
        print("✓ ALL TESTS PASSED")
        print("=" * 70)
        print("\nHandler is ready for deployment!")
        print("\nNext steps:")
        print("  1. Build Docker image: docker build -t intent-classification-vllm .")
        print("  2. Deploy to RunPod: See README deployment section")
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

