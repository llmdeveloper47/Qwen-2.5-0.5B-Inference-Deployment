#!/usr/bin/env python3
"""
Check completeness of experiment results.
Verifies that all expected result files exist and are valid.
"""

import json
import sys
from pathlib import Path
from typing import List, Tuple


def check_experiment_completeness(
    results_dir: str = "./results/experiments",
    quantization_methods: List[str] = None,
    batch_sizes: List[int] = None
) -> Tuple[int, int, List[str]]:
    """
    Check if all expected result files exist.
    
    Returns:
        (found_count, expected_count, missing_files)
    """
    if quantization_methods is None:
        quantization_methods = ["none", "bitsandbytes", "awq", "gptq"]
    
    if batch_sizes is None:
        batch_sizes = [1, 4, 8, 16, 32]
    
    results_path = Path(results_dir)
    
    expected_files = []
    found_files = []
    missing_files = []
    
    print("=" * 70)
    print("Checking Experiment Completeness")
    print("=" * 70)
    print(f"\nResults directory: {results_dir}")
    print(f"Expected quantization methods: {quantization_methods}")
    print(f"Expected batch sizes: {batch_sizes}")
    print()
    
    for method in quantization_methods:
        print(f"\n[{method}]")
        method_dir = results_path / method
        
        if not method_dir.exists():
            print(f"  ⚠ Directory not found: {method_dir}")
            for bs in batch_sizes:
                file_path = method_dir / f"batch_{bs}.json"
                expected_files.append(str(file_path))
                missing_files.append(str(file_path))
                print(f"    ✗ batch_{bs}.json")
            continue
        
        for bs in batch_sizes:
            file_path = method_dir / f"batch_{bs}.json"
            expected_files.append(str(file_path))
            
            if file_path.exists():
                # Validate JSON
                try:
                    with open(file_path) as f:
                        data = json.load(f)
                    
                    # Check required fields
                    required = ['batch_size', 'quantization', 'p95_latency_ms', 'throughput']
                    missing_fields = [f for f in required if f not in data]
                    
                    if missing_fields:
                        print(f"    ⚠ batch_{bs}.json - Missing fields: {missing_fields}")
                    else:
                        found_files.append(str(file_path))
                        print(f"    ✓ batch_{bs}.json")
                        
                except json.JSONDecodeError:
                    print(f"    ✗ batch_{bs}.json - Invalid JSON")
                    missing_files.append(str(file_path))
            else:
                print(f"    ✗ batch_{bs}.json - Not found")
                missing_files.append(str(file_path))
    
    return len(found_files), len(expected_files), missing_files


def check_file_validity(file_path: Path) -> Tuple[bool, str]:
    """Check if a result file is valid."""
    if not file_path.exists():
        return False, "File not found"
    
    try:
        with open(file_path) as f:
            data = json.load(f)
        
        # Check required fields
        required_fields = [
            'batch_size', 'quantization', 'successful', 'failed',
            'avg_latency_ms', 'p95_latency_ms', 'throughput'
        ]
        
        for field in required_fields:
            if field not in data:
                return False, f"Missing required field: {field}"
        
        # Sanity checks
        if data['successful'] == 0:
            return False, "No successful requests"
        
        if data['p95_latency_ms'] <= 0:
            return False, "Invalid latency value"
        
        if data['throughput'] <= 0:
            return False, "Invalid throughput value"
        
        return True, "Valid"
        
    except json.JSONDecodeError:
        return False, "Invalid JSON"
    except Exception as e:
        return False, f"Error: {str(e)}"


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Check completeness of experiment results"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./results/experiments",
        help="Results directory to check"
    )
    parser.add_argument(
        "--quantization-methods",
        nargs="+",
        default=["none", "bitsandbytes", "awq", "gptq"],
        help="Quantization methods to check"
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,4,8,16,32",
        help="Comma-separated batch sizes to check"
    )
    
    args = parser.parse_args()
    
    # Parse batch sizes
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]
    
    # Check completeness
    found, expected, missing = check_experiment_completeness(
        args.results_dir,
        args.quantization_methods,
        batch_sizes
    )
    
    # Print summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  Found: {found}/{expected} result files")
    print(f"  Missing: {len(missing)} files")
    print(f"  Completion: {100 * found / expected:.1f}%")
    
    if missing:
        print(f"\n  Missing files:")
        for file in missing[:10]:  # Show first 10
            print(f"    - {file}")
        
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more")
    
    # Detailed validation
    if found > 0:
        print("\n" + "=" * 70)
        print("Validating Found Files")
        print("=" * 70)
        
        invalid = []
        
        for file in Path(args.results_dir).rglob("batch_*.json"):
            valid, message = check_file_validity(file)
            
            if not valid:
                invalid.append((str(file), message))
                print(f"  ✗ {file.relative_to(args.results_dir)}: {message}")
        
        if not invalid:
            print("  ✓ All found files are valid")
        else:
            print(f"\n  ⚠ Found {len(invalid)} invalid files")
    
    # Exit code
    if found == expected:
        print("\n✓ All experiments complete!")
        sys.exit(0)
    else:
        print(f"\n⚠ Experiments incomplete ({found}/{expected})")
        print(f"\nTo complete experiments, run missing configurations:")
        print(f"  python scripts/test_endpoint.py --latency-test --batch-size BS")
        sys.exit(1)


if __name__ == "__main__":
    main()

