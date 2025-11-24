#!/usr/bin/env python3
"""
Initialize directory structure for experiments and results.
Creates all necessary folders with README files.
"""

from pathlib import Path


def create_directory_structure():
    """Create the complete directory structure for the project."""
    directories = {
        "models": "Downloaded models and model information",
        "results/local_benchmarks/none": "FP16 local benchmark results",
        "results/local_benchmarks/bitsandbytes": "INT8 local benchmark results",
        "results/local_benchmarks/awq": "AWQ local benchmark results",
        "results/local_benchmarks/gptq": "GPTQ local benchmark results",
        "results/experiments/none": "FP16 endpoint experiment results",
        "results/experiments/bitsandbytes": "INT8 endpoint experiment results",
        "results/experiments/awq": "AWQ endpoint experiment results",
        "results/experiments/gptq": "GPTQ endpoint experiment results",
        "results/load_tests/none": "FP16 load test results",
        "results/load_tests/bitsandbytes": "INT8 load test results",
        "results/load_tests/awq": "AWQ load test results",
        "results/load_tests/gptq": "GPTQ load test results",
        "results/analysis": "Analysis outputs and visualizations",
        "experiments/results": "Experiment results (symlink to results/experiments)",
        "experiments/analysis": "Analysis notebooks and outputs",
    }
    
    print("=" * 70)
    print("Initializing Directory Structure")
    print("=" * 70)
    print()
    
    created_count = 0
    
    for dir_path, description in directories.items():
        path = Path(dir_path)
        
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            
            # Create README in each directory
            readme_path = path / ".gitkeep"
            readme_path.touch()
            
            print(f"✓ Created: {dir_path}")
            created_count += 1
        else:
            print(f"  Exists: {dir_path}")
    
    print()
    print("=" * 70)
    print(f"✓ Directory structure initialized")
    print(f"  Created {created_count} new directories")
    print("=" * 70)
    print()
    print("Directory structure:")
    print()
    print("results/")
    print("├── local_benchmarks/")
    print("│   ├── none/")
    print("│   ├── bitsandbytes/")
    print("│   ├── awq/")
    print("│   └── gptq/")
    print("├── experiments/")
    print("│   ├── none/")
    print("│   ├── bitsandbytes/")
    print("│   ├── awq/")
    print("│   └── gptq/")
    print("├── load_tests/")
    print("│   └── [method]/[batch_size]/")
    print("└── analysis/")
    print()


if __name__ == "__main__":
    create_directory_structure()

