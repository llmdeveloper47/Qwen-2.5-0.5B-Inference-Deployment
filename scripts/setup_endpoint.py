#!/usr/bin/env python3
"""
Helper script to display RunPod endpoint configuration.
Since RunPod API for updating environment variables requires GraphQL,
this script generates the configuration that you can copy-paste into RunPod console.
"""

import argparse
import json
from pathlib import Path


def load_config(config_file: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_file) as f:
        return json.load(f)


def display_endpoint_config(config: dict):
    """Display endpoint configuration in a formatted way."""
    print("\n" + "=" * 70)
    print(f"Configuration: {config['name']}")
    print("=" * 70)
    print(f"\nDescription: {config['description']}")
    
    print("\n[Environment Variables]")
    print("-" * 70)
    
    for key, value in config['environment'].items():
        print(f"  {key:<30} = {value}")
    
    print("\n[GPU Configuration]")
    print("-" * 70)
    print(f"  GPU Type: {config['gpu']['type']}")
    print(f"  Min Workers: {config['gpu']['min_workers']}")
    print(f"  Max Workers: {config['gpu']['max_workers']}")
    
    if 'expected_performance' in config:
        print("\n[Expected Performance]")
        print("-" * 70)
        for key, value in config['expected_performance'].items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
    
    if 'use_case' in config:
        print(f"\n[Use Case]")
        print("-" * 70)
        print(f"  {config['use_case']}")
    
    if 'notes' in config:
        print(f"\n[Notes]")
        print("-" * 70)
        for note in config['notes']:
            print(f"  - {note}")
    
    print("\n" + "=" * 70)


def generate_env_vars_script(config: dict, output_file: str = None):
    """Generate shell script to export environment variables."""
    lines = ["#!/bin/bash", "", "# RunPod endpoint environment variables"]
    lines.append(f"# Configuration: {config['name']}")
    lines.append("")
    
    for key, value in config['environment'].items():
        lines.append(f'export {key}="{value}"')
    
    script = "\n".join(lines)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(script)
        print(f"\n✓ Environment script saved to: {output_file}")
    
    return script


def main():
    parser = argparse.ArgumentParser(
        description="Display and generate RunPod endpoint configuration"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        choices=["fp16", "bitsandbytes", "awq", "gptq"],
        help="Configuration to use"
    )
    parser.add_argument(
        "--export-script",
        type=str,
        default=None,
        help="Export environment variables to shell script"
    )
    
    args = parser.parse_args()
    
    # Map config names to files
    config_files = {
        "fp16": "configs/fp16_baseline.json",
        "bitsandbytes": "configs/bitsandbytes_int8.json",
        "awq": "configs/awq_4bit.json",
        "gptq": "configs/gptq_4bit.json",
    }
    
    config_file = config_files[args.config]
    
    # Load configuration
    config = load_config(config_file)
    
    # Display configuration
    display_endpoint_config(config)
    
    # Generate environment variables script if requested
    if args.export_script:
        generate_env_vars_script(config, args.export_script)
    
    print("\n[Setup Instructions]")
    print("-" * 70)
    print("1. Login to RunPod console: https://www.runpod.io/console/serverless")
    print("2. Navigate to your endpoint and click 'Manage' → 'Edit Endpoint'")
    print("3. Expand 'Public Environment Variables'")
    print("4. Copy-paste the environment variables shown above")
    print("5. Save the endpoint and wait for it to restart (~2-3 minutes)")
    print("6. Test the endpoint:")
    print(f"   python scripts/test_endpoint.py --endpoint-id YOUR_ID --api-key YOUR_KEY")
    print("-" * 70)


if __name__ == "__main__":
    main()

