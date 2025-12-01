#!/usr/bin/env python3
"""
Download and verify the intent classification model from Hugging Face.
This script downloads the model, checks its configuration, and saves metadata.
"""

import os
import sys
import json
import argparse
from pathlib import Path

from transformers import AutoTokenizer, AutoConfig
from huggingface_hub import snapshot_download


def download_model(model_id: str, output_dir: str = "./models") -> dict:
    """
    Download model and tokenizer from Hugging Face.
    
    Args:
        model_id: Hugging Face model ID
        output_dir: Directory to save model info
        
    Returns:
        Dictionary with model information
    """
    print("=" * 70)
    print(f"Downloading Model: {model_id}")
    print("=" * 70)
    
    try:
        # Create output directory first
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Download tokenizer
        print("\n[1/4] Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True
        )
        
        # Fix pad token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("  ✓ Set pad_token to eos_token")
        
        print(f"  ✓ Tokenizer downloaded successfully")
        print(f"  - Vocab size: {tokenizer.vocab_size}")
        print(f"  - Pad token: {tokenizer.pad_token}")
        
        # Save tokenizer to disk
        print("\n[2/4] Saving tokenizer to disk...")
        tokenizer.save_pretrained(output_dir)
        print(f"  ✓ Tokenizer saved to: {output_dir}")
        
        # Download model files (without loading into memory)
        print("\n[3/4] Downloading model files...")
        snapshot_download(
            repo_id=model_id,
            local_dir=output_dir,
            local_dir_use_symlinks=False,
            trust_remote_code=True
        )
        print(f"  ✓ Model files downloaded to: {output_dir}")
        
        # Load config only (lightweight)
        config = AutoConfig.from_pretrained(output_dir, trust_remote_code=True)
        
        # Set pad token id in config if needed
        if config.pad_token_id is None:
            config.pad_token_id = tokenizer.pad_token_id
            config.save_pretrained(output_dir)
        
        print(f"  ✓ Model configuration loaded")
        print(f"  - Model type: {config.model_type}")
        print(f"  - Number of labels: {config.num_labels}")
        
        # Calculate model size from files
        model_size_bytes = sum(
            f.stat().st_size for f in Path(output_dir).rglob("*.safetensors")
        )
        if model_size_bytes == 0:
            model_size_bytes = sum(
                f.stat().st_size for f in Path(output_dir).rglob("*.bin")
            )
        model_size_mb = model_size_bytes / (1024 * 1024)
        
        # Extract label mappings
        id2label = config.id2label
        label2id = config.label2id
        
        print(f"\n[4/4] Verifying configuration...")
        print(f"  ✓ Model size: {model_size_mb:.2f} MB")
        print(f"  ✓ Number of classes: {len(id2label)}")
        
        # Display first few labels
        print("\n  Sample labels:")
        for idx in sorted(id2label.keys())[:5]:
            print(f"    {idx}: {id2label[idx]}")
        print(f"    ... ({len(id2label) - 5} more)")
        
        # Save model info
        model_info = {
            "model_id": model_id,
            "model_type": config.model_type,
            "model_size_mb": round(model_size_mb, 2),
            "num_labels": config.num_labels,
            "id2label": id2label,
            "label2id": label2id,
            "vocab_size": tokenizer.vocab_size,
            "max_position_embeddings": getattr(config, "max_position_embeddings", None),
            "hidden_size": getattr(config, "hidden_size", None),
        }
        
        output_path = os.path.join(output_dir, "model_info.json")
        with open(output_path, "w") as f:
            json.dump(model_info, f, indent=2)
        
        print(f"\n  ✓ Model info saved to: {output_path}")
        
        # Skip inference test to save memory
        print(f"\n[Info] Skipping inference test to conserve memory")
        print(f"  Model files downloaded successfully and ready to use")
        print(f"  Test inference will be performed when loading the model for benchmarks")
        
        print("\n" + "=" * 70)
        print("✓ Model download and verification complete!")
        print("=" * 70)
        
        return model_info
        
    except Exception as e:
        print(f"\n✗ Error downloading model: {str(e)}")
        print(f"\nDebug info:")
        print(f"  Model ID: {model_id}")
        print(f"  Output directory: {output_dir}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Download and verify intent classification model"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="codefactory4791/intent-classification-qwen",
        help="Hugging Face model ID"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models",
        help="Directory to save model info"
    )
    
    args = parser.parse_args()
    
    # Download model
    model_info = download_model(args.model_id, args.output_dir)
    
    print(f"\nNext steps:")
    print(f"  1. Run benchmarks: python scripts/benchmark_local.py")
    print(f"  2. Test handler: python scripts/test_local_handler.py")
    print(f"  3. Build Docker: docker build -t intent-classification-vllm .")


if __name__ == "__main__":
    main()

