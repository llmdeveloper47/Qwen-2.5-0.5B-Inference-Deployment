#!/usr/bin/env python3
"""
Memory-efficient model download script.
Downloads model files without loading into RAM.
"""

from pathlib import Path
from transformers import AutoTokenizer, AutoConfig
from huggingface_hub import snapshot_download

model_id = "codefactory4791/intent-classification-qwen"
output_dir = "./models"

print("=" * 70)
print(f"Downloading Model: {model_id}")
print("=" * 70)

# Create output directory
Path(output_dir).mkdir(parents=True, exist_ok=True)

# Download tokenizer
print("\n[1/3] Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.save_pretrained(output_dir)
print(f"  Tokenizer saved to: {output_dir}")

# Download model files (without loading)
print("\n[2/3] Downloading model files...")
snapshot_download(
    repo_id=model_id,
    local_dir=output_dir,
    local_dir_use_symlinks=False
)
print(f"  Model files downloaded to: {output_dir}")

# Load config only
print("\n[3/3] Verifying download...")
config = AutoConfig.from_pretrained(output_dir, trust_remote_code=True)
print(f"  Model type: {config.model_type}")
print(f"  Number of labels: {config.num_labels}")

print("\n" + "=" * 70)
print("Download complete!")
print("=" * 70)
