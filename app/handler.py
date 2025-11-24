"""
RunPod Handler for Intent Classification using vLLM
This handler wraps the vLLM classification model for RunPod serverless deployment.
"""

import os
import time
import logging
from typing import Any, Dict, List, Optional, Union

import runpod
from vllm import LLM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Read configuration from environment variables
MODEL_NAME = os.getenv("MODEL_NAME", "codefactory4791/intent-classification-qwen")
MAX_NUM_SEQS = int(os.getenv("MAX_NUM_SEQS", "16"))
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "512"))
QUANTIZATION = os.getenv("QUANTIZATION", "none")
DTYPE = os.getenv("DTYPE", "auto")
GPU_MEMORY_UTILIZATION = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.95"))
TRUST_REMOTE_CODE = os.getenv("TRUST_REMOTE_CODE", "true").lower() == "true"
ENFORCE_EAGER = os.getenv("ENFORCE_EAGER", "true").lower() == "true"

logger.info(f"Starting vLLM handler with configuration:")
logger.info(f"  MODEL_NAME: {MODEL_NAME}")
logger.info(f"  MAX_NUM_SEQS: {MAX_NUM_SEQS}")
logger.info(f"  MAX_MODEL_LEN: {MAX_MODEL_LEN}")
logger.info(f"  QUANTIZATION: {QUANTIZATION}")
logger.info(f"  DTYPE: {DTYPE}")
logger.info(f"  GPU_MEMORY_UTILIZATION: {GPU_MEMORY_UTILIZATION}")
logger.info(f"  TRUST_REMOTE_CODE: {TRUST_REMOTE_CODE}")
logger.info(f"  ENFORCE_EAGER: {ENFORCE_EAGER}")

# Initialize vLLM model once at startup
# Use task="classify" for classification models
logger.info("Initializing vLLM model...")
start_time = time.time()

try:
    # Handle quantization parameter
    quantization_param = None if QUANTIZATION == "none" else QUANTIZATION
    
    LLM_INSTANCE = LLM(
        model=MODEL_NAME,
        task="classify",
        max_num_seqs=MAX_NUM_SEQS,
        max_model_len=MAX_MODEL_LEN,
        quantization=quantization_param,
        dtype=DTYPE,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        trust_remote_code=TRUST_REMOTE_CODE,
        enforce_eager=ENFORCE_EAGER,
    )
    
    load_time = time.time() - start_time
    logger.info(f"✓ Model loaded successfully in {load_time:.2f}s")
    
except Exception as e:
    logger.error(f"✗ Failed to load model: {str(e)}")
    raise


def classify_batch(prompts: List[str]) -> List[Dict[str, Any]]:
    """
    Run classification on a batch of prompts.
    
    Args:
        prompts: List of text prompts to classify
        
    Returns:
        List of dictionaries containing classification results
    """
    try:
        start_time = time.time()
        
        # Run classification
        outputs = LLM_INSTANCE.classify(prompts)
        
        inference_time = time.time() - start_time
        
        # Process results
        results = []
        for idx, output in enumerate(outputs):
            # Extract probabilities and predicted class
            probs = output.outputs.probs.tolist()
            predicted_class = probs.index(max(probs))
            
            result = {
                "prompt": prompts[idx],
                "predicted_class": predicted_class,
                "confidence": max(probs),
                "probabilities": probs,
            }
            results.append(result)
        
        logger.info(
            f"Classified {len(prompts)} prompts in {inference_time:.3f}s "
            f"({len(prompts)/inference_time:.2f} samples/s)"
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Classification error: {str(e)}")
        raise


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod handler function.
    
    Expected input format:
    {
        "input": {
            "prompts": ["text1", "text2", ...] or "single text"
        }
    }
    
    Returns:
    {
        "results": [
            {
                "prompt": "text",
                "predicted_class": 0,
                "confidence": 0.95,
                "probabilities": [...]
            },
            ...
        ],
        "metadata": {
            "num_prompts": 2,
            "inference_time": 0.123,
            "model_name": "...",
            "quantization": "..."
        }
    }
    """
    start_time = time.time()
    
    try:
        # Validate input
        if not event or 'input' not in event:
            return {
                "error": "Missing 'input' field in request",
                "status": "error"
            }
        
        data = event['input']
        prompts = data.get('prompts')
        
        if not prompts:
            return {
                "error": "No 'prompts' provided in input",
                "status": "error"
            }
        
        # Normalize prompts to list
        if isinstance(prompts, str):
            prompts = [prompts]
        elif not isinstance(prompts, list):
            return {
                "error": "Invalid prompt format. Expected string or list of strings.",
                "status": "error"
            }
        
        # Validate prompt types
        if not all(isinstance(p, str) for p in prompts):
            return {
                "error": "All prompts must be strings",
                "status": "error"
            }
        
        # Run classification
        results = classify_batch(prompts)
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Return successful response
        return {
            "results": results,
            "metadata": {
                "num_prompts": len(prompts),
                "total_time": round(total_time, 4),
                "avg_time_per_prompt": round(total_time / len(prompts), 4),
                "model_name": MODEL_NAME,
                "quantization": QUANTIZATION,
                "max_num_seqs": MAX_NUM_SEQS,
            },
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Handler error: {str(e)}", exc_info=True)
        return {
            "error": str(e),
            "status": "error"
        }


# Start the RunPod serverless worker
if __name__ == "__main__":
    logger.info("Starting RunPod serverless worker...")
    runpod.serverless.start({"handler": handler})

