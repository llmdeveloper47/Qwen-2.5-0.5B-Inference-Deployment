# Deploying `codefactory4791/intent‑classification‑qwen` with vLLM on RunPod

This tutorial walks through deploying your fine‑tuned **intent classification** model on RunPod using the **vLLM** engine.  The goal is to start with a local evaluation of the model, optimise it for inference, and then build a production‑ready endpoint on RunPod that can be tested with **Locust**.  The workflow covers:

1. Preparing your environment and downloading the model.
2. Evaluating the model locally with `vllm.LLM` and optimising batch sizes.
3. Containerising a custom vLLM worker for classification.
4. Setting up CI/CD to build and push the Docker image and deploy it on RunPod.
5. Deploying the model on RunPod and testing it with a simple API request.
6. Load‑testing the endpoint using Locust to measure throughput under different batch sizes.

> **Note:** The model `codefactory4791/intent‑classification‑qwen` is a sequence‑classification model (not a generative chat model).  vLLM supports classification by passing `task="classify"` when you construct an `LLM` instance.  The classification API returns logits and probabilities for each class; see the vLLM documentation for details【21300289411745†L214-L221】.

## 1 Prerequisites

Ensure the following prerequisites before starting:

* **Accounts:** You need accounts on **Hugging Face** and **RunPod**. For private models, create a Hugging Face access token.  RunPod requires an API key and funds.
* **Local hardware:** A GPU‑equipped machine (CUDA ≥ 12.1) to benchmark and quantise the model locally.  CPU inference will work but will be slower.
* **Software:** Install Python 3.10+, Git, Docker, and the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/quick-start#install). Locust will be used later for load testing.
* **Repository:** Create a GitHub repository (e.g. `intent‑classification‑runpod`) that will hold your code and CI/CD workflows.

## 2 Download and Merge the Model Locally

The notebook you supplied fine‑tunes **Qwen 2.5‑0.5B** with a LoRA adapter and pushes the merged model to the Hugging Face Hub.  Below is a standalone script that downloads your published model `codefactory4791/intent‑classification‑qwen`, verifies the label mapping and merges LoRA weights if necessary.  It uses `transformers` and `peft` just like your notebook.

```python
#!/usr/bin/env python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

# Replace with your repo ID and LoRA path if merging locally
MODEL_ID = "codefactory4791/intent‑classification‑qwen"
LORA_CHECKPOINT = "./intent_classification/checkpoint-10064"  # optional

# Download model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)

# If pad token is missing, set it to EOS
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

# Optional: load a LoRA adapter and merge weights
if LORA_CHECKPOINT:
    base_model = model
    model = PeftModel.from_pretrained(base_model, LORA_CHECKPOINT)
    model = model.merge_and_unload()

model.eval()
print("Model and tokenizer loaded")

# Save the merged model back to Hugging Face if needed
model.push_to_hub(MODEL_ID, private=True)
tokenizer.push_to_hub(MODEL_ID, private=True)
```

Running this script on a GPU machine will download your fine‑tuned model and optionally merge LoRA weights.  When you call `push_to_hub`, the model is stored under `codefactory4791/intent‑classification‑qwen`.  This step only needs to be done once.

## 3 Local Evaluation with vLLM

### 3.1 Install vLLM and dependencies

vLLM is a high‑performance inference engine for transformer models.  Install it in a fresh virtual environment:

```bash
pip install --upgrade pip
pip install "vllm>=0.7.0" torch transformers datasets
```

### 3.2 Prepare an evaluation dataset

For benchmarking, load your test split from the `amazon_test` dataset used in the notebook and map text to label IDs.  Here’s a helper function to load the dataset and build the label mappings:

```python
from datasets import load_dataset

def load_eval_dataset(dataset_name: str, split: str = "test"):
    dataset = load_dataset(dataset_name)
    df = dataset[split].to_pandas()
    # rename columns if needed
    if 'query' in df.columns and 'label' in df.columns:
        df = df.rename(columns={'query': 'text', 'label': 'labels'})
    labels = sorted(df['labels'].unique())
    label2id = {lbl: idx for idx, lbl in enumerate(labels)}
    id2label = {idx: lbl for idx, lbl in enumerate(labels)}
    df['label_id'] = df['labels'].map(label2id)
    return df, labels, label2id, id2label

eval_df, labels, label2id, id2label = load_eval_dataset("codefactory4791/amazon_test", "test")
print(len(eval_df), "samples", len(labels), "labels")
```

### 3.3 Classify with vLLM

vLLM supports classification by creating an `LLM` with `task="classify"`【21300289411745†L214-L221】.  The classifier returns logits and probabilities for each class.  The following script benchmarks different batch sizes and measures throughput (samples/s) and latency.  It uses the `classify` method of `LLM` and is suitable for GPU inference:

```python
import time
from vllm import LLM

# The model ID published on Hugging Face
MODEL_ID = "codefactory4791/intent‑classification‑qwen"

# Create an LLM for classification
llm = LLM(model=MODEL_ID, task="classify", enforce_eager=True)

def benchmark(batch_size: int, prompts: list[str]):
    """Run classification for a batch and measure throughput."""
    batches = [prompts[i:i+batch_size] for i in range(0, len(prompts), batch_size)]
    start = time.perf_counter()
    total = 0
    for batch in batches:
        outputs = llm.classify(batch)
        # outputs is a list of ClassificationRequestOutputs; we only count time
        total += len(batch)
    elapsed = time.perf_counter() - start
    return total / elapsed  # samples per second

# Evaluate batch sizes 1, 4, 8, 16, 32
prompts = eval_df['text'].tolist()[:512]  # use a subset for speed
for bs in [1, 4, 8, 16, 32]:
    tput = benchmark(bs, prompts)
    print(f"batch_size={bs:>2} -> {tput:.2f} samples/s")
```

**Interpreting results:**  Smaller batch sizes may produce lower GPU utilisation, while very large batches could exceed memory.  Use these numbers to choose a default `MAX_NUM_SEQS` for your serverless deployment; for example, if 16 gives the best throughput without GPU OOM, set `MAX_NUM_SEQS=16` as an environment variable later.

### 3.4 Quantisation (optional)

vLLM supports quantisation methods like AWQ, GPTQ, and bits‑and‑bytes to reduce memory.  When you deploy on RunPod you can set the `QUANTIZATION` environment variable to apply a method such as `awq` or `bitsandbytes`【562642474504598†L59-L76】.  Quantisation can reduce VRAM requirements but may slightly degrade accuracy; evaluate locally before deploying.  For example:

```bash
# Example: quantise the model with AWQ when serving
export QUANTIZATION=awq
```

## 4 Create a Custom vLLM Worker for Classification

The official vLLM RunPod worker (`runpod/worker-v1-vllm`) assumes generative chat models.  To serve a classification model, fork the repository and modify the handler to use `task="classify"`.  This section describes the directory structure and files you need.

### 4.1 Project layout

In your GitHub repository, arrange files as follows:

```
intent-classification-runpod/
├── .github/
│   └── workflows/
│       └── deploy.yml      # CI/CD pipeline
├── .runpod/
│   └── hub.json            # Endpoint configuration for RunPod Hub
├── app/
│   ├── handler.py          # RunPod handler wrapping vLLM classification
│   └── requirements.txt    # Python dependencies
├── Dockerfile              # Build the custom worker image
└── README.md               # Documentation for your repository
```

### 4.2 Handler implementation (`app/handler.py`)

A RunPod serverless worker must expose a `handler` function taking an input dictionary and returning an output dictionary.  The code below loads the classification model using vLLM on startup and handles both synchronous and streaming requests.  It reads environment variables for the model name, number of sequences per batch, and quantisation method.  Save this file as `app/handler.py`.

```python
import os
from typing import Any, Dict, List
from vllm import LLM

# Read settings from environment variables
MODEL_NAME = os.getenv("MODEL_NAME", "codefactory4791/intent‑classification‑qwen")
MAX_NUM_SEQS = int(os.getenv("MAX_NUM_SEQS", "16"))
QUANTIZATION = os.getenv("QUANTIZATION")

# Create the vLLM model once during start‑up.  Use task="classify".
LLM_INSTANCE = LLM(model=MODEL_NAME, task="classify", enforce_eager=True,
                   max_num_seqs=MAX_NUM_SEQS, quantization=QUANTIZATION)

def classify_batch(prompts: List[str]):
    """Runs classification on a list of prompts and returns probabilities."""
    outputs = LLM_INSTANCE.classify(prompts)
    result = []
    for output in outputs:
        probs = output.outputs.probs  # probability per class
        result.append(probs.tolist())
    return result

def handler(event: Dict[str, Any]):
    """RunPod entrypoint.  Expects {'input': {'prompts': [...]} }"""
    if not event or 'input' not in event:
        return {"error": "Missing input"}

    data = event['input']
    prompts = data.get('prompts')
    if not prompts:
        return {"error": "No prompts provided"}

    # Ensure prompts is a list
    if isinstance(prompts, str):
        prompts = [prompts]
    elif not isinstance(prompts, list):
        return {"error": "Invalid prompt format"}

    try:
        probs = classify_batch(prompts)
        return {"probabilities": probs}
    except Exception as e:
        return {"error": str(e)}
```

The handler reads the `MODEL_NAME` environment variable and initialises a vLLM `LLM` with `task="classify"`【21300289411745†L214-L221】.  Each request must include a JSON body with an `input` field containing a list of `prompts`.  It returns a list of probability vectors, one per prompt.

### 4.3 Dockerfile

To package your handler into a container, build on top of RunPod’s vLLM worker image.  The base image already contains CUDA, PyTorch and vLLM.  You only need to copy your handler code and install additional libraries such as `datasets` if you plan to run evaluations inside the container.

```Dockerfile
FROM runpod/worker-v1-vllm:latest

ENV PYTHONUNBUFFERED=1

# Copy application code
WORKDIR /app
COPY app/ /app/

# Install any extra Python dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Set the handler entrypoint used by RunPod
ENV RUNPOD_HANDLER="app.handler"

# (Optional) pre‑download the model during build to reduce cold start
ARG MODEL_NAME=codefactory4791/intent‑classification‑qwen
RUN python -c "from vllm import LLM; LLM(model='${MODEL_NAME}', task='classify', enforce_eager=True)"

# The base image defines CMD, no need to override
```

Create `app/requirements.txt` containing only the dependencies not already in the base image.  For example:

```
datasets>=2.14.0
```

### 4.4 RunPod Hub configuration (`.runpod/hub.json`)

RunPod’s Hub reads a `hub.json` file to configure your endpoint.  The example below declares your image, passes environment variables to the worker, and exposes pricing.  Adjust values to suit your deployment:

```json
{
  "name": "Intent Classification Qwen via vLLM",
  "description": "Fast intent classification using vLLM.  Serves probabilities for each intent class.",
  "visibility": "private",        
  "image": "ghcr.io/your‑username/intent-classification-runpod:latest",
  "arguments": {
    "MODEL_NAME": "codefactory4791/intent‑classification‑qwen",
    "MAX_NUM_SEQS": "16",
    "QUANTIZATION": "bitsandbytes"
  },
  "network": {
    "type": "http"
  }
}
```

Commit this file under the `.runpod` directory.  When you publish your repository to the RunPod Hub, the Hub reads this configuration and creates a serverless endpoint.

## 5 CI/CD Pipeline

Automate the build and deployment using GitHub Actions.  The workflow below triggers on pushes to the `main` branch.  It builds your Docker image, pushes it to GitHub Container Registry (GHCR), and updates the RunPod endpoint via the RunPod API.

Save the following as `.github/workflows/deploy.yml`:

```yaml
name: Build and Deploy to RunPod

on:
  push:
    branches: [ main ]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}
  RUNPOD_ENDPOINT_ID: ${{ secrets.RUNPOD_ENDPOINT_ID }}
  RUNPOD_API_KEY: ${{ secrets.RUNPOD_API_KEY }}

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Log in to registry
      uses: docker/login-action@v3
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    - name: Build Docker image
      run: |
        docker build -t ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest .
        docker push ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest
    - name: Update RunPod endpoint
      run: |
        echo "Updating RunPod endpoint"
        curl -X POST \
          -H "Authorization: Bearer ${{ env.RUNPOD_API_KEY }}" \
          -H "Content-Type: application/json" \
          -d '{"image": "${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest", "endpointId": "'${{ env.RUNPOD_ENDPOINT_ID }}'"}' \
          https://api.runpod.ai/v2/endpoint/update
```

To use this workflow:

1. Create GitHub secrets `RUNPOD_API_KEY` and `RUNPOD_ENDPOINT_ID`.  The endpoint ID will be created when you first deploy through the RunPod console; copy it from the endpoint’s details page (see step 6).  The API key can be generated in the RunPod console under **API Keys**.
2. Push your code to GitHub.  On each push to `main`, the workflow builds your image, pushes it to GHCR, and sends a POST request to the RunPod API to update the endpoint with the new image.

## 6 Deploy on RunPod

### 6.1 Create the endpoint

Follow RunPod’s vLLM deployment guide:

1. Log in to the RunPod console and open the **Serverless** tab.  Click **Deploy** on the vLLM worker card.
2. In the **Model (optional)** field, enter your Hugging Face model name.  The environment variable `MODEL_NAME` can also be set later【562642474504598†L65-L68】.
3. Click **Advanced** and set your model’s maximum context length (`MAX_MODEL_LEN`), data type (`DTYPE`), memory utilisation, and other settings as needed【138970325587887†L264-L276】.  You can also set `MAX_NUM_SEQS` based on your benchmarking results.
4. Choose a GPU type with enough VRAM (e.g. 16 GB or 24 GB) and click **Create Endpoint**.  Note down the **Endpoint ID**.

Alternatively, if you publish your repository to the RunPod Hub, you can deploy directly from the Hub by clicking **Deploy** on your repository.  The Hub reads `.runpod/hub.json` and provisions an endpoint using your container image.

### 6.2 Edit environment variables

After the endpoint is created, you can customise it by editing the environment variables.  Navigate to the endpoint details page, click **Manage → Edit Endpoint**, expand **Public Environment Variables** and set variables like `MAX_MODEL_LEN`, `DTYPE`, `GPU_MEMORY_UTILIZATION` or `OPENAI_SERVED_MODEL_NAME_OVERRIDE`【138970325587887†L264-L289】.  Save the endpoint to apply changes.

### 6.3 Test the endpoint

Once the endpoint status changes to **Running**, you can send a classification request.  Replace `ENDPOINT_ID` with your actual ID and use your API key in the `Authorization` header.  The request body must wrap the input inside an `input` object because RunPod’s native API expects this structure.

```bash
API_KEY=<your-runpod-api-key>
ENDPOINT_ID=<your-endpoint-id>

curl -X POST \
  -H "Authorization: Bearer ${API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
        "input": {
          "prompts": ["Book me a flight", "Order a pizza for tonight"]
        }
      }' \
  https://api.runpod.ai/v2/${ENDPOINT_ID}/run
```

The response should include a `probabilities` field containing arrays of probabilities for each class.  If you prefer to use the OpenAI‑compatible path, call `/openai/v1/…` and adjust the request accordingly.

## 7 Load Testing with Locust

Locust is a Python tool for simulating concurrent users and measuring API performance.  The following script sends classification requests to your RunPod endpoint.  Save it as `locustfile.py` locally.

```python
from locust import HttpUser, task, between
import json

class ClassificationUser(HttpUser):
    wait_time = between(0.5, 2.0)

    @task
    def classify(self):
        # Example batch of prompts (replace with your own)
        prompts = [
            "Book me a flight to San Francisco next Tuesday.",
            "Add milk and bread to my shopping list.",
            "Schedule a meeting with John tomorrow at 3 PM."
        ]
        payload = {"input": {"prompts": prompts}}
        headers = {
            "Authorization": f"Bearer {self.environment.parsed_options.api_key}",
            "Content-Type": "application/json"
        }
        # Use the /run endpoint for native API
        self.client.post("/run", data=json.dumps(payload), headers=headers)

    # Provide a default host; Locust will override this via CLI
    host = "https://api.runpod.ai/v2/REPLACE_ENDPOINT_ID"
```

### Running the load test

1. Install locust: `pip install locust`.
2. Run the test with your RunPod API key and endpoint ID:

```bash
locust -f locustfile.py --host=https://api.runpod.ai/v2/<ENDPOINT_ID> --headless -u 10 -r 2 --run-time 5m --api-key=<YOUR_RUNPOD_API_KEY>
```

This command spawns 10 users (`-u 10`) ramping up at 2 users/s (`-r 2`) for 5 minutes.  Locust reports statistics such as average response time and requests per second.  Adjust the number of users to explore the throughput limits of your deployment.  Remember to vary the batch size in your `locustfile.py` prompts list to simulate different workloads.

## 8 Monitoring and Optimisation

* **Monitor GPU utilisation:** During load tests, monitor GPU memory and utilisation on RunPod’s metrics page.  Decrease `MAX_NUM_SEQS` or enable quantisation (`QUANTIZATION=bitsandbytes` or `awq`) if you encounter OOM errors【562642474504598†L59-L76】.
* **Tune concurrency:** If latency spikes at higher concurrency, increase the endpoint’s `MAX_CONCURRENCY` environment variable【562642474504598†L98-L100】 or scale out by creating additional replicas.
* **Adjust context length:** For classification tasks, the input sequences are short, so you can lower `MAX_MODEL_LEN` (for example to 512) to save memory.  Set it in the endpoint configuration【138970325587887†L264-L276】.

## 9 Conclusion

By combining **vLLM**’s high‑throughput inference engine with **RunPod**’s serverless GPUs, you can deploy a fine‑tuned classification model as a scalable API in a few steps.  The key points are:

* Use vLLM’s classification task by passing `task="classify"` when constructing the `LLM`【21300289411745†L214-L221】.
* Configure your worker via environment variables to control model name, quantisation, maximum sequences per batch and context length【562642474504598†L59-L100】【138970325587887†L264-L276】.
* Automate deployment through a GitHub Actions workflow that builds your container and updates the RunPod endpoint.
* Test locally to determine optimal batch sizes, then load test the production endpoint with Locust.

With these steps, you should have a robust, production‑ready intent‑classification service running on RunPod.
