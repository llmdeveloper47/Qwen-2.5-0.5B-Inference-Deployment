# Experiment Log

Use this template to track your quantization and batch size experiments.

## Experiment Overview

**Date Started:** YYYY-MM-DD
**Researcher:** Your Name
**Objective:** Test quantization methods and batch sizes to optimize latency/throughput trade-off

## Hardware Configuration

- **GPU**: NVIDIA A100 40GB (RunPod)
- **CUDA Version**: 12.1
- **Driver Version**: XXX.XX
- **vLLM Version**: 0.7.0+

## Experiment Matrix

| Quantization | Batch Sizes | Status | Results File |
|--------------|-------------|--------|--------------|
| none (FP16)  | 1,4,8,16,32 | ⏳ Pending | results/experiments/none/ |
| bitsandbytes | 1,4,8,16,32 | ⏳ Pending | results/experiments/bitsandbytes/ |
| awq          | 1,4,8,16,32 | ⏳ Pending | results/experiments/awq/ |
| gptq         | 1,4,8,16,32 | ⏳ Pending | results/experiments/gptq/ |

**Status Legend:**
- ⏳ Pending
- 🔄 In Progress
- ✅ Complete
- ❌ Failed/Skipped

---

## Experiment 1: FP16 Baseline

**Date:** YYYY-MM-DD
**Quantization:** none
**Configuration:** See `configs/fp16_baseline.json`

### Results

| Batch Size | Avg Latency (ms) | P95 Latency (ms) | P99 Latency (ms) | Throughput (samples/s) | Notes |
|------------|------------------|------------------|------------------|------------------------|-------|
| 1          |                  |                  |                  |                        |       |
| 4          |                  |                  |                  |                        |       |
| 8          |                  |                  |                  |                        |       |
| 16         |                  |                  |                  |                        |       |
| 32         |                  |                  |                  |                        |       |

### Observations

- GPU Memory Usage: X.X GB
- Cold Start Time: XXs
- Accuracy: XX.XX%
- Notes: [Your observations here]

### Issues/Challenges

- [Any issues encountered]

---

## Experiment 2: BitsAndBytes INT8

**Date:** YYYY-MM-DD
**Quantization:** bitsandbytes
**Configuration:** See `configs/bitsandbytes_int8.json`

### Results

| Batch Size | Avg Latency (ms) | P95 Latency (ms) | P99 Latency (ms) | Throughput (samples/s) | Notes |
|------------|------------------|------------------|------------------|------------------------|-------|
| 1          |                  |                  |                  |                        |       |
| 4          |                  |                  |                  |                        |       |
| 8          |                  |                  |                  |                        |       |
| 16         |                  |                  |                  |                        |       |
| 32         |                  |                  |                  |                        |       |

### Observations

- GPU Memory Usage: X.X GB
- Memory Reduction vs FP16: XX%
- Accuracy: XX.XX% (degradation: X.XX%)
- Notes: [Your observations here]

### Comparison to FP16

- Latency difference: +/- XX%
- Throughput difference: +/- XX%
- Memory savings: XX GB

---

## Experiment 3: AWQ 4-bit

**Date:** YYYY-MM-DD
**Quantization:** awq
**Configuration:** See `configs/awq_4bit.json`

### Pre-Quantization Setup

- [ ] Model pre-quantized?: Yes/No
- [ ] Quantization method used: [AutoAWQ/Other]
- [ ] Quantization time: XX minutes
- [ ] Quantized model size: XX MB

### Results

| Batch Size | Avg Latency (ms) | P95 Latency (ms) | P99 Latency (ms) | Throughput (samples/s) | Notes |
|------------|------------------|------------------|------------------|------------------------|-------|
| 1          |                  |                  |                  |                        |       |
| 4          |                  |                  |                  |                        |       |
| 8          |                  |                  |                  |                        |       |
| 16         |                  |                  |                  |                        |       |
| 32         |                  |                  |                  |                        |       |

### Observations

- GPU Memory Usage: X.X GB
- Memory Reduction vs FP16: XX%
- Accuracy: XX.XX% (degradation: X.XX%)
- Speed improvement vs FP16: +XX%

---

## Experiment 4: GPTQ 4-bit

**Date:** YYYY-MM-DD
**Quantization:** gptq
**Configuration:** See `configs/gptq_4bit.json`

### Pre-Quantization Setup

- [ ] Model pre-quantized?: Yes/No
- [ ] Quantization method used: [AutoGPTQ/Other]
- [ ] Quantization time: XX minutes
- [ ] Quantized model size: XX MB

### Results

| Batch Size | Avg Latency (ms) | P95 Latency (ms) | P99 Latency (ms) | Throughput (samples/s) | Notes |
|------------|------------------|------------------|------------------|------------------------|-------|
| 1          |                  |                  |                  |                        |       |
| 4          |                  |                  |                  |                        |       |
| 8          |                  |                  |                  |                        |       |
| 16         |                  |                  |                  |                        |       |
| 32         |                  |                  |                  |                        |       |

### Observations

- GPU Memory Usage: X.X GB
- Performance vs AWQ: [Better/Worse/Similar]
- Accuracy: XX.XX% (degradation: X.XX%)

---

## Load Testing Results

### Configuration
- **Tool**: Locust
- **Users**: 10-50 concurrent users
- **Test Duration**: 5 minutes per test
- **Batch Size**: Varied (1, 4, 8, 16, 32)

### Results Summary

| Quantization | Batch Size | Users | RPS | P95 (ms) | P99 (ms) | Failures | Notes |
|--------------|------------|-------|-----|----------|----------|----------|-------|
|              |            |       |     |          |          |          |       |

---

## Key Findings

### Best Configuration for Latency
- **Quantization**: [Method]
- **Batch Size**: [Size]
- **P95 Latency**: XX.XX ms
- **Reasoning**: [Why this configuration performs best for latency]

### Best Configuration for Throughput
- **Quantization**: [Method]
- **Batch Size**: [Size]
- **Throughput**: XXX samples/s
- **Reasoning**: [Why this configuration performs best for throughput]

### Best Balanced Configuration
- **Quantization**: [Method]
- **Batch Size**: [Size]
- **Balance Score**: X.XXX
- **Reasoning**: [Why this is the best overall choice]

### Cost Analysis
- **Most Cost-Effective**: [Configuration]
- **Cost per 1K requests**: $X.XX
- **Cost per 1M requests**: $XX.XX

---

## Recommendations

### Production Deployment

For production deployment, we recommend:

**Configuration:** [Recommended config]

**Reasoning:**
- [Reason 1]
- [Reason 2]
- [Reason 3]

**Deployment Settings:**
```
QUANTIZATION=[value]
MAX_NUM_SEQS=[value]
GPU_MEMORY_UTILIZATION=[value]
```

### Cost Optimization

To optimize costs:
1. Use [Quantization method] for XX% memory reduction
2. Batch size of [X] provides best throughput/latency ratio
3. Consider spot instances for [X]% cost savings

### Scaling Recommendations

- **Low Traffic** (<100 req/min): Single worker, batch size 1-4
- **Medium Traffic** (100-1000 req/min): Auto-scaling 1-3 workers, batch size 8-16
- **High Traffic** (>1000 req/min): Auto-scaling 3-10 workers, batch size 16-32

---

## Lessons Learned

### What Worked Well

1. [Finding 1]
2. [Finding 2]
3. [Finding 3]

### Challenges Faced

1. **Challenge**: [Description]
   **Solution**: [How you solved it]

2. **Challenge**: [Description]
   **Solution**: [How you solved it]

### Unexpected Findings

- [Unexpected finding 1]
- [Unexpected finding 2]

---

## Future Work

### Potential Optimizations

1. [ ] Test with different model architectures
2. [ ] Experiment with different `MAX_MODEL_LEN` values
3. [ ] Try mixed-precision inference
4. [ ] Test on different GPU types (A10, L40, etc.)
5. [ ] Implement response caching for common queries
6. [ ] Test continuous batching strategies

### Next Experiments

1. [Planned experiment 1]
2. [Planned experiment 2]

---

## Appendix

### Useful Commands

```bash
# Quick test endpoint
python scripts/test_endpoint.py --endpoint-id XXX --api-key XXX

# Run specific batch size benchmark
python scripts/benchmark_local.py --batch-sizes 8 --quantization bitsandbytes

# Generate fresh analysis
python scripts/analyze_results.py --results-dir ./results

# Create summary CSV
python scripts/summarize_results.py --results-dir ./results

# Generate PDF report
python scripts/generate_report.py --results-dir ./results
```

### Dataset Information

- **Name**: Amazon Intent Classification Test Set
- **Source**: https://huggingface.co/datasets/codefactory4791/amazon_test
- **Samples**: 10,350 (test split)
- **Classes**: 23 intent categories
- **Format**: Text reviews with intent labels

### Model Information

- **Base Model**: Qwen 2.5-0.5B-Instruct
- **Fine-tuned Model**: codefactory4791/intent-classification-qwen
- **Parameters**: ~494M
- **Task**: Sequence classification (23 classes)
- **Accuracy**: ~92% on test set

---

**Last Updated:** [Date]
**Version:** 1.0

