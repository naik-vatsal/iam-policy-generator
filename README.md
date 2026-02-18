# 🔐 IAM Policy Generator

### Convert natural language → valid AWS IAM JSON policies using a fine-tuned LLM with RAG

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Model](https://img.shields.io/badge/Base_Model-Mistral--7B-purple)

---

## What It Does

Type a description. Get a valid IAM policy.

**Input:**
```
Allow a Lambda function to read from DynamoDB table users and write logs to CloudWatch
```

**Output:**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["dynamodb:GetItem", "dynamodb:Query", "dynamodb:Scan"],
      "Resource": "arn:aws:dynamodb:*:*:table/users"
    },
    {
      "Effect": "Allow",
      "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
      "Resource": "arn:aws:logs:*:*:*"
    }
  ]
}
```

---

## How It Works

```
User Description
       │
       ▼
┌──────────────┐     ┌──────────────────┐
│  Service      │────▶│  AWS Action       │
│  Detection    │     │  Catalog (RAG)    │
└──────┬───────┘     └────────┬─────────┘
       │                      │
       ▼                      ▼
┌──────────────────────────────────────┐
│  Fine-tuned Mistral-7B (QLoRA)       │
│  with retrieved actions as context   │
└──────────────────┬───────────────────┘
                   │
                   ▼
            ┌──────────────┐
            │  JSON Valid?  │──── ✅ Return policy
            └──────┬───────┘
                   │ ❌
                   ▼
            Return raw + error flag
```

1. **Detect** which AWS services the user mentions (S3, EC2, Lambda, etc.)
2. **Retrieve** valid IAM actions for those services from a local catalog (20+ services, 300+ actions)
3. **Inject** the actions as context into the prompt
4. **Generate** the policy using fine-tuned Mistral-7B with QLoRA
5. **Validate** the output JSON and return

---

## Quick Start

### Prerequisites

- Python 3.10+
- NVIDIA GPU with 16+ GB VRAM (A100/V100/T4)
- CUDA 12.1+

### Installation

```bash
git clone https://github.com/yourusername/iam-policy-generator.git
cd iam-policy-generator
pip install -r requirements.txt
```

### Usage

**Python API:**
```python
from src.rag_pipeline import RAGPolicyGenerator

generator = RAGPolicyGenerator("results/config_a/final_model")

# Generate a policy
result = generator.generate("Allow read-only access to S3 bucket customer-data")
print(result["policy"])

# Compare RAG vs no-RAG
generator.compare_rag_vs_no_rag("Allow ECS task to pull from ECR and read secrets")
```

**Gradio Web UI:**
```bash
python src/rag_app.py
# Opens at http://localhost:7860
```

**Basic inference (no RAG):**
```python
from src.inference_pipeline import IAMPolicyGenerator

generator = IAMPolicyGenerator("results/config_a/final_model")
result = generator.generate("Deny all EC2 terminate operations")
```

---

## Results

### Baseline Comparison (151 test examples)

| Metric | Zero-Shot | Few-Shot | Fine-Tuned |
|--------|-----------|----------|------------|
| JSON Valid Rate | 0.7% | 2.0% | **60.3%** |
| Schema Valid Rate | 78.8%* | 7.3% | **60.3%** |
| Service Accuracy | 28.7% | 0.0% | **52.0%** |
| Effect Accuracy | 98.3% | 100.0% | **95.6%** |

*Zero-shot wraps JSON in markdown code blocks — not directly parseable.

### Accuracy by Policy Complexity

| Complexity | Accuracy | Example |
|------------|----------|---------|
| Simple | 76.2% | Single service, 1 statement |
| Medium | 66.7% | 2-3 services or conditions |
| Complex | 47.1% | 4+ statements, multi-service |

### Hyperparameter Search (4 configs)

| Config | LR | LoRA Rank | Best Eval Loss |
|--------|-----|-----------|----------------|
| **A** ✅ | 2e-4 | 16 | **0.3019** |
| B | 1e-4 | 32 | 0.3025 |
| C | 5e-4 | 16 | 0.3207 (overfit) |
| D | 2e-4 | 8 | 0.3114 (underfit) |

---

## Project Structure

```
iam-policy-generator/
│
├── src/                                # Source code
│   ├── inference_pipeline.py           # Basic inference class
│   ├── rag_pipeline.py                 # RAG-enhanced inference with AWS action catalog
│   └── rag_app.py                      # Gradio web interface
│
├── training/                           # Training notebooks
│   ├── 01_Dataset_Preparation.ipynb    # Data collection, cleaning, splitting
│   └── 02_Model_Training.ipynb        # Training, evaluation, error analysis
│
├── data/
│   └── processed/                      # Final dataset
│       ├── train.jsonl                 # 1,189 examples (80%)
│       ├── val.jsonl                   # 148 examples (10%)
│       └── test.jsonl                  # 151 examples (10%)
│
├── results/
│   ├── config_a/final_model/           # Best LoRA adapter (168 MB, Git LFS)
│   ├── config_*/training_logs.json     # Training logs for all 4 configs
│   ├── evaluation_metrics.json         # Full evaluation results
│   ├── evaluation_charts.png           # Comparison visualizations
│   ├── hp_comparison_chart.png         # Hyperparameter comparison chart
│   ├── error_analysis.json             # Error categorization
│   ├── zero_shot_results.json          # Baseline: zero-shot outputs
│   ├── few_shot_results.json           # Baseline: few-shot outputs
│   └── finetuned_results.json          # Fine-tuned model outputs
│
├── docs/
│   └── technical_report.pdf            # 7-page technical report
│
├── requirements.txt
├── LICENSE                             # MIT
├── .gitignore
└── .gitattributes                      # Git LFS for model weights
```

---

## Technical Details

### Model

| Parameter | Value |
|-----------|-------|
| Base model | [Mistral-7B-v0.3](https://huggingface.co/mistralai/Mistral-7B-v0.3) |
| Fine-tuning method | QLoRA (4-bit NF4 quantization) |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Target modules | q, k, v, o, gate, up, down projections |
| Trainable parameters | 41.9M (0.58% of 7.29B) |
| GPU memory | 4.1 GB |
| Training time | 23 min on A100-80GB |

### Dataset

| Property | Value |
|----------|-------|
| Total examples | 1,488 |
| Source 1 | AWS managed policies via boto3 API (1,417) |
| Source 2 | Hand-crafted synthetic examples (71) |
| Services covered | 30+ AWS services |
| Format | Alpaca instruction format |
| Split | 80% train / 10% val / 10% test |

### RAG Pipeline

| Property | Value |
|----------|-------|
| Action catalog | 20+ AWS services, 300+ actions |
| Service detection | Keyword matching with aliases |
| Context injection | Prepended to instruction prompt |
| Fallback | Works without RAG (uses fine-tuned model only) |

---

## Reproducing Results

### Step 1: Prepare Dataset

```bash
# Configure AWS credentials
aws configure

# Run dataset notebook
jupyter notebook training/01_Dataset_Preparation.ipynb
```

### Step 2: Train Model

```bash
# Requires GPU (A100 recommended)
jupyter notebook training/02_Model_Training.ipynb
```

### Step 3: Evaluate

Evaluation cells are included in the training notebook (Sections 8-13).

### Step 4: Run Inference

```python
from src.rag_pipeline import RAGPolicyGenerator
generator = RAGPolicyGenerator("results/config_a/final_model")
generator.generate_and_print("Allow read-only access to S3 bucket my-data")
```

---

## Error Analysis

| Error Category | Rate | Root Cause | Fix |
|---------------|------|------------|-----|
| ✅ Correct | 29.1% | — | — |
| Truncated JSON | 39.7% | Complex policies exceed token limit | Increase max_new_tokens |
| Wrong services | 28.5% | Hallucinated action names | **RAG pipeline** (implemented) |
| Wrong effect | 2.6% | Allow/Deny confusion | More deny examples in training |
| Invalid schema | 0.0% | — | Model never breaks structure ✅ |

---

## Roadmap

- [x] Fine-tuned Mistral-7B with QLoRA
- [x] RAG pipeline with AWS action catalog
- [x] Gradio web interface
- [x] Three-way evaluation (zero-shot / few-shot / fine-tuned)
- [x] Error analysis and visualizations
- [ ] Constrained JSON decoding (grammar-guided generation)
- [ ] Expand catalog to all 300+ AWS services
- [ ] Expand dataset to 5,000+ examples
- [ ] Model distillation for CPU deployment
- [ ] VS Code extension
- [ ] CLI tool (`iam-gen "Allow S3 read for bucket X"`)

---

## Contributing

Contributions welcome! Areas where help is needed:

1. **AWS Action Catalog**: Expand coverage beyond 20 services
2. **Dataset**: More complex multi-service policy examples
3. **Evaluation**: Additional metrics (action-level precision/recall)
4. **Deployment**: Docker container, AWS Lambda packaging

---

## Citation

If you use this project in your research:

```bibtex
@software{naik2026iam,
  author = {Naik, Vatsal},
  title = {IAM Policy Generator: Fine-tuned LLM with RAG for AWS IAM Policy Generation},
  year = {2026},
  url = {https://github.com/yourusername/iam-policy-generator}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
