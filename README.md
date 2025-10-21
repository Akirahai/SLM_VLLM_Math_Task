# Zero-Shot Math LLM Evaluation

A simple toolkit to evaluate Large Language Models (LLMs) on mathematical reasoning benchmarks using zero-shot prompting. It uses `vLLM` for fast inference and automates accuracy calculation.

### ✨ Core Features
- **Zero-Shot Evaluation:** Test models without few-shot examples.
- **CoT & Non-CoT Modes:** Supports both Chain-of-Thought and direct-answer prompting.
- **Fast Inference:** Powered by `vLLM` for high-throughput evaluation.
- **Automated Grading:** Automatically computes accuracy on standard math datasets.
- **Simple & Reproducible:** Easy setup and scriptable workflow.

---

## 🚀 Quick Start

Follow these steps to set up and run your first evaluation.

### 1. Setup Environment
```bash
# Clone the repository
git clone <your-repository-url>
cd <your-repository-directory>

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Add Hugging Face Token
Create a file named `.env` and add your token to it.
```bash
echo "HF_ACCESS_TOKEN=your_hf_token_here" > .env
```

---

## 🧩 How to Run

The process is two steps: first run inference, then aggregate the results.

### Step 1: Run Inference
Run this command for each dataset you want to evaluate.

**With Chain-of-Thought (CoT):**
```bash
python evaluate_causal.py \
    --gpus 0 \
    --model "mistralai/Mathstral-7B-v0.1" \
    --task gsm8k \
    --experiment exp_13_10 \
    --cot
```

**Without CoT (direct answering):**
```bash
python evaluate_causal.py \
    --gpus 0 \
    --model "mistralai/Mathstral-7B-v0.1" \
    --task gsm8k \
    --experiment exp_13_10
```
> **Note:** The outputs (CSV files and summaries) will be saved in the `my_first_eval/` folder.

### Step 2: Compute Final Accuracy
After running inference on all desired datasets (e.g., `gsm8k`, `asdiv`, `svamp`), aggregate the results.
```bash
python accuracy_evaluation.py \
    --model "mistralai/Mathstral-7B-v0.1" \
    --task_list gsm8k asdiv svamp \
    --experiment exp_13_10
```
> This will create a final `accuracy_results.txt` file in the experiment directory.

---

## 🧭 Future Plans
- Add token usage and latency tracking.
- Improve batch scripting for running multiple models in parallel.