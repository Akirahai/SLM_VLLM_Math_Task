# Enhanced VLLM Metrics

This update adds comprehensive performance metrics to your VLLM evaluation pipeline beyond just accuracy.

## New Metrics Added

1. **Accuracy** (existing) - Correctness of generated answers
2. **Average Latency** (new) - Time taken per inference in seconds
3. **Average Generated Tokens** (new) - Number of tokens generated per response

## Changes Made

### 1. Enhanced `metrics.py`
- Added timing measurement in `validate_causal()` function
- Enhanced `compute_metrics()` to accept tokenizer and latencies
- Added token counting using `tokenizer.tokenize(text)`
- New CSV output includes `latency` and `token_count` columns

### 2. Updated `evaluate_causal.py`
- Passes tokenizer to metrics functions
- Enhanced output file now includes all three metrics
- Proper timing measurement for both single and two-pass inference

### 3. Enhanced `accuracy_evaluation.py`
- Updated to read and display all three metrics
- Enhanced tabular output with separate rows for latency and token metrics
- Weighted averages across multiple tasks

## Usage

### Basic Usage (unchanged)
```bash
python evaluate_causal.py --model meta-llama/Llama-3.1-8B-Instruct --task gsm8k --experiment exp_13_10 --cot
```

### Analyzing Results
```bash
python example_enhanced_metrics.py --csv_path exp_13_10/model_name/gsm8k/results_cot.csv --model_name "Llama-3.1-8B" --task_name "GSM8K"
```

### Evaluation with Enhanced Metrics
```bash
python accuracy_evaluation.py --experiment exp_13_10 --model meta-llama/Llama-3.1-8B-Instruct --task_list gsm8k svamp asdiv
```

## Output Format

### CSV Files
Now include additional columns:
- `pred`: Extracted prediction
- `label`: Ground truth label
- `full_pred`: Complete generated text
- `full_label`: Complete label text
- `question`: Original question
- `latency`: Per-sample inference time (seconds)
- `token_count`: Number of generated tokens

### Console Output
```
Accuracy w/ greedy decoding: 0.7234
Average Latency: 0.8421 seconds
Average Generated Tokens: 142.34
```

### Enhanced Evaluation Table
```
+-------------------------+--------+-------+-------+---------+---------------+
| Model meta-llama/...    | gsm8k  | svamp | asdiv | Average | Total Samples |
+=========================+========+=======+=======+=========+===============+
| COT                     | 72.34% | 68.91%| 75.12%| 72.12%  | 1500          |
+-------------------------+--------+-------+-------+---------+---------------+
| COT Latency (s)         | 0.8421 | 0.7892| 0.9123| 0.8479  |               |
+-------------------------+--------+-------+-------+---------+---------------+
| COT Avg Tokens          | 142.34 | 128.67| 156.89| 142.63  |               |
+-------------------------+--------+-------+-------+---------+---------------+
```

## Performance Insights

The enhanced metrics provide valuable insights:

1. **Latency Analysis**: Identify bottlenecks and compare model efficiency
2. **Token Efficiency**: Understand verbosity vs accuracy trade-offs
3. **Comprehensive Evaluation**: Balance accuracy, speed, and resource usage

## Backward Compatibility

- Existing scripts continue to work
- Enhanced features are optional
- CSV files maintain existing columns while adding new ones
- Legacy metric calculations remain unchanged

## Dependencies

No new dependencies added - uses existing libraries:
- `time` (built-in) for latency measurement
- `tokenizer.tokenize()` (existing) for token counting
- Enhanced pandas DataFrame structure