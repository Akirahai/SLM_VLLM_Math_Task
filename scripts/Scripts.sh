#!/bin/bash

# Define the models and datasets
models=(
  # "Qwen/Qwen2.5-7B-Instruct"
  "Qwen/Qwen2.5-Math-7B-Instruct"
  "meta-llama/Llama-3.1-8B-Instruct"
  "mistralai/Mathstral-7B-v0.1"
  "mistralai/Mistral-7B-Instruct-v0.3"
)

datasets=(
  "gsm8k"
  "svamp"
  "asdiv"
)

task_list=("gsm8k" "svamp" "asdiv")

exp="exp_13_10"


# Loop through each model and dataset combination
for model in "${models[@]}"; do
  for dataset in "${datasets[@]}"; do
    echo "Running CoT evaluation for model: $model on dataset: $dataset"
    python evaluate_causal.py --gpus 3 --model "$model" --task "$dataset" --experiment "$exp" --cot
    sleep 50
    echo "Running non-CoT evaluation for model: $model on dataset: $dataset"
    python evaluate_causal.py --gpus 4 --model "$model" --task "$dataset" --experiment "$exp"
  done

  echo "Running total evaluation for model: $model"
  python accuracy_evaluation.py --model "$model" --task_list "${task_list[@]}" --experiment "$exp"
done