import pandas as pd
import torch
from tqdm import tqdm
from SLM_VLLM_Math_Task.grader import math_equal
import swifter
import re
import time
"""
This file is to compute the metrics for the generated answers. It extracts answer from the generated text and compare it with the label.
The metrics include accuracy, average latency, and average generated tokens.
"""
def extract_boxed(text):
    # match = re.search(r"\\boxed\{(.*?)\}", text)
    # return match.group(1).strip() if match else ""
    matches = re.findall(r"\\boxed\{(.*?)\}", text)
    return matches[-1].strip() if matches else ""

def len_extract_boxed(text):
    matches = re.findall(r"\\boxed\{(.*?)\}", text)
    return len(matches[-1].strip()) if matches else 0



def compute_metrics(decoded_preds, decoded_labels, all_queses, tokenizer=None, latencies=None, save=None):
    # decoded_preds: predicted seq
    # decoded_labels: gold seq
    # tokenizer: tokenizer for computing token counts
    # latencies: list of latencies for each prediction
    full_preds, full_labels = decoded_preds, decoded_labels

    decoded_preds = [extract_boxed(i) for i in decoded_preds]
    print(decoded_labels[0])
    decoded_labels = [i[i.find("The answer is") :].replace("The answer is", "").strip() for i in decoded_labels]

    # Compute token counts for full predictions
    token_counts = []
    if tokenizer is not None:
        for pred in full_preds:
            try:
                tokens = tokenizer.tokenize(str(pred))
                token_counts.append(len(tokens))
            except:
                token_counts.append(0)
    else:
        token_counts = [0] * len(full_preds)

    # Create DataFrame with additional metrics
    df_data = {
        'pred': decoded_preds, 
        'label': decoded_labels, 
        'full_pred': full_preds, 
        'full_label': full_labels, 
        'question': all_queses,
        'token_count': token_counts
    }
    
    # Add latency if provided
    if latencies is not None:
        df_data['latency'] = latencies
    else:
        df_data['latency'] = [0.0] * len(decoded_preds)
    
    df = pd.DataFrame(df_data)
    
    if save is not None:
        df.to_csv(save, index=False)

    preds = df['pred'].tolist()
    labels = df['label'].tolist()

    df['pair'] = [(str(pred), str(label)) for pred, label in zip(preds, labels)]
    df['check'] = df['pair'].swifter.apply(lambda x: math_equal(x[0], x[1], timeout=True))
    track_lst = df['check'].tolist()

    # Compute metrics
    accuracy = sum(track_lst)/len(track_lst)
    avg_latency = df['latency'].mean() if latencies is not None else 0.0
    avg_tokens = df['token_count'].mean()

    result = {
        "accuracy": accuracy,
        "avg_latency": avg_latency,
        "avg_generated_tokens": avg_tokens
    }
    
    return result

@torch.no_grad()
def validate_causal(llm, sampling_params, questions, labels, tokenizer=None, name=None, original_questions=None):
    if original_questions is None:
        original_questions = questions
    
    # Measure total inference time
    start_time = time.time()
    initial_completions = llm.generate(questions, sampling_params=sampling_params)
    initial_inference_time = time.time() - start_time
    
    # Initialize latencies list
    latencies = []
    initial_latency_per_sample = initial_inference_time / len(questions)
    
    if not questions[0].endswith("The answer is \\boxed{"): # if we are using fewshot_cot_prompt
        initial_completions_text = [i.outputs[0].text for i in initial_completions]
        # Prepare a list for the final completions, and lists for modified questions and their indices
        final_completions = [None] * len(questions)
        modified_questions = []
        indices_to_modify = []
        first_completions = []
        
        # Initialize latencies for all samples with initial inference time
        latencies = [initial_latency_per_sample] * len(questions)
        
        for idx, comp in enumerate(initial_completions_text):
            if "boxed" in comp and (len_extract_boxed(comp) >= 1):
                final_completions[idx] = comp
            else:
                # Modify the question by appending " Therefore, the answer is" at the end
                print("The completion got problems")
                print("Completion:", comp)
                modified_question = questions[idx] + comp + " Therefore, the answer is \\boxed{"
                modified_questions.append(modified_question)
                indices_to_modify.append(idx)
                first_completions.append(comp + " Therefore, the answer is \\boxed{")
        
        if len(modified_questions) > 0:
            sampling_params.max_tokens = 64 # Limit the max tokens for the second pass
            start_time_second = time.time()
            new_completions = llm.generate(modified_questions, sampling_params=sampling_params)
            second_inference_time = time.time() - start_time_second
            second_latency_per_sample = second_inference_time
            
            new_completions_text = [i.outputs[0].text for i in new_completions]
            for i, (index, new_comp, first_com) in enumerate(zip(indices_to_modify, new_completions_text, first_completions)):
                final_completions[index] = first_com + new_comp
                # Update latency for samples that needed second inference
                latencies[index] = initial_latency_per_sample + second_latency_per_sample

        preds = final_completions

    else: # add The answer is to convenient extract the answer while computing metrics
        preds = ["The answer is \\boxed{" + i.outputs[0].text for i in initial_completions]
        # All samples have the same latency
        latencies = [initial_latency_per_sample] * len(questions)

    results = compute_metrics(
        preds, 
        labels, 
        all_queses=original_questions, 
        tokenizer=tokenizer, 
        latencies=latencies, 
        save=name
    )
    
    return results


def print_enhanced_metrics(results, model_name="Model", task_name="Task"):
    """
    Print enhanced metrics in a formatted way
    """
    print(f"\n=== {model_name} Results on {task_name} ===")
    print(f"Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"Average Latency: {results['avg_latency']:.4f} seconds")
    print(f"Average Generated Tokens: {results['avg_generated_tokens']:.2f}")
    print("=" * 50)


# def evaluate_accuracy(model_name, dataset, experiment):
#     # Load the dataset
#     try:
#         file_path = f"{expriement}/{model_name}/{dataset}.csv"
#         df = pd.read_csv(file_path)
#     except FileNotFoundError:
#         raise ValueError("Dataset  not found. Please check the dataset name and path.")
    
#     predictions = df['pred'].tolist()
#     ground_truth = df['label'].tolist()
    
#     # Compute accuracy
#     correct_predictions = sum([1 for pred, gt in zip(predictions, ground_truth) if pred == gt])
#     total_predictions = len(predictions)
#     accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
#     return accuracy