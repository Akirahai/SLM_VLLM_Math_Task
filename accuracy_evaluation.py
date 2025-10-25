import pandas as pd
import argparse
from sklearn.metrics import accuracy_score
from tabulate import tabulate
import os
from SLM_VLLM_Math_Task.grader import math_equal
import swifter

def parse_args():
    parser = argparse.ArgumentParser(description="Eignvalue Analysis Setting")
    parser.add_argument('--experiment', type=str, default='22_4_exp', help='Experiment name')
    parser.add_argument('--main_task', type=str, default='Mathematical', help='Task name, mathematical or common reasoning')
    parser.add_argument('--task_list', type=str, nargs='+', default= ['gsm8k','svamp','asdiv','r2ata'], help='Task List, only gsm8k, svamp, asdiv, r2ata')
    parser.add_argument('--model', type=str, help='Model name or path of the model saved in models directory')
    return parser.parse_args()



def evaluate_accuracy(model_name, dataset, experiment, result):
    try:
        file_path = f"{experiment}/{model_name}/{dataset}/{result}.csv"
        print("Loading dataset...")
        print(f"Loading {file_path}...")
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise ValueError("Dataset not found. Please check the dataset name and path.")
    
    preds = df['pred'].tolist()
    labels = df['label'].tolist()

    df['pair'] = [(str(pred), str(label)) for pred, label in zip(preds, labels)]
    df['check'] = df['pair'].swifter.apply(lambda x: math_equal(x[0], x[1], timeout=True))
    track_lst = df['check'].tolist()

    accuracy = sum(track_lst)/len(track_lst)
    
    # Calculate additional metrics if columns exist
    avg_latency = df['latency'].mean() if 'latency' in df.columns else 0.0
    avg_tokens = df['token_count'].mean() if 'token_count' in df.columns else 0.0
    
    return accuracy, len(track_lst), avg_latency, avg_tokens

if __name__== "__main__":
    args = parse_args()
    task_list = args.task_list
    experiment = args.experiment
    model_name = args.model
    
    # Write tabulate of dataset in txt file
    # Write for COT
    row_COT = ["COT"]
    row_COT_latency = ["COT Latency (s)"]
    row_COT_tokens = ["COT Avg Tokens"]
    total_true = 0
    total_len = 0
    total_latency = 0
    total_tokens = 0
    tag_cot = "_cot"
    
    for task in task_list:
        try:
            acc, len_pred, avg_latency, avg_tokens = evaluate_accuracy(model_name, task, experiment, f"results{tag_cot}")
        except ValueError as e:
            print(str(e))
            acc = 0.0
            len_pred = 0
            avg_latency = 0.0
            avg_tokens = 0.0
        row_COT.append(f"{acc:.2%}")
        row_COT_latency.append(f"{avg_latency:.4f}")
        row_COT_tokens.append(f"{avg_tokens:.2f}")
        total_true += acc * len_pred
        total_len += len_pred
        total_latency += avg_latency * len_pred
        total_tokens += avg_tokens * len_pred

    average_accuracy = total_true / total_len if total_len > 0 else 0
    average_latency = total_latency / total_len if total_len > 0 else 0
    average_tokens = total_tokens / total_len if total_len > 0 else 0
    
    row_COT.append(f"{average_accuracy:.2%}")
    row_COT.append(f"{total_len}")
    row_COT_latency.append(f"{average_latency:.4f}")
    row_COT_latency.append("")
    row_COT_tokens.append(f"{average_tokens:.2f}")
    row_COT_tokens.append("")
    
    
    # Write for Non-COT
    row_NonCOT = ["Non-COT"]
    row_NonCOT_latency = ["Non-COT Latency (s)"]
    row_NonCOT_tokens = ["Non-COT Avg Tokens"]
    total_true = 0
    total_len = 0
    total_latency = 0
    total_tokens = 0
    tag_noncot = "_noncot"
    
    for task in task_list:
        try:
            acc, len_pred, avg_latency, avg_tokens = evaluate_accuracy(model_name, task, experiment, f"results{tag_noncot}")
        except ValueError as e:
            print(str(e))
            acc = 0.0
            len_pred = 0
            avg_latency = 0.0
            avg_tokens = 0.0
        row_NonCOT.append(f"{acc:.2%}")
        row_NonCOT_latency.append(f"{avg_latency:.4f}")
        row_NonCOT_tokens.append(f"{avg_tokens:.2f}")
        total_true += acc * len_pred
        total_len += len_pred
        total_latency += avg_latency * len_pred
        total_tokens += avg_tokens * len_pred

    average_accuracy = total_true / total_len if total_len > 0 else 0
    average_latency = total_latency / total_len if total_len > 0 else 0
    average_tokens = total_tokens / total_len if total_len > 0 else 0
    
    row_NonCOT.append(f"{average_accuracy:.2%}")
    row_NonCOT.append(f"{total_len}")
    row_NonCOT_latency.append(f"{average_latency:.4f}")
    row_NonCOT_latency.append("")
    row_NonCOT_tokens.append(f"{average_tokens:.2f}")
    row_NonCOT_tokens.append("")
    
    
    headers = [f"Model {model_name}"] + task_list + ["Average"] + ["Total Samples"]
    table = [row_COT, row_COT_latency, row_COT_tokens, row_NonCOT, row_NonCOT_latency, row_NonCOT_tokens]

    tabulated_result = tabulate(table, headers=headers, tablefmt="grid")

    print(tabulated_result)
    
    save_txt_dir = f"{experiment}/{model_name}/accuracy_results.txt"
    
    with open(save_txt_dir, "w") as f:
        f.write(tabulated_result)