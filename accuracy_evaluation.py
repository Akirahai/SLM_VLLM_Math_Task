import pandas as pd
import argparse
from sklearn.metrics import accuracy_score
from tabulate import tabulate
import os
from grader import math_equal
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

    result = sum(track_lst)/len(track_lst)
    
    return result, len(track_lst)

if __name__== "__main__":
    args = parse_args()
    task_list = args.task_list
    experiment = args.experiment
    model_name = args.model
    
    # Write tabulate of dataset in txt file
    # Write for COT
    row_COT = ["COT"]
    total_true = 0
    total_len = 0
    tag_cot = "_cot"
    
    for task in task_list:
        try:
            acc, len_pred = evaluate_accuracy(model_name, task, experiment, f"results{tag_cot}")
        except ValueError as e:
            print(str(e))
            acc = 0.0
            len_pred = 0
        row_COT.append(f"{acc:.2%}")
        total_true += acc * len_pred
        total_len += len_pred

    average_accuracy = total_true / total_len
    
    row_COT.append(f"{average_accuracy:.2%}")
    row_COT.append(f"{total_len}")
    
    
    # Write for Non-COT
    row_NonCOT = ["Non-COT"]
    total_true = 0
    total_len = 0
    tag_noncot = "_noncot"
    
    for task in task_list:
        try:
            acc, len_pred = evaluate_accuracy(model_name, task, experiment, f"results{tag_noncot}")
        except ValueError as e:
            print(str(e))
            acc = 0.0
            len_pred = 0
        row_NonCOT.append(f"{acc:.2%}")
        total_true += acc * len_pred
        total_len += len_pred

    average_accuracy = total_true / total_len
    
    row_NonCOT.append(f"{average_accuracy:.2%}")
    row_NonCOT.append(f"{total_len}")
    
    
    headers = [f"Model {model_name}"] + task_list + ["Average"] + ["Total Samples"]
    table = [row_COT, row_NonCOT]

    tabulated_result = tabulate(table, headers=headers, tablefmt="grid")

    print(tabulated_result)
    
    save_txt_dir = f"{experiment}/{model_name}/accuracy_results.txt"
    
    with open(save_txt_dir, "w") as f:
        f.write(tabulated_result)