import pandas as pd
import torch
from tqdm import tqdm
from grader import math_equal
import swifter
import re
"""
This file is to compute the metrics for the generated answers. It extracts answer from the generated text and compare it with the label.
The metric is accuracy.
"""
def extract_boxed(text):
    # match = re.search(r"\\boxed\{(.*?)\}", text)
    # return match.group(1).strip() if match else ""
    matches = re.findall(r"\\boxed\{(.*?)\}", text)
    return matches[-1].strip() if matches else ""

def len_extract_boxed(text):
    matches = re.findall(r"\\boxed\{(.*?)\}", text)
    return len(matches[-1].strip()) if matches else 0



def compute_metrics(decoded_preds, decoded_labels, all_queses, save=None):
    # decoded_preds: predicted seq
    # decoded_labels: gold seq
    full_preds, full_labels = decoded_preds, decoded_labels

    decoded_preds = [extract_boxed(i) for i in decoded_preds]
    print(decoded_labels[0])
    decoded_labels = [i[i.find("The answer is") :].replace("The answer is", "").strip() for i in decoded_labels]

    df = pd.DataFrame({'pred': decoded_preds, 'label': decoded_labels, 'full_pred': full_preds, 'full_label': full_labels, 'question': all_queses})
    if save is not None:
        df.to_csv(save, index=False)

    preds = df['pred'].tolist()
    labels = df['label'].tolist()

    df['pair'] = [(str(pred), str(label)) for pred, label in zip(preds, labels)]
    df['check'] = df['pair'].swifter.apply(lambda x: math_equal(x[0], x[1], timeout=True))
    track_lst = df['check'].tolist()

    result = {"accuracy": sum(track_lst)/len(track_lst)}
    
    return result

@torch.no_grad()
def validate_causal(llm, sampling_params, questions, labels, name=None, original_questions=None):
    if original_questions is None:
        original_questions = questions
    initial_completions = llm.generate(questions, sampling_params=sampling_params)

    if not questions[0].endswith("The answer is \\boxed{"): # if we are using fewshot_cot_prompt
        initial_completions = [i.outputs[0].text for i in initial_completions]
        # Prepare a list for the final completions, and lists for modified questions and their indices
        final_completions = [None] * len(questions)
        modified_questions = []
        indices_to_modify = []
        first_completions = []
        for idx, comp in enumerate(initial_completions):
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
            new_completions = llm.generate(modified_questions, sampling_params=sampling_params)
            new_completions = [i.outputs[0].text for i in new_completions]
            for index, new_comp, first_com in zip(indices_to_modify, new_completions, first_completions):
                final_completions[index] = first_com + new_comp

        preds = final_completions

    else: # add The answer is to convinient extract the answer while computing metrics
        preds = ["The answer is \\boxed{" + i.outputs[0].text for i in initial_completions]

    results = compute_metrics(preds, labels, save=name, all_queses=original_questions)
    avg_accuracy = results['accuracy']

    return avg_accuracy


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