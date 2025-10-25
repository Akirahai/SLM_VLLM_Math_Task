import argparse
import os
import logging
from dotenv import load_dotenv
load_dotenv()
access_token = os.getenv("HF_ACCESS_TOKEN")
logging.getLogger("vllm.engine.scheduler").setLevel(logging.ERROR)
os.environ["VLLM_USE_V1"] = "0"

import shutil
from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoModelForCausalLM
from vllm import LLM, SamplingParams

from metrics import validate_causal

from read_data import  zeroshot_cot_template, zeroshot_noncot_template, read_data
from utils import set_seed, setup_directories


# CUDA_VISIBLE_DEVICES=0 python evaluate_causal.py --model_gen mistral --size 7B --seed 0 --MAX_LEN 1024 --cot
# CUDA_VISIBLE_DEVICES=1 python evaluate_causal.py --model_gen mistral --size 13B --seed 0 --MAX_LEN 1024 --cot
# CUDA_VISIBLE_DEVICES=2 python evaluate_causal.py --model_gen llama --size 7B --seed 0 --MAX_LEN 1024 --cot
# CUDA_VISIBLE_DEVICES=3 python evaluate_causal.py --model_gen llama --size 13B --seed 0 --MAX_LEN 1024 --cot

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation setting details")
    parser.add_argument('--gpus', type=int, nargs='+', default=[5, 6], help='List of gpus to use')
    parser.add_argument('--device', type=int, default=1, help='GPU id')
    parser.add_argument('--experiment', type=str, default='8_4_exp', help='Experiment name')
    parser.add_argument('--task', type=str, default='gsm8k', help='Task name, only gsm8k, svamp, asdiv, r2ata')
    parser.add_argument('--model', type=str, help='Model name or path of the model saved in models directory')
    # parser.add_argument('--model_gen', type=str, choices=["llama", "mistral"], default='mistral', help='Model family')
    # parser.add_argument('--size', type=str, choices=['7B', '13B', '8B'], default='13B', help='Model size')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument("--MAX_LEN", type=int, default=1024, help="Max length limitation for tokenized texts")
    parser.add_argument("--cot", action="store_true", help="Enable Few-shot CoT.")
    
    
    return parser.parse_args()


if __name__== "__main__":
    
    args = parse_args()
    
    GPU_list = ','.join(map(str, args.gpus))

    os.environ['CUDA_DEVICE_ORDER'] =  'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES']=  GPU_list

    print(f"Using GPU", os.environ['CUDA_VISIBLE_DEVICES'])

    device = "cuda"

    # import torch
    
    # if args.use_gpu and torch.cuda.is_available(): 
    #     device = torch.device(f'cuda')
    #       # Change to your suitable GPU device
    #     print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    # else:
    #     device = torch.device('cpu')
    #     print("Using CPU")
    

    from huggingface_hub import login
    login(token=access_token)

    model_name = args.model
    set_seed(args.seed)
    experiment = args.experiment


    MAX_LEN = args.MAX_LEN  
    
    
    
    save_dir = f"{experiment}/{model_name}/{args.task}"
    os.makedirs(save_dir, exist_ok=True)


    if args.task in ['gsm8k', 'svamp', 'asdiv', 'r2ata']:
        input, output = read_data(args.task)
    else:
        raise ValueError("Task not supported. Please choose from ['gsm8k', 'svamp', 'asdiv', 'r2ata']")

    if args.cot:
        fewshot_template = zeroshot_cot_template
        tag = "_cot"
    else:
        fewshot_template = zeroshot_noncot_template
        tag = '_noncot'

    save_txt_dir = f"{experiment}/{model_name}/{args.task}/results{tag}.txt"
    
    # Template for few-shot CoT and NonCot
    formated_input = [fewshot_template.format_map({'instruction': sentence}) for sentence in input]
    # formated_test_input = [fewshot_template.format_map({'instruction': sentence}) for sentence in test_input]
    # formated_svamp_input = [fewshot_template.format_map({'instruction': sentence}) for sentence in svamp_input]
    # formated_asdiv_input = [fewshot_template.format_map({'instruction': sentence}) for sentence in asdiv_input]
    # formated_r2ata_input = [fewshot_template.format_map({'instruction': sentence}) for sentence in r2ata_input]

    
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if args.cot:
        stop_tokens = tokenizer.additional_special_tokens
        sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=MAX_LEN, stop=stop_tokens)
    else:
        stop_tokens = ['.', '\n'] + tokenizer.additional_special_tokens
        sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=MAX_LEN, stop=stop_tokens)
    tensor_parallel_size = 1
    llm = LLM(model=model_name,tensor_parallel_size=tensor_parallel_size)

    test_results = validate_causal(
        llm, 
        sampling_params, 
        formated_input, 
        output, 
        tokenizer=tokenizer,
        name=os.path.join(save_dir, f'results{tag}.csv'), 
        original_questions=input
    )

    # Extract individual metrics
    test_accuracy = test_results['accuracy']
    avg_latency = test_results['avg_latency']
    avg_tokens = test_results['avg_generated_tokens']

    # Save enhanced results to a text file
    with open(save_txt_dir, "w") as file:
        message_accuracy = f"Accuracy w/ greedy decoding: {test_accuracy:.4f}"
        message_latency = f"Average Latency: {avg_latency:.4f} seconds"
        message_tokens = f"Average Generated Tokens: {avg_tokens:.2f}"
        
        full_message = f"{message_accuracy}\n{message_latency}\n{message_tokens}"
        
        print(message_accuracy)           # Print to console
        print(message_latency)
        print(message_tokens)
        print(full_message, file=file)   # Write to file