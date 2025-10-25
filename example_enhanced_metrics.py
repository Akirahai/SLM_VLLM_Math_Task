"""
Example script showing how to use the enhanced metrics functionality
This demonstrates the new features: accuracy, average latency, and average generated tokens
"""

import pandas as pd
from metrics import print_enhanced_metrics
from SLM_VLLM_Math_Task.grader import math_equal
import swifter

def analyze_results_file(csv_path, model_name="Unknown", task_name="Unknown"):
    """
    Analyze a results CSV file and print enhanced metrics
    """
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded results from: {csv_path}")
        print(f"Total samples: {len(df)}")
        
        # Check if we have the enhanced columns
        required_cols = ['pred', 'label']
        enhanced_cols = ['latency', 'token_count']
        
        if not all(col in df.columns for col in required_cols):
            print(f"Error: Missing required columns. Found: {df.columns.tolist()}")
            return
        
        # Calculate accuracy
        preds = df['pred'].tolist()
        labels = df['label'].tolist()
        df['pair'] = [(str(pred), str(label)) for pred, label in zip(preds, labels)]
        df['check'] = df['pair'].swifter.apply(lambda x: math_equal(x[0], x[1], timeout=True))
        accuracy = df['check'].mean()
        
        # Calculate enhanced metrics if available
        avg_latency = df['latency'].mean() if 'latency' in df.columns else 0.0
        avg_tokens = df['token_count'].mean() if 'token_count' in df.columns else 0.0
        
        # Create results dictionary
        results = {
            'accuracy': accuracy,
            'avg_latency': avg_latency,
            'avg_generated_tokens': avg_tokens
        }
        
        # Print formatted results
        print_enhanced_metrics(results, model_name, task_name)
        
        # Additional analysis
        if 'latency' in df.columns:
            print(f"Latency stats - Min: {df['latency'].min():.4f}s, Max: {df['latency'].max():.4f}s, Std: {df['latency'].std():.4f}s")
        
        if 'token_count' in df.columns:
            print(f"Token count stats - Min: {df['token_count'].min():.0f}, Max: {df['token_count'].max():.0f}, Std: {df['token_count'].std():.2f}")
        
        return results
        
    except Exception as e:
        print(f"Error analyzing file {csv_path}: {e}")
        return None


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze enhanced metrics from results CSV")
    parser.add_argument('--csv_path', type=str, required=True, help='Path to results CSV file')
    parser.add_argument('--model_name', type=str, default='Unknown', help='Model name for display')
    parser.add_argument('--task_name', type=str, default='Unknown', help='Task name for display')
    
    args = parser.parse_args()
    
    analyze_results_file(args.csv_path, args.model_name, args.task_name)