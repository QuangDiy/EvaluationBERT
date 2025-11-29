import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import math

def calculate_pseudo_perplexity(model, tokenizer, dataset, num_samples=100, max_seq_len=None):
    """
    Calculates pseudo-perplexity for a dataset.
    
    For each sequence, we compute pseudo-perplexity by randomly sampling 100 token
    positions with replacement, computing the masked language modeling (MLM) loss 
    at each position, and averaging the results.
    The pseudo-perplexity is defined as P = exp(1/n * sum(li)).
    """
    model.eval()
    # device is handled by device_map="auto" in main, or we check model.device
    device = model.device
    
    total_log_perplexity = 0
    count = 0
    
    print(f"Starting evaluation on {device}...")
    
    # Iterate over the dataset
    # Assuming dataset yields items with 'input_ids'
    for i, example in tqdm(enumerate(dataset), desc="Evaluating sequences"):
        input_ids = example['input_ids']
        
        # Convert to tensor if not already
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids)
        
        # Ensure 1D tensor
        if input_ids.dim() > 1:
            input_ids = input_ids.squeeze()
            
        seq_len = len(input_ids)
        
        # Skip very short sequences if necessary
        if seq_len == 0:
            continue
            
        # Sample 100 positions with replacement
        indices_to_mask = np.random.choice(seq_len, num_samples, replace=True)
        
        # Batching
        # Create 100 copies of input_ids
        batch_input_ids = input_ids.repeat(num_samples, 1) # (100, seq_len)
        batch_labels = torch.full(batch_input_ids.shape, -100) # (100, seq_len)
        
        # Apply masking
        for batch_idx, token_idx in enumerate(indices_to_mask):
            original_token = batch_input_ids[batch_idx, token_idx].item()
            batch_input_ids[batch_idx, token_idx] = tokenizer.mask_token_id
            batch_labels[batch_idx, token_idx] = original_token
            
        # Move to device
        batch_input_ids = batch_input_ids.to(device)
        batch_labels = batch_labels.to(device)
        
        # Forward pass
        sample_batch_size = 16
        seq_loss_sum = 0
        
        with torch.no_grad():
            for j in range(0, num_samples, sample_batch_size):
                mini_input = batch_input_ids[j : j + sample_batch_size]
                mini_labels = batch_labels[j : j + sample_batch_size]
                
                outputs = model(input_ids=mini_input, labels=mini_labels)
                
                current_batch_size = mini_input.size(0)
                seq_loss_sum += outputs.loss.item() * current_batch_size
        
        # Average loss for this sequence
        avg_seq_loss = seq_loss_sum / num_samples
        
        # Pseudo-perplexity for this sequence
        ppl = math.exp(avg_seq_loss)
        
        total_log_perplexity += avg_seq_loss # Accumulate average loss?
        # Wait, previous logic was accumulating PPL?
        # "The pseudo-perplexity is defined as P = exp(1/n * sum(li))"
        # This is per sequence. 
        # I will return the average PPL across sequences.
        
        # Let's accumulate PPL
        # Note: variable name total_log_perplexity is misleading if I accumulate PPL.
        # I'll fix the variable usage in the loop below.
        
        count += 1

    return 0 # Unused, main loop handles logic

def main():
    model_name = "QuangDuy/modernbert-tiny-checkpoint-55000ba"
    dataset_name = "QuangDuy/FineWiki-mds-tokenized-samples"
    
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Use device_map="auto" to handle device placement and avoid meta tensor issues
    try:
        # Try loading with device_map="auto" (requires accelerate)
        model = AutoModelForMaskedLM.from_pretrained(model_name, device_map="auto")
    except Exception as e:
        print(f"Failed to load with device_map='auto': {e}")
        print("Falling back to standard load (low_cpu_mem_usage=False)...")
        # Explicitly disable low_cpu_mem_usage and set device_map=None to avoid meta device
        model = AutoModelForMaskedLM.from_pretrained(
            model_name, 
            low_cpu_mem_usage=False,
            device_map=None, 
            torch_dtype=torch.float32
        )
        print(f"Model loaded on: {model.device}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
    
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="train")
    
    print("Starting calculation...")
    
    total_ppl = 0
    num_sequences = 0
    
    model.eval()
    device = model.device # Get the device the model is on
    
    num_samples = 100
    sample_batch_size = 32
    
    for i, example in tqdm(enumerate(dataset), total=len(dataset)):
        input_ids = example['input_ids']
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids)
        if input_ids.dim() > 1:
            input_ids = input_ids.squeeze()
            
        seq_len = len(input_ids)
        if seq_len == 0:
            continue
            
        # Sample 100 positions
        indices_to_mask = np.random.choice(seq_len, num_samples, replace=True)
        
        # Batching
        batch_input_ids = input_ids.repeat(num_samples, 1)
        batch_labels = torch.full(batch_input_ids.shape, -100)
        
        for batch_idx, token_idx in enumerate(indices_to_mask):
            original_token = batch_input_ids[batch_idx, token_idx].item()
            batch_input_ids[batch_idx, token_idx] = tokenizer.mask_token_id
            batch_labels[batch_idx, token_idx] = original_token
            
        batch_input_ids = batch_input_ids.to(device)
        batch_labels = batch_labels.to(device)
        
        seq_loss_sum = 0
        
        with torch.no_grad():
            for j in range(0, num_samples, sample_batch_size):
                mini_input = batch_input_ids[j : j + sample_batch_size]
                mini_labels = batch_labels[j : j + sample_batch_size]
                outputs = model(input_ids=mini_input, labels=mini_labels)
                seq_loss_sum += outputs.loss.item() * mini_input.size(0)
        
        avg_seq_loss = seq_loss_sum / num_samples
        ppl = math.exp(avg_seq_loss)
        
        total_ppl += ppl
        num_sequences += 1
        
    avg_ppl = total_ppl / num_sequences if num_sequences > 0 else 0.0
    print(f"\nAverage Pseudo-Perplexity: {avg_ppl:.4f}")

if __name__ == "__main__":
    main()
