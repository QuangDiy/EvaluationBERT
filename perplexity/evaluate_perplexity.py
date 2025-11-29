import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoConfig
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import math
from streaming import StreamingDataset
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import os
import json

def calculate_pseudo_perplexity(model, tokenizer, dataset, num_samples=100, max_seq_len=None):
    model.eval()
    device = model.device
    
    total_log_perplexity = 0
    count = 0
    
    print(f"Starting evaluation on {device}...")
    
    for i, example in tqdm(enumerate(dataset), desc="Evaluating sequences"):
        input_ids = example['input_ids']
        
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids)
        
        if input_ids.dim() > 1:
            input_ids = input_ids.squeeze()
            
        seq_len = len(input_ids)
        
        if seq_len == 0:
            continue
            
        indices_to_mask = np.random.choice(seq_len, num_samples, replace=True)
        
        batch_input_ids = input_ids.repeat(num_samples, 1)
        batch_labels = torch.full(batch_input_ids.shape, -100)
        
        for batch_idx, token_idx in enumerate(indices_to_mask):
            original_token = batch_input_ids[batch_idx, token_idx].item()
            batch_input_ids[batch_idx, token_idx] = tokenizer.mask_token_id
            batch_labels[batch_idx, token_idx] = original_token
            
        batch_input_ids = batch_input_ids.to(device)
        batch_labels = batch_labels.to(device)
        
        sample_batch_size = 16
        seq_loss_sum = 0
        
        with torch.no_grad():
            for j in range(0, num_samples, sample_batch_size):
                mini_input = batch_input_ids[j : j + sample_batch_size]
                mini_labels = batch_labels[j : j + sample_batch_size]
                
                outputs = model(input_ids=mini_input, labels=mini_labels)
                
                current_batch_size = mini_input.size(0)
                seq_loss_sum += outputs.loss.item() * current_batch_size
        
        avg_seq_loss = seq_loss_sum / num_samples
        ppl = math.exp(avg_seq_loss)
        
        total_log_perplexity += avg_seq_loss
        count += 1

    return 0

def main():
    model_name = "QuangDuy/modernbert-tiny-checkpoint-55000ba"
    dataset_name = "QuangDuy/FineWiki-mds-tokenized-samples"
    
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target device: {device}")
    
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_config(config)
    model = model.to(device)
    
    weights_file = hf_hub_download(repo_id=model_name, filename="model.safetensors")
    checkpoint_state_dict = load_file(weights_file)
    
    model_state_dict = model.state_dict()
    checkpoint_keys = set(checkpoint_state_dict.keys())
    model_keys = set(model_state_dict.keys())
    
    new_state_dict = {}
    
    for model_key in model_keys:
        if model_key in checkpoint_keys:
            new_state_dict[model_key] = checkpoint_state_dict[model_key]
        else:
            possible_mappings = [
                model_key.replace('model.', ''),
                f'model.{model_key}',
                model_key.replace('embeddings.tok_embeddings', 'embeddings.word_embeddings'),
                model_key.replace('embeddings.word_embeddings', 'embeddings.tok_embeddings'),
                model_key.replace('bert.', 'model.'),
                model_key.replace('model.', 'bert.'),
            ]
            
            found = False
            for possible_key in possible_mappings:
                if possible_key in checkpoint_keys:
                    new_state_dict[model_key] = checkpoint_state_dict[possible_key]
                    found = True
                    break
            
            if not found:
                new_state_dict[model_key] = model_state_dict[model_key]
    
    model.load_state_dict(new_state_dict, strict=False)
    
    if hasattr(model, 'tie_weights'):
        model.tie_weights()
    print(f"Loading dataset: {dataset_name}")
    
    try:
        
        print("Attempting to load as MDS format (MosaicML streaming dataset)...")
        
        print("Downloading MDS index and shard files...")
        local_cache_dir = "./mds_cache"
        os.makedirs(local_cache_dir, exist_ok=True)
        
        index_file = hf_hub_download(
            repo_id=dataset_name,
            filename="index.json",
            repo_type="dataset",
            local_dir=local_cache_dir
        )
        
        shard_file = hf_hub_download(
            repo_id=dataset_name,
            filename="shard.00000.mds.zstd",
            repo_type="dataset",
            local_dir=local_cache_dir
        )
        
        print(f"Downloaded MDS files to {local_cache_dir}")
        
        print("Loading dataset from local MDS files...")
        dataset = StreamingDataset(
            local=local_cache_dir,
            shuffle=False,
            batch_size=1
        )
        print(f"Loaded MDS dataset with {len(dataset)} samples")
        is_mds_format = True
        
    except ImportError:
        print("streaming library not found. Trying standard datasets library...")
        is_mds_format = False
        dataset = load_dataset(dataset_name, split="train")
        
    except Exception as e:
        print(f"Failed to load as MDS format: {e}")
        print("Falling back to standard datasets library...")
        is_mds_format = False
        dataset = load_dataset(dataset_name, split="train")
    
    print(f"Dataset size: {len(dataset)}")
    needs_tokenization = False
    
    if len(dataset) > 0:
        first_example = dataset[0]
        print(f"First example keys: {list(first_example.keys())}")
        
        if 'input_ids' in first_example:
            input_key = 'input_ids'
        elif 'tokens' in first_example:
            input_key = 'tokens'
        elif 'token_ids' in first_example:
            input_key = 'token_ids'
        elif 'text' in first_example:
            print("Dataset contains 'text' field. Tokenizing on-the-fly...")
            input_key = 'text'
            needs_tokenization = True
        else:
            raise ValueError(f"Could not find input_ids field. Available keys: {list(first_example.keys())}")
    else:
        raise ValueError("Dataset is empty!")
    
    print(f"Using field '{input_key}' for input data")
    
    print("Starting calculation...")
    
    total_ppl = 0
    num_sequences = 0
    
    results_data = []
    
    model.eval()
    device = model.device
    
    num_samples = 100  
    sample_batch_size = 8  
    
    for i, example in tqdm(enumerate(dataset), total=len(dataset)):
        if needs_tokenization:
            text = example[input_key]
            encoded = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
            input_ids = encoded['input_ids'].squeeze()
        else:
            input_ids = example[input_key]
            
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids)
        
        if input_ids.dtype != torch.int64:
            input_ids = input_ids.long()
            
        if input_ids.dim() > 1:
            input_ids = input_ids.squeeze()
            
        seq_len = len(input_ids)
        if seq_len == 0:
            continue
            
        indices_to_mask = np.random.choice(seq_len, num_samples, replace=True)
        
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
        
        results_data.append({
            "sequence_index": i,
            "sequence_length": seq_len,
            "pseudo_perplexity": float(ppl),
            "average_loss": float(avg_seq_loss)
        })
        
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
    avg_ppl = total_ppl / num_sequences if num_sequences > 0 else 0.0
    print(f"\nAverage Pseudo-Perplexity: {avg_ppl:.4f}")
    
    seq_lengths = [item["sequence_length"] for item in results_data]
    ppl_values = [item["pseudo_perplexity"] for item in results_data]
    
    results = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "evaluation_config": {
            "num_samples_per_sequence": num_samples,
            "sample_batch_size": sample_batch_size,
            "total_sequences_evaluated": num_sequences
        },
        "summary_statistics": {
            "average_pseudo_perplexity": float(avg_ppl),
            "min_perplexity": float(min(ppl_values)) if ppl_values else 0.0,
            "max_perplexity": float(max(ppl_values)) if ppl_values else 0.0,
            "median_perplexity": float(np.median(ppl_values)) if ppl_values else 0.0,
            "std_perplexity": float(np.std(ppl_values)) if ppl_values else 0.0,
            "min_sequence_length": int(min(seq_lengths)) if seq_lengths else 0,
            "max_sequence_length": int(max(seq_lengths)) if seq_lengths else 0,
            "mean_sequence_length": float(np.mean(seq_lengths)) if seq_lengths else 0.0,
            "median_sequence_length": float(np.median(seq_lengths)) if seq_lengths else 0.0
        },
        "per_sequence_results": results_data
    }
    
    output_filename = "perplexity_evaluation_results.json"
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to '{output_filename}'")
    print(f"\nSummary Statistics:")
    print(f"  Total sequences evaluated: {num_sequences}")
    print(f"  Average perplexity: {avg_ppl:.4f}")
    print(f"  Min perplexity: {min(ppl_values):.4f}")
    print(f"  Max perplexity: {max(ppl_values):.4f}")
    print(f"  Median perplexity: {np.median(ppl_values):.4f}")
    print(f"  Std perplexity: {np.std(ppl_values):.4f}")
    print(f"  Sequence length range: {min(seq_lengths)} - {max(seq_lengths)} tokens")
    print(f"  Mean sequence length: {np.mean(seq_lengths):.2f} tokens")

if __name__ == "__main__":
    main()
