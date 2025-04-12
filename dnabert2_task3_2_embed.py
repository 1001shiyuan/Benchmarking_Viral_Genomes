import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM, HfArgumentParser
from dataclasses import dataclass, field
from typing import Optional, List
import os
import numpy as np
import pandas as pd
import psutil
import sys
import traceback
import datetime
import multiprocessing

os.environ["TOKENIZERS_PARALLELISM"] = "true"

if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

@dataclass
class ModelArguments:
##############################################################################################
    model_name_or_path: Optional[str] = field(default="zhihan1996/DNABERT-2-117M")
###############################################################################################

@dataclass
class DataArguments:
    contigs_path: str = field(default="/work/sgk270/dnabert2_task3_new/contigs")
    full_seqs_path: str = field(default="/work/sgk270/dnabert2_task3_new/full_seqs")
    contig_lengths: List[int] = field(default_factory=lambda: [500, 1000, 3000, 5000, 10000])

@dataclass
class PrecomputeArguments:
    batch_size: int = field(default=32)
    num_workers: int = field(default=4)
    output_dir: str = field(default="/work/sgk270/dnabert2_task3_new/contig_embed")
    full_output_dir: str = field(default="/work/sgk270/dnabert2_task3_new/full_embed")

def log_with_time(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)

def print_memory_usage():
    process = psutil.Process(os.getpid())
    log_with_time(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

class SequenceDataset(Dataset):
    def __init__(self, sequences, sequence_ids, length_group):
        self.sequences = sequences
        self.sequence_ids = sequence_ids
        self.length_group = length_group

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.sequence_ids[idx], self.length_group

def load_dataset(directory: str, length_group: str, is_full_sequence: bool):
    file_prefix = "full_sequences" if is_full_sequence else "contigs"
    sequences_path = os.path.join(directory, f"{file_prefix}.npy")
    sequence_ids_path = os.path.join(directory, "sequence_ids.npy")

    if not os.path.exists(sequences_path) or not os.path.exists(sequence_ids_path):
        raise FileNotFoundError(f"Missing {file_prefix}.npy or sequence_ids.npy in {directory}")

    log_with_time(f"Loading sequences from {directory}...")
    sequences = np.load(sequences_path, allow_pickle=True)
    sequence_ids = np.load(sequence_ids_path, allow_pickle=True)
    log_with_time(f"Loaded {len(sequences)} sequences from {directory} (Group: {length_group})")

    return sequences, sequence_ids

def extract_embeddings(base_model, dataset, tokenizer, output_dir, split_name, batch_size, num_workers, device, stats):
    log_with_time(f"Starting embedding extraction for {split_name} split with {len(dataset)} sequences...")
    base_model.eval()
    all_embeddings = []
    all_sequence_ids = []

    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    total_batches = len(dataloader)
    log_with_time(f"Created dataloader with {total_batches} batches")

    with torch.no_grad():
        for batch_idx, (sequences, sequence_ids, _) in enumerate(dataloader):
            if batch_idx % 10 == 0 or batch_idx == total_batches - 1:
                log_with_time(f"Processing batch {batch_idx+1}/{total_batches} ({(batch_idx+1)/total_batches*100:.1f}%)")
            
            sequences = [str(s).strip() for s in sequences]
            all_sequence_ids.extend(sequence_ids)
            
            tokens_before = [tokenizer.tokenize(seq) for seq in sequences]
            for tokens in tokens_before:
                stats['total'] += 1
################################################################################################################
                if len(tokens) > 204800:
                    stats['truncated'] += 1

            inputs = tokenizer(sequences, return_tensors='pt', padding=True, truncation=True, max_length=204800)
################################################################################################################
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = base_model(**inputs)
            last_hidden_state = outputs[0]
            embeddings = last_hidden_state[:, 0, :]

            all_embeddings.append(embeddings.cpu().numpy())

    log_with_time(f"Finished processing all batches for {split_name} split")
    
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_sequence_ids = np.array(all_sequence_ids)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{split_name}_embeddings.npz')
    log_with_time(f"Saving embeddings to {output_path}...")
    
    np.savez(output_path, embeddings=all_embeddings, sequence_ids=all_sequence_ids)
    log_with_time(f"Saved {split_name} embeddings with shape {all_embeddings.shape}")

def process_train_val_contigs(base_model, tokenizer, data_args, output_dir, precompute_args, device, stats):
    log_with_time("Processing train and val contig embeddings...")
    for split in ['train', 'val']:
        split_path = os.path.join(data_args.contigs_path, split)
        if not os.path.exists(split_path):
            log_with_time(f"Skipping {split} split as path does not exist: {split_path}")
            continue
            
        try:
            sequences, sequence_ids = load_dataset(split_path, split, False)
            dataset = SequenceDataset(sequences, sequence_ids, f"{split}_combined")
            extract_embeddings(base_model, dataset, tokenizer, output_dir, split, precompute_args.batch_size, precompute_args.num_workers, device, stats)
        except Exception as e:
            log_with_time(f"Error processing {split} split: {e}")
            traceback.print_exc()

def process_test_contigs(base_model, tokenizer, data_args, output_dir, precompute_args, device, stats):
    log_with_time("Processing test contig embeddings by length...")
    test_base_path = os.path.join(data_args.contigs_path, "test")
    if not os.path.exists(test_base_path):
        log_with_time(f"Skipping test as path does not exist: {test_base_path}")
        return
    
    for contig_length in data_args.contig_lengths:
        length_dir = os.path.join(test_base_path, f"length_{contig_length}")
        if not os.path.exists(length_dir):
            log_with_time(f"Skipping test for length {contig_length} as path does not exist: {length_dir}")
            continue
            
        length_output_dir = os.path.join(output_dir, f"length_{contig_length}")
        try:
            sequences, sequence_ids = load_dataset(length_dir, f"test_{contig_length}", False)
            dataset = SequenceDataset(sequences, sequence_ids, f"test_length_{contig_length}")
            extract_embeddings(base_model, dataset, tokenizer, length_output_dir, "test", precompute_args.batch_size, precompute_args.num_workers, device, stats)
        except Exception as e:
            log_with_time(f"Error processing test for length {contig_length}: {e}")
            traceback.print_exc()

def process_full_sequences(base_model, tokenizer, data_path, output_dir, precompute_args, device, stats):
    log_with_time("Processing full sequence embeddings...")
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(data_path, split)
        if not os.path.exists(split_path):
            log_with_time(f"Skipping {split} split as path does not exist: {split_path}")
            continue
            
        try:
            sequences, sequence_ids = load_dataset(split_path, split, True)
            dataset = SequenceDataset(sequences, sequence_ids, 'full_sequence')
            extract_embeddings(base_model, dataset, tokenizer, output_dir, split, precompute_args.batch_size, precompute_args.num_workers, device, stats)
        except Exception as e:
            log_with_time(f"Error processing full sequence {split} split: {e}")
            traceback.print_exc()

def main():
    log_with_time("Starting the embedding extraction process...")
    log_with_time(f"Using multiprocessing start method: {multiprocessing.get_start_method()}")
    
    parser = HfArgumentParser((ModelArguments, DataArguments, PrecomputeArguments))
    model_args, data_args, precompute_args = parser.parse_args_into_dataclasses()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_with_time(f"Using device: {device}")
    
    if torch.cuda.is_available():
        log_with_time(f"GPU: {torch.cuda.get_device_name(0)}")
        log_with_time(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    os.makedirs(precompute_args.output_dir, exist_ok=True)
    os.makedirs(precompute_args.full_output_dir, exist_ok=True)
    
    for contig_length in data_args.contig_lengths:
        os.makedirs(os.path.join(precompute_args.output_dir, f"length_{contig_length}"), exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    base_model = model.bert.to(device)

    contig_stats = {'total': 0, 'truncated': 0}
    full_seq_stats = {'total': 0, 'truncated': 0}

    process_train_val_contigs(base_model, tokenizer, data_args, precompute_args.output_dir, precompute_args, device, contig_stats)
    process_test_contigs(base_model, tokenizer, data_args, precompute_args.output_dir, precompute_args, device, contig_stats)
    process_full_sequences(base_model, tokenizer, data_args.full_seqs_path, precompute_args.full_output_dir, precompute_args, device, full_seq_stats)

    log_with_time("Truncation Statistics:")
    if contig_stats['total'] > 0:
        log_with_time(f"Contigs: {contig_stats['truncated']} out of {contig_stats['total']} truncated ({contig_stats['truncated']/contig_stats['total']*100:.2f}%)")
    if full_seq_stats['total'] > 0:
        log_with_time(f"Full Sequences: {full_seq_stats['truncated']} out of {full_seq_stats['total']} truncated ({full_seq_stats['truncated']/full_seq_stats['total']*100:.2f}%)")

    log_with_time("Embedding extraction completed")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_with_time("Error in main execution:")
        print(traceback.format_exc(), flush=True)
        sys.exit(1)
