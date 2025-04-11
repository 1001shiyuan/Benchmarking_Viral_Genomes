import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, HfArgumentParser
from dataclasses import dataclass, field
from typing import Optional, List
import os
import numpy as np
import datetime
import psutil
import multiprocessing
import traceback
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "true"
if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

@dataclass
class ModelArguments:
##################################################################################################################
    model_name_or_path: Optional[str] = field(default="neuralbioinfo/prokbert-mini")
##################################################################################################################

@dataclass
class DataArguments:
    contigs_path: str = field(default="/work/sgk270/dnabert2_task3_new/contigs")
    full_seqs_path: str = field(default="/work/sgk270/dnabert2_task3_new/full_seqs")
    contig_lengths: List[int] = field(default_factory=lambda: [500, 1000, 3000, 5000, 10000])

@dataclass
class PrecomputeArguments:
    batch_size: int = field(default=32)
    num_workers: int = field(default=4)
    output_dir: str = field(default="/work/sgk270/prokbert_task3_new/contig_embed")
    full_output_dir: str = field(default="/work/sgk270/prokbert_task3_new/full_embed")

def log_with_time(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)

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

def extract_embeddings(model, dataset, tokenizer, output_dir, split_name, batch_size, num_workers, device, stats):
    log_with_time(f"Extracting embeddings for {split_name} ({len(dataset)} sequences)...")
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    all_embeddings = []
    all_sequence_ids = []

    with torch.no_grad():
        for sequences, sequence_ids, _ in dataloader:
            sequences = [str(s).strip() for s in sequences]
            all_sequence_ids.extend(sequence_ids)
            tokens_before = [tokenizer.tokenize(seq) for seq in sequences]
            for tokens in tokens_before:
                stats['total'] += 1
##################################################################################################################
                if len(tokens) > 512:
                    stats['truncated'] += 1
            inputs = tokenizer(sequences, return_tensors='pt', padding=True, truncation=True, max_length=512)
##################################################################################################################
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            if hasattr(outputs, "last_hidden_state"):
                embeddings = outputs.last_hidden_state[:, 0, :]
            elif hasattr(outputs, "hidden_states"):
                embeddings = outputs.hidden_states[-1][:, 0, :]
            else:
                raise ValueError("Model output missing last_hidden_state or hidden_states")
            all_embeddings.append(embeddings.cpu().numpy())

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{split_name}_embeddings.npz")
    np.savez(out_path, embeddings=np.concatenate(all_embeddings, axis=0), sequence_ids=np.array(all_sequence_ids))
    log_with_time(f"Saved {split_name} embeddings to {out_path}")

def process_train_val_contigs(model, tokenizer, data_args, output_dir, precompute_args, device, stats):
    log_with_time("Processing train/val contigs...")
    for split in ['train', 'val']:
        split_path = os.path.join(data_args.contigs_path, split)
        if os.path.exists(split_path):
            sequences, sequence_ids = load_dataset(split_path, split, False)
            dataset = SequenceDataset(sequences, sequence_ids, f"{split}_combined")
            extract_embeddings(model, dataset, tokenizer, output_dir, split, precompute_args.batch_size, precompute_args.num_workers, device, stats)

def process_test_contigs(model, tokenizer, data_args, output_dir, precompute_args, device, stats):
    log_with_time("Processing test contigs by length...")
    test_base_path = os.path.join(data_args.contigs_path, "test")
    if not os.path.exists(test_base_path):
        return
    for contig_length in data_args.contig_lengths:
        length_dir = os.path.join(test_base_path, f"length_{contig_length}")
        if os.path.exists(length_dir):
            sequences, sequence_ids = load_dataset(length_dir, f"test_{contig_length}", False)
            dataset = SequenceDataset(sequences, sequence_ids, f"test_length_{contig_length}")
            extract_embeddings(model, dataset, tokenizer, os.path.join(output_dir, f"length_{contig_length}"), "test", precompute_args.batch_size, precompute_args.num_workers, device, stats)

def process_full_sequences(model, tokenizer, data_path, output_dir, precompute_args, device, stats):
    log_with_time("Processing full sequences...")
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(data_path, split)
        if os.path.exists(split_path):
            sequences, sequence_ids = load_dataset(split_path, split, True)
            dataset = SequenceDataset(sequences, sequence_ids, 'full_sequence')
            extract_embeddings(model, dataset, tokenizer, output_dir, split, precompute_args.batch_size, precompute_args.num_workers, device, stats)

def main():
    log_with_time("Starting ProkBERT embedding extraction for lifestyle classification (Task 3)")
    parser = HfArgumentParser((ModelArguments, DataArguments, PrecomputeArguments))
    model_args, data_args, precompute_args = parser.parse_args_into_dataclasses()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_with_time(f"Using device: {device}")

    os.makedirs(precompute_args.output_dir, exist_ok=True)
    os.makedirs(precompute_args.full_output_dir, exist_ok=True)
    for length in data_args.contig_lengths:
        os.makedirs(os.path.join(precompute_args.output_dir, f"length_{length}"), exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_args.model_name_or_path, trust_remote_code=True, output_hidden_states=True).to(device)

    contig_stats = {'total': 0, 'truncated': 0}
    full_stats = {'total': 0, 'truncated': 0}

    process_train_val_contigs(model, tokenizer, data_args, precompute_args.output_dir, precompute_args, device, contig_stats)
    process_test_contigs(model, tokenizer, data_args, precompute_args.output_dir, precompute_args, device, contig_stats)
    process_full_sequences(model, tokenizer, data_args.full_seqs_path, precompute_args.full_output_dir, precompute_args, device, full_stats)

    log_with_time("Truncation statistics:")
    log_with_time(f"Contigs: {contig_stats['truncated']} of {contig_stats['total']} truncated")
    log_with_time(f"Full sequences: {full_stats['truncated']} of {full_stats['total']} truncated")
    log_with_time("Embedding extraction completed.")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        log_with_time("Error during embedding extraction")
        print(traceback.format_exc(), flush=True)
        sys.exit(1)
