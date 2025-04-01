#!/usr/bin/env python3
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM, HfArgumentParser
from dataclasses import dataclass, field
from typing import Optional, List
import os
import numpy as np
import psutil
import traceback
import sys

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="zhihan1996/DNABERT-2-117M")

@dataclass
class DataArguments:
    pos_contigs_path: str = field(default="/work/sgk270/dnabert2_task1/pos_contigs")
    neg_contigs_path: str = field(default="/work/sgk270/dnabert2_task1/neg_contigs")
    pos_full_seqs_path: str = field(default="/work/sgk270/dnabert2_task1/pos_full_seqs")
    neg_full_seqs_path: str = field(default="/work/sgk270/dnabert2_task1/neg_full_seqs")
    contig_lengths: List[int] = field(default_factory=lambda: [500, 1000, 3000, 5000, 10000])

@dataclass
class EmbeddingArguments:
    batch_size: int = field(default=32)
    pos_contigs_embed_dir: str = field(default="pos_contigs_embed")
    neg_contigs_embed_dir: str = field(default="neg_contigs_embed")
    pos_full_embed_dir: str = field(default="pos_full_embed")
    neg_full_embed_dir: str = field(default="neg_full_embed")

def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB", flush=True)

class SequenceDataset(Dataset):
    def __init__(self, sequences, label):
        self.sequences = sequences
        self.label = label

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.label

def load_sequences(seq_path):
    try:
        sequences = np.load(seq_path, allow_pickle=True)
        print(f"Loaded {len(sequences)} sequences from {seq_path}")
        return sequences
    except Exception as e:
        print(f"Error loading {seq_path}: {e}")
        return np.array([])

def extract_embeddings(base_model, dataset, tokenizer, output_path, split_name, batch_size, device):
    base_model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for sequences, labels in dataloader:
            sequences = [str(s).strip() for s in sequences]
            all_labels.extend(labels)

            inputs = tokenizer(sequences, return_tensors='pt', padding=True, truncation=True, max_length=2048)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = base_model(**inputs)
            cls_embeddings = outputs[0][:, 0, :]
            all_embeddings.append(cls_embeddings.cpu().numpy())

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_labels = np.array(all_labels)

    os.makedirs(output_path, exist_ok=True)
    np.savez(os.path.join(output_path, f"{split_name}_embeddings.npz"),
             embeddings=all_embeddings, labels=all_labels)
    print(f"Saved {split_name} embeddings to {output_path}")

def process_split(base_model, tokenizer, data_dir, output_dir, label, split, tokenizer_args, device, is_full_seq, length=None):
    if length:
        data_dir = os.path.join(data_dir, f"length_{length}")
        output_dir = os.path.join(output_dir, f"length_{length}")
        split = f"test_length_{length}"

    input_file = os.path.join(data_dir, "full_sequences.npy" if is_full_seq else "contigs.npy")
    if not os.path.exists(input_file):
        print(f"Skipping {split} as input file not found: {input_file}")
        return

    sequences = load_sequences(input_file)
    dataset = SequenceDataset(sequences, label)
    extract_embeddings(base_model, dataset, tokenizer, output_dir, split, tokenizer_args.batch_size, device)

def process_all(base_model, tokenizer, data_args, embed_args, device):
    configs = [
        (data_args.pos_contigs_path, embed_args.pos_contigs_embed_dir, 1, False),
        (data_args.neg_contigs_path, embed_args.neg_contigs_embed_dir, 0, False),
        (data_args.pos_full_seqs_path, embed_args.pos_full_embed_dir, 1, True),
        (data_args.neg_full_seqs_path, embed_args.neg_full_embed_dir, 0, True),
    ]

    for data_dir, output_dir, label, is_full_seq in configs:
        print(f"\nProcessing {'positive' if label == 1 else 'negative'} {'full sequences' if is_full_seq else 'contigs'}")

        # Handle train, val, test
        for split in ['train', 'val', 'test']:
            split_path = os.path.join(data_dir, split)
            if not os.path.exists(split_path):
                print(f"Skipping split {split} at {split_path}")
                continue
            process_split(base_model, tokenizer, split_path, output_dir, label, split, embed_args, device, is_full_seq)

        # Handle test contig sub-lengths
        if not is_full_seq and os.path.exists(os.path.join(data_dir, "test")):
            for length in data_args.contig_lengths:
                process_split(
                    base_model,
                    tokenizer,
                    os.path.join(data_dir, "test"),
                    output_dir,
                    label,
                    "test",
                    embed_args,
                    device,
                    is_full_seq,
                    length=length
                )

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, EmbeddingArguments))
    model_args, data_args, embed_args = parser.parse_args_into_dataclasses()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Make output dirs
    for path in [embed_args.pos_contigs_embed_dir, embed_args.neg_contigs_embed_dir,
                 embed_args.pos_full_embed_dir, embed_args.neg_full_embed_dir]:
        os.makedirs(path, exist_ok=True)

    for length in data_args.contig_lengths:
        os.makedirs(os.path.join(embed_args.pos_contigs_embed_dir, f"length_{length}"), exist_ok=True)
        os.makedirs(os.path.join(embed_args.neg_contigs_embed_dir, f"length_{length}"), exist_ok=True)

    # Save label info
    label_info = {'mapping': {1: 'positive', 0: 'negative'}}
    for path in [embed_args.pos_contigs_embed_dir, embed_args.neg_contigs_embed_dir,
                 embed_args.pos_full_embed_dir, embed_args.neg_full_embed_dir]:
        np.save(os.path.join(path, 'label_info.npy'), label_info)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    base_model = model.bert.to(device)

    print_memory_usage()
    process_all(base_model, tokenizer, data_args, embed_args, device)

    print("\nEmbedding extraction complete.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\nError during execution:", flush=True)
        print(traceback.format_exc(), flush=True)
        sys.exit(1)
