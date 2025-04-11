import os
import torch
import traceback
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, HfArgumentParser
from dataclasses import dataclass, field
from typing import Optional, List
import sys

@dataclass
class ModelArguments:
########################################################################################################
    model_name_or_path: Optional[str] = field(default="neuralbioinfo/prokbert-mini-long")
########################################################################################################

@dataclass
class DataArguments:
    contigs_path: str = field(default="/work/sgk270/dnabert2_task2_new/contigs")
    full_seqs_path: str = field(default="/work/sgk270/dnabert2_task2_new/full_seqs")
    contig_lengths: List[int] = field(default_factory=lambda: [500, 1000, 3000, 5000, 10000])

@dataclass
class PrecomputeArguments:
    batch_size: int = field(default=32)
    output_dir: str = field(default="/work/sgk270/prokbert_task2_new/contig_embed")
    full_output_dir: str = field(default="/work/sgk270/prokbert_task2_new/full_embed")

class SequenceDataset(Dataset):
    def __init__(self, sequences, sequence_ids):
        self.sequences = sequences
        self.sequence_ids = sequence_ids

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.sequence_ids[idx]

def load_dataset(directory: str, is_full_sequence: bool):
    file_prefix = "full_sequences" if is_full_sequence else "contigs"
    sequences_path = os.path.join(directory, f"{file_prefix}.npy")
    sequence_ids_path = os.path.join(directory, "sequence_ids.npy")

    if not os.path.exists(sequences_path) or not os.path.exists(sequence_ids_path):
        raise FileNotFoundError(f"Missing {file_prefix}.npy or sequence_ids.npy in {directory}")

    sequences = np.load(sequences_path, allow_pickle=True)
    sequence_ids = np.load(sequence_ids_path, allow_pickle=True)
    print(f"Loaded {len(sequences)} from {directory}")
    return sequences, sequence_ids

def extract_embeddings(model, dataset, tokenizer, output_dir, split_name, batch_size, device):
    model.eval()
    all_embeddings = []
    all_sequence_ids = []

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for sequences, sequence_ids in dataloader:
            sequences = [str(s).strip() for s in sequences]
            all_sequence_ids.extend(sequence_ids)

##################################################################################################################
            inputs = tokenizer(sequences, return_tensors='pt', padding=True, truncation=True, max_length=512)
##################################################################################################################
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            if hasattr(outputs, 'last_hidden_state'):
                embeddings = outputs.last_hidden_state[:, 0, :]
            elif hasattr(outputs, 'hidden_states'):
                embeddings = outputs.hidden_states[-1][:, 0, :]
            else:
                raise AttributeError("Cannot find CLS embedding in outputs.")

            all_embeddings.append(embeddings.cpu().numpy())

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_sequence_ids = np.array(all_sequence_ids)

    os.makedirs(output_dir, exist_ok=True)
    np.savez(
        os.path.join(output_dir, f"{split_name}_embeddings.npz"),
        embeddings=all_embeddings,
        sequence_ids=all_sequence_ids
    )
    print(f"Saved {split_name} embeddings: {all_embeddings.shape}")

def process_split(split_name, base_path, output_dir, is_full_sequence, model, tokenizer, batch_size, device):
    try:
        sequences, sequence_ids = load_dataset(base_path, is_full_sequence)
        dataset = SequenceDataset(sequences, sequence_ids)
        extract_embeddings(model, dataset, tokenizer, output_dir, split_name, batch_size, device)
    except Exception as e:
        print(f"Error processing {split_name} at {base_path}: {e}")
        traceback.print_exc()

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, PrecomputeArguments))
    model_args, data_args, precompute_args = parser.parse_args_into_dataclasses()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_args.model_name_or_path, trust_remote_code=True, output_hidden_states=True).to(device)

    # Create output directories
    os.makedirs(precompute_args.output_dir, exist_ok=True)
    os.makedirs(precompute_args.full_output_dir, exist_ok=True)
    for length in data_args.contig_lengths:
        os.makedirs(os.path.join(precompute_args.output_dir, f"length_{length}"), exist_ok=True)

    # Process train/val combined contigs
    for split in ['train', 'val']:
        contig_path = os.path.join(data_args.contigs_path, split)
        process_split(split, contig_path, precompute_args.output_dir, is_full_sequence=False,
                      model=model, tokenizer=tokenizer, batch_size=precompute_args.batch_size, device=device)

    # Process test contigs by length
    for length in data_args.contig_lengths:
        length_dir = os.path.join(data_args.contigs_path, "test", f"length_{length}")
        output_dir = os.path.join(precompute_args.output_dir, f"length_{length}")
        process_split("test", length_dir, output_dir, is_full_sequence=False,
                      model=model, tokenizer=tokenizer, batch_size=precompute_args.batch_size, device=device)

    # Process full sequences
    for split in ['train', 'val', 'test']:
        full_seq_path = os.path.join(data_args.full_seqs_path, split)
        process_split(split, full_seq_path, precompute_args.full_output_dir, is_full_sequence=True,
                      model=model, tokenizer=tokenizer, batch_size=precompute_args.batch_size, device=device)

    print("All embedding tasks complete.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Unhandled exception in main:")
        traceback.print_exc()
        sys.exit(1)
