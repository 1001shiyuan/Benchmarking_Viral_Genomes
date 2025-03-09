import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM, HfArgumentParser
from dataclasses import dataclass, field
from typing import Optional, List
import os
import numpy as np
import pandas as pd
import psutil
import sys
import traceback
from sklearn.preprocessing import LabelEncoder

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="zhihan1996/DNABERT-2-117M")

@dataclass
class DataArguments:
    # Updated paths to use lifestyle data
    contigs_path: str = field(default="/work/sgk270/dnabert2_task3/lifestyle_contigs")
    full_seqs_path: str = field(default="/work/sgk270/dnabert2_task3/lifestyle_full_seqs")
    contig_lengths: List[int] = field(default_factory=lambda: [500, 1000, 3000, 5000, 10000])

@dataclass
class PrecomputeArguments:
    batch_size: int = field(default=32)
    # Updated output directories for lifestyle results
    output_dir: str = field(default="lifestyle_contig_embed")
    full_output_dir: str = field(default="lifestyle_full_embed")

def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB", flush=True)

class SequenceDataset(Dataset):
    def __init__(self, sequences, sequence_ids, labels_dict, length_group):
        self.sequences = sequences
        self.sequence_ids = sequence_ids
        self.labels_dict = labels_dict  # Dictionary containing lifestyle labels
        self.length_group = length_group

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        labels = {
            'lifestyle': self.labels_dict['lifestyle'][idx]
        }
        return self.sequences[idx], self.sequence_ids[idx], labels, self.length_group

def load_metadata_and_prepare_labels():
    """Load lifestyle metadata and prepare label encoders."""
    # Updated to use lifestyle metadata
    metadata = pd.read_csv("/work/sgk270/dataset_for_benchmarking/lifestyle/combined_metadata.tsv", sep='\t')
    
    # For lifestyle data, we'll use the 'lifestyle' column which contains categories like 'lytic', 'lysogenic', etc.
    metadata = metadata.dropna(subset=['lifestyle'])
    
    # Create label encoder for lifestyle
    label_encoders = {
        'lifestyle': LabelEncoder()
    }
    
    # Fit label encoder and create dictionary
    label_dicts = {}
    label_encoders['lifestyle'].fit(metadata['lifestyle'].unique())
    label_dicts['lifestyle'] = dict(zip(metadata['sequence_id'], metadata['lifestyle']))
    
    return metadata, label_dicts, label_encoders

def load_dataset(directory: str, length_group: str, is_full_sequence: bool):
    """Load sequences and their IDs."""
    file_prefix = "full_sequences" if is_full_sequence else "contigs"
    sequences_path = os.path.join(directory, f"{file_prefix}.npy")
    sequence_ids_path = os.path.join(directory, "sequence_ids.npy")

    if not os.path.exists(sequences_path) or not os.path.exists(sequence_ids_path):
        raise FileNotFoundError(f"Missing {file_prefix}.npy or sequence_ids.npy in {directory}")

    sequences = np.load(sequences_path, allow_pickle=True)
    sequence_ids = np.load(sequence_ids_path, allow_pickle=True)
    print(f"Loaded {len(sequences)} sequences from {directory} (Group: {length_group})")

    return sequences, sequence_ids, [length_group] * len(sequences)

def extract_embeddings(base_model, dataset, tokenizer, output_dir, split_name, batch_size, device, stats, label_encoders):
    """Extract CLS embeddings and save with lifestyle labels."""
    base_model.eval()
    all_embeddings = []
    all_labels = {
        'lifestyle': []
    }
    all_sequence_ids = []

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for sequences, sequence_ids, labels_dict, _ in dataloader:
            sequences = [str(s).strip() for s in sequences]
            all_sequence_ids.extend(sequence_ids)
            
            # Count truncations before tokenization and track length statistics
            tokens_before = [tokenizer.tokenize(seq) for seq in sequences]
            for tokens in tokens_before:
                token_length = len(tokens)
                stats['total_samples'] += 1
                stats['total_length'] += token_length
                
                if token_length > 2048:
                    stats['truncated_samples'] += 1
                    stats['truncated_length'] += (token_length - 2048)

            inputs = tokenizer(sequences, return_tensors='pt', padding=True, truncation=True, max_length=2048)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = base_model(**inputs)
            last_hidden_state = outputs[0]
            embeddings = last_hidden_state[:, 0, :]

            all_embeddings.append(embeddings.cpu().numpy())
            all_labels['lifestyle'].extend(labels_dict['lifestyle'])

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_labels['lifestyle'] = np.array(all_labels['lifestyle'])
    all_sequence_ids = np.array(all_sequence_ids)

    # Convert numerical labels to text labels
    text_labels = {}
    text_labels['lifestyle_text'] = np.array([
        label_encoders['lifestyle'].classes_[idx] for idx in all_labels['lifestyle']
    ])

    os.makedirs(output_dir, exist_ok=True)
    np.savez(
        os.path.join(output_dir, f'{split_name}_embeddings.npz'),
        embeddings=all_embeddings,
        sequence_ids=all_sequence_ids,
        lifestyle_labels=all_labels['lifestyle'],
        lifestyle_text=text_labels['lifestyle_text']
    )
    print(f"Saved {split_name} embeddings with shape {all_embeddings.shape}")

def process_train_val_contigs(base_model, tokenizer, data_args, output_dir, label_dicts, label_encoders, precompute_args, device, stats):
    """Process train and val contigs that are combined across all lengths."""
    print("\nProcessing train and val contig embeddings...")
    
    # Process combined train and val contigs
    for split in ['train', 'val']:
        split_path = os.path.join(data_args.contigs_path, split)
        if not os.path.exists(split_path):
            print(f"Skipping {split} split as path does not exist: {split_path}")
            continue
            
        try:
            sequences, sequence_ids, _ = load_dataset(split_path, split, False)
            valid_indices = [i for i, seq_id in enumerate(sequence_ids) 
                           if seq_id in label_dicts['lifestyle']]

            # Prepare lifestyle labels
            labels_dict = {'lifestyle': []}
            labels = [label_dicts['lifestyle'][seq_id] for seq_id in sequence_ids[valid_indices]]
            labels_dict['lifestyle'] = label_encoders['lifestyle'].transform(labels)

            dataset = SequenceDataset(
                sequences[valid_indices],
                sequence_ids[valid_indices],
                labels_dict,
                f"{split}_combined"
            )

            extract_embeddings(
                base_model,
                dataset,
                tokenizer,
                output_dir,
                split,
                precompute_args.batch_size,
                device,
                stats,
                label_encoders
            )
        except FileNotFoundError as e:
            print(f"Warning: {e}. Skipping {split} split.")
        except Exception as e:
            print(f"Error processing {split} split: {e}")
            traceback.print_exc()

def process_test_contigs(base_model, tokenizer, data_args, output_dir, label_dicts, label_encoders, precompute_args, device, stats):
    """Process test contigs which are separated by length."""
    print("\nProcessing test contig embeddings by length...")
    
    # For test, process each length separately
    test_base_path = os.path.join(data_args.contigs_path, "test")
    if not os.path.exists(test_base_path):
        print(f"Skipping test as path does not exist: {test_base_path}")
        return
    
    for contig_length in data_args.contig_lengths:
        length_dir = os.path.join(test_base_path, f"length_{contig_length}")
        if not os.path.exists(length_dir):
            print(f"Skipping test for length {contig_length} as path does not exist: {length_dir}")
            continue
            
        # Get the output directory for this length (already created in main)
        length_output_dir = os.path.join(output_dir, f"length_{contig_length}")
            
        try:
            sequences, sequence_ids, _ = load_dataset(length_dir, f"test_{contig_length}", False)
            valid_indices = [i for i, seq_id in enumerate(sequence_ids) 
                           if seq_id in label_dicts['lifestyle']]

            # Prepare lifestyle labels
            labels_dict = {'lifestyle': []}
            labels = [label_dicts['lifestyle'][seq_id] for seq_id in sequence_ids[valid_indices]]
            labels_dict['lifestyle'] = label_encoders['lifestyle'].transform(labels)

            dataset = SequenceDataset(
                sequences[valid_indices],
                sequence_ids[valid_indices],
                labels_dict,
                f"test_length_{contig_length}"
            )

            extract_embeddings(
                base_model,
                dataset,
                tokenizer,
                length_output_dir,
                "test",
                precompute_args.batch_size,
                device,
                stats,
                label_encoders
            )
        except FileNotFoundError as e:
            print(f"Warning: {e}. Skipping test for length {contig_length}.")
        except Exception as e:
            print(f"Error processing test for length {contig_length}: {e}")
            traceback.print_exc()

def process_full_sequences(base_model, tokenizer, data_path, output_dir, label_dicts, label_encoders, precompute_args, device, stats):
    """Process full sequence data."""
    print("\nProcessing full sequence embeddings...")
    
    # Process train, val, and test splits
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(data_path, split)
        if not os.path.exists(split_path):
            print(f"Skipping {split} split as path does not exist: {split_path}")
            continue
            
        try:
            sequences, sequence_ids, _ = load_dataset(split_path, split, True)
            valid_indices = [i for i, seq_id in enumerate(sequence_ids) 
                           if seq_id in label_dicts['lifestyle']]

            # Prepare lifestyle labels
            labels_dict = {'lifestyle': []}
            labels = [label_dicts['lifestyle'][seq_id] for seq_id in sequence_ids[valid_indices]]
            labels_dict['lifestyle'] = label_encoders['lifestyle'].transform(labels)

            dataset = SequenceDataset(
                sequences[valid_indices],
                sequence_ids[valid_indices],
                labels_dict,
                'full_sequence'
            )

            extract_embeddings(
                base_model,
                dataset,
                tokenizer,
                output_dir,
                split,
                precompute_args.batch_size,
                device,
                stats,
                label_encoders
            )
        except FileNotFoundError as e:
            print(f"Warning: {e}. Skipping full sequence {split} split.")
        except Exception as e:
            print(f"Error processing full sequence {split} split: {e}")
            traceback.print_exc()

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, PrecomputeArguments))
    model_args, data_args, precompute_args = parser.parse_args_into_dataclasses()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    metadata, label_dicts, label_encoders = load_metadata_and_prepare_labels()
    print(f"Number of lifestyle categories: {len(label_encoders['lifestyle'].classes_)}")
    print(f"Lifestyle categories: {label_encoders['lifestyle'].classes_}")

    # Create output directories
    os.makedirs(precompute_args.output_dir, exist_ok=True)
    os.makedirs(precompute_args.full_output_dir, exist_ok=True)
    
    # Create length-specific output directories for test contigs
    for contig_length in data_args.contig_lengths:
        length_output_dir = os.path.join(precompute_args.output_dir, f"length_{contig_length}")
        os.makedirs(length_output_dir, exist_ok=True)

    # Save label encoders
    encoder_data = {
        'lifestyle': {'classes': label_encoders['lifestyle'].classes_}
    }
    
    # Save to the full sequence directory
    np.save(os.path.join(precompute_args.full_output_dir, 'label_encoders.npy'), encoder_data)
    
    # Save to the main contig directory and each length-specific directory
    np.save(os.path.join(precompute_args.output_dir, 'label_encoders.npy'), encoder_data)
    for contig_length in data_args.contig_lengths:
        length_output_dir = os.path.join(precompute_args.output_dir, f"length_{contig_length}")
        np.save(os.path.join(length_output_dir, 'label_encoders.npy'), encoder_data)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True
    )

    model = AutoModelForMaskedLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True
    )

    base_model = model.bert.to(device)

    # Initialize enhanced statistics counters
    contig_stats = {
        'total_samples': 0, 
        'truncated_samples': 0, 
        'total_length': 0, 
        'truncated_length': 0
    }
    
    full_seq_stats = {
        'total_samples': 0, 
        'truncated_samples': 0, 
        'total_length': 0, 
        'truncated_length': 0
    }

    # Process train and val contigs (combined across all lengths)
    process_train_val_contigs(
        base_model, 
        tokenizer, 
        data_args, 
        precompute_args.output_dir, 
        label_dicts, 
        label_encoders, 
        precompute_args, 
        device, 
        contig_stats
    )
    
    # Process test contigs (separate by length)
    process_test_contigs(
        base_model, 
        tokenizer, 
        data_args, 
        precompute_args.output_dir, 
        label_dicts, 
        label_encoders, 
        precompute_args, 
        device, 
        contig_stats
    )

    # Process full sequence embeddings
    process_full_sequences(
        base_model, 
        tokenizer, 
        data_args.full_seqs_path, 
        precompute_args.full_output_dir, 
        label_dicts, 
        label_encoders, 
        precompute_args, 
        device, 
        full_seq_stats
    )

    # Print enhanced truncation statistics
    print("\nTruncation Statistics:")
    
    if contig_stats['total_samples'] > 0:
        print("\nContigs:")
        print(f"  Samples: {contig_stats['truncated_samples']} out of {contig_stats['total_samples']} sequences truncated " + 
              f"({contig_stats['truncated_samples']/contig_stats['total_samples']*100:.2f}%)")
        
        if contig_stats['total_length'] > 0:
            print(f"  Length: {contig_stats['truncated_length']:,} out of {contig_stats['total_length']:,} tokens truncated " + 
                  f"({contig_stats['truncated_length']/contig_stats['total_length']*100:.2f}%)")
    
    if full_seq_stats['total_samples'] > 0:
        print("\nFull Sequences:")
        print(f"  Samples: {full_seq_stats['truncated_samples']} out of {full_seq_stats['total_samples']} sequences truncated " + 
              f"({full_seq_stats['truncated_samples']/full_seq_stats['total_samples']*100:.2f}%)")
        
        if full_seq_stats['total_length'] > 0:
            print(f"  Length: {full_seq_stats['truncated_length']:,} out of {full_seq_stats['total_length']:,} tokens truncated " + 
                  f"({full_seq_stats['truncated_length']/full_seq_stats['total_length']*100:.2f}%)")

    print("\nEmbedding extraction completed")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\nError in main execution:", flush=True)
        print(traceback.format_exc(), flush=True)
        sys.exit(1)