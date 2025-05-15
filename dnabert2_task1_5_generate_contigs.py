#!/usr/bin/env python3
import os
import numpy as np
import random
import glob
import gzip
from multiprocessing import Pool, cpu_count
from typing import List, Tuple
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

def read_sequences(file_path: str) -> List[Tuple[str, str, int]]:
    sequences = []
    try:
        with gzip.open(file_path, 'rt') if file_path.endswith('.gz') else open(file_path, 'r') as handle:
            for record in SeqIO.parse(handle, "fasta"):
                sequences.append((record.id, str(record.seq), len(record.seq)))
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return sequences

def read_all_sequences(directory: str, pattern: str = "*.fna.gz") -> List[Tuple[str, str, int]]:
    all_files = glob.glob(os.path.join(directory, "**", pattern), recursive=True)
    print(f"Found {len(all_files)} files in {directory}")
    with Pool(cpu_count()) as pool:
        all_sequences = pool.map(read_sequences, all_files)
    sequences = [seq for batch in all_sequences for seq in batch]
    print(f"Loaded {len(sequences)} sequences from {directory}")
    return sequences

def split_sequences_by_length_ratio(sequences: List[Tuple[str, str, int]], ratios: List[float]):
    random.shuffle(sequences)
    total_length = sum(length for _, _, length in sequences)
    train_target = ratios[0] * total_length
    val_target = ratios[1] * total_length

    train, val, test = [], [], []
    current_length = 0
    for seq_id, seq, length in sequences:
        if current_length < train_target:
            train.append((seq_id, seq, length))
        elif current_length < train_target + val_target:
            val.append((seq_id, seq, length))
        else:
            test.append((seq_id, seq, length))
        current_length += length
    return train, val, test

def save_sequences(sequences: List[Tuple[str, str, int]], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    ids = [seq_id for seq_id, _, _ in sequences]
    seqs = [seq for _, seq, _ in sequences]
    np.save(os.path.join(out_dir, "full_sequences.npy"), np.array(seqs, dtype=object))
    np.save(os.path.join(out_dir, "sequence_ids.npy"), np.array(ids, dtype=object))
    print(f"Saved {len(sequences)} sequences to {out_dir}")

def generate_fixed_length_contigs(sequence: str, seq_id: str, length: int) -> List[Tuple[str, str, int]]:
    return [(seq_id, sequence[i:i+length], length)
            for i in range(0, len(sequence) - length + 1, length)]

def generate_contigs(sequences: List[Tuple[str, str, int]], length: int) -> List[Tuple[str, str, int]]:
    all_contigs = []
    for seq_id, seq, _ in sequences:
        all_contigs.extend(generate_fixed_length_contigs(seq, seq_id, length))
    return all_contigs

def save_contigs(contigs: List[Tuple[str, str, int]], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    contig_strs = [c for _, c, _ in contigs]
    contig_ids = [i for i, _, _ in contigs]
    np.save(os.path.join(out_dir, "contigs.npy"), np.array(contig_strs, dtype=object))
    np.save(os.path.join(out_dir, "sequence_ids.npy"), np.array(contig_ids, dtype=object))
    print(f"Saved {len(contigs)} contigs to {out_dir}")

def process_dataset(input_path: str, output_base: str, prefix: str):
    print(f"\nProcessing: {prefix.upper()} dataset")
    sequences = read_all_sequences(input_path)
    train, val, test = split_sequences_by_length_ratio(sequences, [0.8, 0.1, 0.1])

    # Save full sequences
    save_sequences(train, os.path.join(output_base, f"{prefix}_full_seqs/train"))
    save_sequences(val, os.path.join(output_base, f"{prefix}_full_seqs/val"))
    save_sequences(test, os.path.join(output_base, f"{prefix}_full_seqs/test"))

    # Contig lengths
    lengths = [500, 1000, 3000, 5000, 10000]
    all_train_contigs, all_val_contigs = [], []

    for length in lengths:
        print(f"\nGenerating contigs of length {length} for {prefix.upper()}")
        train_contigs = generate_contigs(train, length)
        val_contigs = generate_contigs(val, length)
        test_contigs = generate_contigs(test, length)

        all_train_contigs.extend(train_contigs)
        all_val_contigs.extend(val_contigs)

        test_out = os.path.join(output_base, f"{prefix}_contigs/test/length_{length}")
        save_contigs(test_contigs, test_out)

    save_contigs(all_train_contigs, os.path.join(output_base, f"{prefix}_contigs/train"))
    save_contigs(all_val_contigs, os.path.join(output_base, f"{prefix}_contigs/val"))

def main():
    random.seed(42)
    base_dir = "/work/sgk270/dnabert2_task1_new"
    pos_input = os.path.join(base_dir, "balanced_pos")
    neg_input = os.path.join(base_dir, "balanced_neg")

    process_dataset(pos_input, base_dir, "pos")
    process_dataset(neg_input, base_dir, "neg")

if __name__ == "__main__":
    main()
