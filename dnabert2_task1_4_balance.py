#!/usr/bin/env python3
import os
import random
import glob
import gzip
import argparse
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from multiprocessing import Pool, cpu_count

def list_files(directory, pattern):
    return glob.glob(os.path.join(directory, "**", pattern), recursive=True)

def count_bases_in_file(file_path):
    total_bases = 0
    total_seqs = 0
    try:
        with (gzip.open(file_path, 'rt') if file_path.endswith('.gz') else open(file_path, 'r')) as f:
            for record in SeqIO.parse(f, "fasta"):
                total_bases += len(record.seq)
                total_seqs += 1
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    return total_bases, total_seqs

def collect_seqs_from_file(file_path):
    seqs = []
    try:
        with (gzip.open(file_path, 'rt') if file_path.endswith('.gz') else open(file_path, 'r')) as f:
            for record in SeqIO.parse(f, "fasta"):
                seqs.append(record)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    return seqs

def count_total_bases_parallel(directory, pattern):
    files = list_files(directory, pattern)
    if not files:
        print(f"No files matching '{pattern}' in {directory}")
        return 0, 0
    print(f"Counting bases from {len(files)} files in parallel...")
    with Pool(cpu_count()) as pool:
        results = pool.map(count_bases_in_file, files)
    total_bases = sum(x[0] for x in results)
    total_seqs = sum(x[1] for x in results)
    return total_bases, total_seqs

def collect_all_sequences_parallel(directory, pattern):
    files = list_files(directory, pattern)
    if not files:
        print(f"No files matching '{pattern}' in {directory}")
        return []
    print(f"Collecting sequences from {len(files)} files in parallel...")
    with Pool(cpu_count()) as pool:
        all_seqs_nested = pool.map(collect_seqs_from_file, files)
    all_seqs = [rec for sublist in all_seqs_nested for rec in sublist]
    return all_seqs

def downsample_and_save(sequences, target_bases, output_dir, dataset_type):
    os.makedirs(output_dir, exist_ok=True)
    random.shuffle(sequences)

    sampled = []
    current_bases = 0
    for seq in sequences:
        if current_bases >= target_bases:
            break
        sampled.append(seq)
        current_bases += len(seq.seq)

    output_file = os.path.join(output_dir, f"{dataset_type}.fna.gz")
    with gzip.open(output_file, 'wt') as f:
        SeqIO.write(sampled, f, "fasta")

    print(f"Wrote {len(sampled):,} sequences ({current_bases:,} bases) to {output_file}")
    return len(sampled), current_bases

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parallel base-matching balance of datasets')
    parser.add_argument('--positive', type=str, required=True)
    parser.add_argument('--negative', type=str, required=True)
    parser.add_argument('--pos_pattern', type=str, default="*.fasta")
    parser.add_argument('--neg_pattern', type=str, default="filtered_*.fna.gz")
    parser.add_argument('--output_dir', type=str, default="./")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    pos_output_dir = os.path.join(args.output_dir, "balanced_pos")
    neg_output_dir = os.path.join(args.output_dir, "balanced_neg")

    print("\nCounting positive dataset...")
    pos_bases, pos_seqs = count_total_bases_parallel(args.positive, args.pos_pattern)
    print(f"Positive: {pos_bases:,} bp in {pos_seqs:,} sequences")

    print("\nCounting negative dataset...")
    neg_bases, neg_seqs = count_total_bases_parallel(args.negative, args.neg_pattern)
    print(f"Negative: {neg_bases:,} bp in {neg_seqs:,} sequences")

    if pos_bases <= neg_bases:
        target_bases = pos_bases
        print(f"\nPositive dataset is smaller. Matching negative to {target_bases:,} bp")
    else:
        target_bases = neg_bases
        print(f"\nNegative dataset is smaller. Matching positive to {target_bases:,} bp")

    print("\nLoading positive sequences...")
    pos_seqs = collect_all_sequences_parallel(args.positive, args.pos_pattern)
    print(f"Loaded {len(pos_seqs):,} positive sequences")

    print("\nLoading negative sequences...")
    neg_seqs = collect_all_sequences_parallel(args.negative, args.neg_pattern)
    print(f"Loaded {len(neg_seqs):,} negative sequences")

    if pos_bases <= neg_bases:
        print("\nSaving full positive dataset...")
        n_pos, bp_pos = downsample_and_save(pos_seqs, pos_bases, pos_output_dir, "positive")

        print("\nDownsampling negative dataset...")
        n_neg, bp_neg = downsample_and_save(neg_seqs, pos_bases, neg_output_dir, "negative")
    else:
        print("\nDownsampling positive dataset...")
        n_pos, bp_pos = downsample_and_save(pos_seqs, neg_bases, pos_output_dir, "positive")

        print("\nSaving full negative dataset...")
        n_neg, bp_neg = downsample_and_save(neg_seqs, neg_bases, neg_output_dir, "negative")

    print("\nFinal Statistics:")
    print(f"Positive: {bp_pos:,} bp in {n_pos:,} sequences")
    print(f"Negative: {bp_neg:,} bp in {n_neg:,} sequences")
    print(f"Ratio (Positive/Negative): {bp_pos / bp_neg:.4f}")
