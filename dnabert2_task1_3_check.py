#!/usr/bin/env python3
import os
import glob
import gzip
import argparse
from Bio import SeqIO
from multiprocessing import Pool, cpu_count

def count_file_stats(file_path):
    samples = 0
    total_bp = 0
    try:
        open_func = gzip.open if file_path.endswith('.gz') else open
        with open_func(file_path, 'rt') as handle:
            for record in SeqIO.parse(handle, 'fasta'):
                samples += 1
                total_bp += len(record.seq)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    return samples, total_bp

def process_dataset(directory, pattern, label):
    print(f"\nAnalyzing {label} dataset: {directory}")
    search_pattern = os.path.join(directory, "**", pattern)
    files = glob.glob(search_pattern, recursive=True)
    if not files:
        print(f"No files matching pattern '{pattern}' in {directory}")
        return 0, 0, 0

    print(f"Found {len(files)} files for {label} dataset. Processing in parallel...")

    with Pool(processes=cpu_count()) as pool:
        results = pool.map(count_file_stats, files)

    total_samples = sum(r[0] for r in results)
    total_bp = sum(r[1] for r in results)

    return total_samples, total_bp, len(files)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute statistics for positive and negative datasets")
    parser.add_argument("--positive", type=str, required=True, help="Directory containing positive samples")
    parser.add_argument("--negative", type=str, required=True, help="Directory containing negative samples")
    parser.add_argument("--pos_pattern", type=str, default="*.fasta", help="Filename pattern for positive samples")
    parser.add_argument("--neg_pattern", type=str, default="filtered_*.fna.gz", help="Filename pattern for negative samples")
    args = parser.parse_args()

    if not os.path.isdir(args.positive):
        print(f"Error: Positive directory '{args.positive}' not found")
        exit(1)
    if not os.path.isdir(args.negative):
        print(f"Error: Negative directory '{args.negative}' not found")
        exit(1)

    print("\n" + "="*50)
    print("POSITIVE DATASET STATISTICS")
    print("="*50)
    pos_samples, pos_bp, pos_files = process_dataset(args.positive, args.pos_pattern, "positive")

    print("\n" + "="*50)
    print("NEGATIVE DATASET STATISTICS")
    print("="*50)
    neg_samples, neg_bp, neg_files = process_dataset(args.negative, args.neg_pattern, "negative")

    print("\n" + "="*50)
    print("DATASET SUMMARY")
    print("="*50)
    print(f"Positive files: {pos_files}")
    print(f"Positive samples: {pos_samples:,}")
    print(f"Positive total base pairs: {pos_bp:,} bp")
    print(f"Average positive length: {pos_bp/pos_samples:,.2f} bp/sample" if pos_samples > 0 else "No samples")

    print()
    print(f"Negative files: {neg_files}")
    print(f"Negative samples: {neg_samples:,}")
    print(f"Negative total base pairs: {neg_bp:,} bp")
    print(f"Average negative length: {neg_bp/neg_samples:,.2f} bp/sample" if neg_samples > 0 else "No samples")

    print()
    print(f"Total samples: {pos_samples + neg_samples:,}")
    print(f"Total base pairs: {pos_bp + neg_bp:,} bp")
    if neg_samples > 0:
        print(f"Positive/Negative sample ratio: {pos_samples / neg_samples:.4f}")
    else:
        print("Cannot compute ratio (no negative samples)")
