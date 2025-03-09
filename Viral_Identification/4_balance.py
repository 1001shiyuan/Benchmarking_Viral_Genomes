#!/usr/bin/env python3
import os
import random
from Bio import SeqIO
import glob
import gzip
import argparse

def count_total_bases(directory, pattern="*.fna.gz"):
    """Count total number of bases in all sequence files in a directory."""
    total_bases = 0
    total_sequences = 0
    
    # Find all files matching the pattern
    all_files = glob.glob(os.path.join(directory, "**", pattern), recursive=True)
    
    if not all_files:
        print(f"No files matching '{pattern}' found in {directory}")
        return 0, 0
    
    for file_path in all_files:
        try:
            # Handle gzipped files
            if file_path.endswith('.gz'):
                with gzip.open(file_path, 'rt') as f:
                    for record in SeqIO.parse(f, "fasta"):
                        total_bases += len(record.seq)
                        total_sequences += 1
            # Handle regular files
            else:
                with open(file_path, 'r') as f:
                    for record in SeqIO.parse(f, "fasta"):
                        total_bases += len(record.seq)
                        total_sequences += 1
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    
    return total_bases, total_sequences

def collect_all_sequences(directory, pattern="*.fna.gz"):
    """Collect all sequences from a directory into a list."""
    sequences = []
    
    # Find all files matching the pattern
    all_files = glob.glob(os.path.join(directory, "**", pattern), recursive=True)
    
    if not all_files:
        print(f"No files matching '{pattern}' found in {directory}")
        return sequences
    
    print(f"Found {len(all_files)} files matching pattern '{pattern}' in {directory}")
    
    for idx, file_path in enumerate(all_files, 1):
        try:
            # Handle gzipped files
            if file_path.endswith('.gz'):
                with gzip.open(file_path, 'rt') as f:
                    for record in SeqIO.parse(f, "fasta"):
                        sequences.append(record)
            # Handle regular files
            else:
                with open(file_path, 'r') as f:
                    for record in SeqIO.parse(f, "fasta"):
                        sequences.append(record)
                        
            if idx % 100 == 0 or idx == len(all_files):
                print(f"Processed {idx}/{len(all_files)} files, collected {len(sequences)} sequences so far...")
                
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    
    return sequences

def downsample_and_save(sequences, target_bases, output_dir, dataset_type):
    """Randomly sample sequences until reaching target base count."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Shuffle sequences
    random.shuffle(sequences)
    
    # Sample sequences until reaching target
    sampled_sequences = []
    current_bases = 0
    
    for seq in sequences:
        if current_bases >= target_bases:
            break
        sampled_sequences.append(seq)
        current_bases += len(seq.seq)
    
    # Save sampled sequences
    output_file = os.path.join(output_dir, f"{dataset_type}.fna.gz")
    with gzip.open(output_file, 'wt') as f:
        SeqIO.write(sampled_sequences, f, "fasta")
    
    print(f"Wrote {len(sampled_sequences):,} sequences ({current_bases:,} bases) to {output_file}")
    return len(sampled_sequences), current_bases

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Balance positive and negative datasets')
    parser.add_argument('--positive', type=str, required=True, 
                        help='Directory containing positive samples')
    parser.add_argument('--negative', type=str, required=True,
                        help='Directory containing negative samples')
    parser.add_argument('--pos_pattern', type=str, default="*.fasta",
                        help='File pattern for positive samples (default: *.fasta)')
    parser.add_argument('--neg_pattern', type=str, default="filtered_*.fna.gz",
                        help='File pattern for negative samples (default: filtered_*.fna.gz)')
    parser.add_argument('--output_dir', type=str, default="./",
                        help='Output directory (default: current directory)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Create output directories
    pos_output_dir = os.path.join(args.output_dir, "balanced_pos")
    neg_output_dir = os.path.join(args.output_dir, "balanced_neg")
    
    # Count bases in positive dataset
    print("\nCounting bases in positive dataset...")
    pos_bases, pos_seqs = count_total_bases(args.positive, args.pos_pattern)
    print(f"Positive dataset: {pos_bases:,} bases in {pos_seqs:,} sequences")
    
    # Count bases in negative dataset
    print("\nCounting bases in negative dataset...")
    neg_bases, neg_seqs = count_total_bases(args.negative, args.neg_pattern)
    print(f"Negative dataset: {neg_bases:,} bases in {neg_seqs:,} sequences")
    
    # Determine which dataset is smaller
    if pos_bases <= neg_bases:
        smaller_dataset = "positive"
        target_bases = pos_bases
        print(f"\nPositive dataset is smaller ({pos_bases:,} bases vs {neg_bases:,} bases)")
    else:
        smaller_dataset = "negative"
        target_bases = neg_bases
        print(f"\nNegative dataset is smaller ({neg_bases:,} bases vs {pos_bases:,} bases)")
    
    # Collect sequences from both datasets
    print("\nCollecting sequences from positive dataset...")
    pos_sequences = collect_all_sequences(args.positive, args.pos_pattern)
    print(f"Total positive sequences collected: {len(pos_sequences):,}")
    
    print("\nCollecting sequences from negative dataset...")
    neg_sequences = collect_all_sequences(args.negative, args.neg_pattern)
    print(f"Total negative sequences collected: {len(neg_sequences):,}")
    
    # Balance datasets
    if smaller_dataset == "positive":
        # Copy all positive sequences to output directory
        print("\nCopying all positive sequences...")
        n_sampled_pos, sampled_pos_bases = downsample_and_save(
            pos_sequences, 
            pos_bases,  # Use all positive sequences
            pos_output_dir,
            "positive"
        )
        
        # Downsample negative sequences to match positive dataset
        print("\nDownsampling negative sequences...")
        n_sampled_neg, sampled_neg_bases = downsample_and_save(
            neg_sequences, 
            pos_bases,  # Match positive dataset size
            neg_output_dir,
            "negative"
        )
    else:
        # Downsample positive sequences to match negative dataset
        print("\nDownsampling positive sequences...")
        n_sampled_pos, sampled_pos_bases = downsample_and_save(
            pos_sequences, 
            neg_bases,  # Match negative dataset size
            pos_output_dir,
            "positive"
        )
        
        # Copy all negative sequences to output directory
        print("\nCopying all negative sequences...")
        n_sampled_neg, sampled_neg_bases = downsample_and_save(
            neg_sequences, 
            neg_bases,  # Use all negative sequences
            neg_output_dir,
            "negative"
        )
    
    # Print final statistics
    print("\nFinal Statistics:")
    print(f"Positive: {sampled_pos_bases:,} bases in {n_sampled_pos:,} sequences")
    print(f"Negative: {sampled_neg_bases:,} bases in {n_sampled_neg:,} sequences")
    print(f"Ratio (Positive/Negative): {sampled_pos_bases/sampled_neg_bases:.4f}")