#!/usr/bin/env python3
import os
import glob
import gzip
from Bio import SeqIO
import argparse

def count_samples_and_length(directory, file_pattern="*.fna.gz", recursive=True):
    """
    Count the number of samples and total sequence length in a directory.
    
    Args:
        directory: Directory containing sequences
        file_pattern: Pattern to match files (default: *.fna.gz)
        recursive: Whether to search recursively (default: True)
    
    Returns:
        tuple: (total_samples, total_bp, file_count)
    """
    print(f"\nAnalyzing directory: {directory}")
    
    # Find all files matching the pattern
    search_pattern = os.path.join(directory, "**", file_pattern) if recursive else os.path.join(directory, file_pattern)
    files = glob.glob(search_pattern, recursive=recursive)
    
    if not files:
        print(f"No files matching '{file_pattern}' found in {directory}")
        return 0, 0, 0
    
    print(f"Found {len(files)} files matching pattern '{file_pattern}'")
    
    total_samples = 0
    total_bp = 0
    
    # Process each file
    for idx, file_path in enumerate(files, 1):
        file_samples = 0
        file_bp = 0
        
        try:
            # Handle gzipped files
            if file_path.endswith('.gz'):
                with gzip.open(file_path, 'rt') as handle:
                    for record in SeqIO.parse(handle, 'fasta'):
                        file_samples += 1
                        file_bp += len(record.seq)
            # Handle regular files
            else:
                with open(file_path, 'r') as handle:
                    for record in SeqIO.parse(handle, 'fasta'):
                        file_samples += 1
                        file_bp += len(record.seq)
                
            total_samples += file_samples
            total_bp += file_bp
            
            if idx % 10 == 0 or idx == len(files):
                print(f"Processed {idx}/{len(files)} files...")
                
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    
    return total_samples, total_bp, len(files)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate statistics for sequence datasets')
    parser.add_argument('--positive', type=str, required=True, 
                        help='Directory containing positive samples')
    parser.add_argument('--negative', type=str, required=True,
                        help='Directory containing negative samples')
    parser.add_argument('--pattern', type=str, default="*.fna.gz",
                        help='File pattern to search for (default: *.fna.gz)')
    
    args = parser.parse_args()
    
    # Check if directories exist
    if not os.path.isdir(args.positive):
        print(f"Error: Positive directory '{args.positive}' does not exist")
    
    if not os.path.isdir(args.negative):
        print(f"Error: Negative directory '{args.negative}' does not exist")
    
    if not os.path.isdir(args.positive) or not os.path.isdir(args.negative):
        exit(1)
    
    # Calculate statistics for positive dataset
    print("\n" + "="*50)
    print("POSITIVE DATASET STATISTICS")
    print("="*50)
    pos_samples, pos_bp, pos_files = count_samples_and_length(args.positive, args.pattern)
    
    # Calculate statistics for negative dataset
    print("\n" + "="*50)
    print("NEGATIVE DATASET STATISTICS")
    print("="*50)
    neg_samples, neg_bp, neg_files = count_samples_and_length(args.negative, args.pattern)
    
    # Print summary
    print("\n" + "="*50)
    print("DATASET SUMMARY")
    print("="*50)
    print(f"Positive dataset:")
    print(f"  - Files: {pos_files}")
    print(f"  - Samples: {pos_samples:,}")
    print(f"  - Total base pairs: {pos_bp:,} bp")
    print(f"  - Average sequence length: {pos_bp/pos_samples:,.2f} bp per sample" if pos_samples > 0 else "  - No samples found")
    print()
    print(f"Negative dataset:")
    print(f"  - Files: {neg_files}")
    print(f"  - Samples: {neg_samples:,}")
    print(f"  - Total base pairs: {neg_bp:,} bp")
    print(f"  - Average sequence length: {neg_bp/neg_samples:,.2f} bp per sample" if neg_samples > 0 else "  - No samples found")
    print()
    print(f"Total samples: {pos_samples + neg_samples:,}")
    print(f"Total base pairs: {pos_bp + neg_bp:,} bp")
    print(f"Positive/Negative sample ratio: {pos_samples/neg_samples:.4f}" if neg_samples > 0 else "Cannot calculate ratio (no negative samples)")