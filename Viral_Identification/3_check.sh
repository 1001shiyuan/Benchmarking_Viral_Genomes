#!/bin/bash
#SBATCH --job-name=dataset_stats
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --partition=compute1
#SBATCH --output=dataset_stats_%j.out
#SBATCH --error=dataset_stats_%j.err

# Print job information
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"

# Load necessary modules
module purge
module load anaconda3

# Activate virtual environment
if [ ! -d "bert1" ]; then
    echo "Creating virtual environment bert1"
    python -m venv bert1
    source bert1/bin/activate
    pip install biopython
else
    echo "Activating existing virtual environment bert1"
    source bert1/bin/activate
fi

# Set paths with absolute paths
POSITIVE_DIR="/work/sgk270/dataset_for_benchmarking/identification"
NEGATIVE_DIR="/work/sgk270/dnabert2_task1/filtered_negative"

# Create the statistics script
cat > 3_check_updated.py << 'EOL'
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
    parser.add_argument('--pos_pattern', type=str, default="*.fasta",
                        help='File pattern to search for positive samples (default: *.fasta)')
    parser.add_argument('--neg_pattern', type=str, default="filtered_*.fna.gz",
                        help='File pattern to search for negative samples (default: filtered_*.fna.gz)')
    
    args = parser.parse_args()
    
    # Check if directories exist
    if not os.path.isdir(args.positive):
        print(f"Error: Positive directory '{args.positive}' does not exist")
        # Try to find the directory by searching
        print("Searching for possible positive directories:")
        possible_dirs = glob.glob("/work/sgk270/*/identification")
        for d in possible_dirs:
            print(f"  - {d}")
    
    if not os.path.isdir(args.negative):
        print(f"Error: Negative directory '{args.negative}' does not exist")
        # Try to find filtered directories
        print("Searching for possible negative directories:")
        possible_dirs = glob.glob("/work/sgk270/*/filtered_*")
        for d in possible_dirs:
            print(f"  - {d}")
    
    if not os.path.isdir(args.positive) or not os.path.isdir(args.negative):
        exit(1)
    
    # Calculate statistics for positive dataset
    print("\n" + "="*50)
    print("POSITIVE DATASET STATISTICS")
    print("="*50)
    pos_samples, pos_bp, pos_files = count_samples_and_length(args.positive, args.pos_pattern)
    
    # Calculate statistics for negative dataset
    print("\n" + "="*50)
    print("NEGATIVE DATASET STATISTICS")
    print("="*50)
    neg_samples, neg_bp, neg_files = count_samples_and_length(args.negative, args.neg_pattern)
    
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
EOL

# Make the script executable
chmod +x 3_check_updated.py

# Run the updated statistics script
python 3_check_updated.py \
    --positive ${POSITIVE_DIR} \
    --negative ${NEGATIVE_DIR} \
    --pos_pattern "*.fasta" \
    --neg_pattern "filtered_*.fna.gz"

echo "Job finished at: $(date)"