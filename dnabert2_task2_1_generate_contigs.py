import os
import random
import csv
from typing import List, Tuple, Dict, Set
import numpy as np
import glob
from Bio import SeqIO

def read_valid_sequence_ids(metadata_path: str) -> Set[str]:
    """
    Read metadata from CSV file and identify sequences with taxonomic information.
    Returns: Set of sequence_ids that have at least one taxonomic level.
    """
    valid_seq_ids = set()
    
    with open(metadata_path, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            seq_id = row['sequence_id']
            # Check if the sequence has at least one taxonomic level filled
            if row['Order'] or row['Family'] or row['Genus']:
                valid_seq_ids.add(seq_id)
    
    print(f"Found {len(valid_seq_ids)} sequences with taxonomic information")
    return valid_seq_ids

def read_sequence_files(directory_path: str, valid_seq_ids: Set[str]) -> List[Tuple[str, str, int]]:
    """
    Read sequences from multiple FASTA files in a directory.
    Only includes sequences with IDs in valid_seq_ids.
    Returns: List of tuples (sequence_id, sequence, length).
    """
    sequences = []
    fasta_files = glob.glob(os.path.join(directory_path, "*.fasta"))
    processed_files = 0
    
    for fasta_file in fasta_files:
        # Extract base filename without extension
        base_filename = os.path.basename(fasta_file).split('.')[0]
        
        # Skip if the base filename is not in valid_seq_ids
        if base_filename not in valid_seq_ids:
            continue
            
        # Read sequences from the file
        for record in SeqIO.parse(fasta_file, "fasta"):
            # Keep the original base_filename as the sequence_id for easy metadata lookup
            sequences.append((base_filename, str(record.seq), len(record.seq)))
        
        processed_files += 1
    
    print(f"Read {len(sequences)} sequences from {processed_files} FASTA files")
    return sequences

def split_sequences_by_length_ratio(sequences: List[Tuple[str, str, int]], ratios: List[float]) -> List[List[Tuple[str, str, int]]]:
    """Split sequences into groups based on total length ratios."""
    random.shuffle(sequences)

    # Calculate total length
    total_length = sum(length for _, _, length in sequences)
    train_target = ratios[0] * total_length
    val_target = ratios[1] * total_length

    # Initialize splits
    train_split, val_split, test_split = [], [], []
    current_length = 0

    # Distribute sequences based on cumulative length
    for seq_id, seq, length in sequences:
        if current_length < train_target:
            train_split.append((seq_id, seq, length))
        elif current_length < train_target + val_target:
            val_split.append((seq_id, seq, length))
        else:
            test_split.append((seq_id, seq, length))
        current_length += length

    # Print achieved ratios
    final_lengths = [sum(seq[2] for seq in train_split), sum(seq[2] for seq in val_split), sum(seq[2] for seq in test_split)]
    final_ratios = [length / total_length for length in final_lengths]

    print("\nAchieved split ratios:")
    print(f"Train: {final_ratios[0]:.3f} (target: {ratios[0]:.3f})")
    print(f"Validation: {final_ratios[1]:.3f} (target: {ratios[1]:.3f})")
    print(f"Test: {final_ratios[2]:.3f} (target: {ratios[2]:.3f})")

    return train_split, val_split, test_split

def save_full_sequences(sequences: List[Tuple[str, str, int]], output_dir: str):
    """Save full sequences and IDs to .npy files."""
    os.makedirs(output_dir, exist_ok=True)

    sequence_ids = [seq_id for seq_id, _, _ in sequences]
    full_sequences = [seq for _, seq, _ in sequences]

    np.save(os.path.join(output_dir, "full_sequences.npy"), np.array(full_sequences))
    np.save(os.path.join(output_dir, "sequence_ids.npy"), np.array(sequence_ids))

    print(f"Saved {len(sequences)} full sequences to {output_dir}")

def generate_fixed_length_contigs(sequence: str, seq_id: str, contig_length: int) -> List[Tuple[str, str, int]]:
    """Generate non-overlapping contigs of fixed length from a sequence."""
    contigs = []
    num_contigs = len(sequence) // contig_length

    for i in range(num_contigs):
        start = i * contig_length
        end = start + contig_length
        contig = sequence[start:end]
        # Keep the original sequence ID for easy metadata lookup
        contigs.append((seq_id, contig, len(contig)))

    return contigs

def save_contigs(contigs: List[Tuple[str, str, int]], output_dir: str):
    """Save contigs to .npy files."""
    os.makedirs(output_dir, exist_ok=True)
    
    contigs_list = [contig for _, contig, _ in contigs]
    seq_ids = [seq_id for seq_id, _, _ in contigs]
    
    np.save(os.path.join(output_dir, "contigs.npy"), np.array(contigs_list))
    np.save(os.path.join(output_dir, "sequence_ids.npy"), np.array(seq_ids))
    print(f"Saved {len(contigs)} contigs")

def main():
    random.seed(42)

    # Define paths
    FASTA_DIR = "/work/sgk270/dataset_for_benchmarking/combine3/final_dataset/with_meta"
    METADATA_PATH = "/work/sgk270/dataset_for_benchmarking/combine3/final_dataset/combined_metadata.csv"
    CONTIGS_DIR = "/work/sgk270/dnabert2_task2_new/contigs"
    FULL_SEQS_DIR = "/work/sgk270/dnabert2_task2_new/full_seqs"

    # Define contig lengths
    CONTIG_LENGTHS = [500, 1000, 3000, 5000, 10000]

    # Create directories
    for base_dir in [CONTIGS_DIR, FULL_SEQS_DIR]:
        os.makedirs(base_dir, exist_ok=True)

    # Read metadata to identify sequences with taxonomic information
    print("Reading metadata...")
    valid_seq_ids = read_valid_sequence_ids(METADATA_PATH)

    # Read sequences from FASTA files
    print("\nReading sequences...")
    sequences = read_sequence_files(FASTA_DIR, valid_seq_ids)
    print(f"Total sequence length: {sum(length for _, _, length in sequences):,} bp")

    # Split sequences by length ratio (80:10:10)
    print("\nSplitting sequences by length ratio (80:10:10)...")
    train_sequences, val_sequences, test_sequences = split_sequences_by_length_ratio(sequences, [0.8, 0.1, 0.1])

    # Save full sequences
    print("\nSaving full sequences...")
    save_full_sequences(train_sequences, os.path.join(FULL_SEQS_DIR, "train"))
    save_full_sequences(val_sequences, os.path.join(FULL_SEQS_DIR, "val"))
    save_full_sequences(test_sequences, os.path.join(FULL_SEQS_DIR, "test"))

    # Generate contigs for all 5 lengths
    print("\nGenerating contigs of all lengths...")

    # Initialize structures to hold contigs for train and val (all lengths together)
    all_train_contigs = []
    all_val_contigs = []
    
    # Generate contigs for each length
    for contig_length in CONTIG_LENGTHS:
        print(f"\nProcessing contigs of length {contig_length}bp...")
        
        # Process training sequences and add to combined collection
        print("Processing training sequences...")
        train_contigs = [
            contig 
            for seq_id, seq, _ in train_sequences 
            for contig in generate_fixed_length_contigs(seq, seq_id, contig_length)
        ]
        print(f"Generated {len(train_contigs)} training contigs of length {contig_length}")
        all_train_contigs.extend(train_contigs)
        
        # Process validation sequences and add to combined collection
        print("Processing validation sequences...")
        val_contigs = [
            contig 
            for seq_id, seq, _ in val_sequences 
            for contig in generate_fixed_length_contigs(seq, seq_id, contig_length)
        ]
        print(f"Generated {len(val_contigs)} validation contigs of length {contig_length}")
        all_val_contigs.extend(val_contigs)
        
        # Process test sequences and save to length-specific folders
        print("Processing test sequences...")
        test_contigs = [
            contig 
            for seq_id, seq, _ in test_sequences 
            for contig in generate_fixed_length_contigs(seq, seq_id, contig_length)
        ]
        print(f"Generated {len(test_contigs)} test contigs of length {contig_length}")
        
        # Save test contigs for this length in separate folder
        test_length_dir = os.path.join(CONTIGS_DIR, "test", f"length_{contig_length}")
        save_contigs(test_contigs, test_length_dir)

    # Save combined train contigs (all lengths)
    print("\nSaving combined training contigs of all lengths...")
    train_dir = os.path.join(CONTIGS_DIR, "train")
    save_contigs(all_train_contigs, train_dir)
    
    # Save combined validation contigs (all lengths)
    print("Saving combined validation contigs of all lengths...")
    val_dir = os.path.join(CONTIGS_DIR, "val")
    save_contigs(all_val_contigs, val_dir)

if __name__ == "__main__":
    main()