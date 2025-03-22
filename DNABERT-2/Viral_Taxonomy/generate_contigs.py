import os
import random
from typing import List, Tuple, Dict
import numpy as np

def read_sequences(file_path: str) -> List[Tuple[str, str, int]]:
    """
    Read sequences from FASTA file.
    Returns: List of tuples (sequence_id, sequence, length).
    """
    sequences = []
    current_seq = []
    current_id = None

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_seq and current_id:
                    full_seq = ''.join(current_seq)
                    sequences.append((current_id, full_seq, len(full_seq)))
                current_seq = []
                current_id = line[1:].split()[0]
            else:
                current_seq.append(line)

        # Process the last sequence
        if current_seq and current_id:
            full_seq = ''.join(current_seq)
            sequences.append((current_id, full_seq, len(full_seq)))

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
        contigs.append((seq_id, contig, len(contig)))

    return contigs

def save_contigs(contigs: List[Tuple[str, str, int]], output_dir: str):
    """Save contigs to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    contigs_list = [contig for _, contig, _ in contigs]
    seq_ids = [seq_id for seq_id, _, _ in contigs]
    
    np.save(os.path.join(output_dir, "contigs.npy"), np.array(contigs_list))
    np.save(os.path.join(output_dir, "sequence_ids.npy"), np.array(seq_ids))
    print(f"Saved {len(contigs)} contigs")

def main():
    random.seed(42)

    # Define paths
    INPUT_FILE = "/work/sgk270/dataset_for_benchmarking/ICTV/ICTV_sequences.fasta"
    CONTIGS_DIR = "/work/sgk270/dnabert2_task2/contigs"
    FULL_SEQS_DIR = "/work/sgk270/dnabert2_task2/full_seqs"

    # Define contig lengths
    CONTIG_LENGTHS = [500, 1000, 3000, 5000, 10000]

    # Create directories
    for base_dir in [CONTIGS_DIR, FULL_SEQS_DIR]:
        os.makedirs(base_dir, exist_ok=True)

    # Read sequences
    print("Reading sequences...")
    sequences = read_sequences(INPUT_FILE)
    print(f"Found {len(sequences)} sequences")
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