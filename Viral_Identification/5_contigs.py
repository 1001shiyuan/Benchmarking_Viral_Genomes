#!/usr/bin/env python3
import os
import random
import gzip
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import numpy as np

def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def process_and_save_full_sequences(input_file, output_dir, label, split_ratio=[0.8, 0.1, 0.1]):
    """Process a FASTA file and save full sequences to train/val/test splits."""
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        return []
    
    # Create output directories
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    test_dir = os.path.join(output_dir, "test")
    ensure_dir(train_dir)
    ensure_dir(val_dir)
    ensure_dir(test_dir)
    
    # Read and shuffle sequences
    print(f"Reading sequences from {input_file}")
    sequences = []
    with gzip.open(input_file, 'rt') as handle:
        for record in SeqIO.parse(handle, 'fasta'):
            sequences.append(record)
    
    random.shuffle(sequences)
    
    # Calculate split indices
    total = len(sequences)
    train_end = int(total * split_ratio[0])
    val_end = train_end + int(total * split_ratio[1])
    
    # Split sequences
    train_seqs = sequences[:train_end]
    val_seqs = sequences[train_end:val_end]
    test_seqs = sequences[val_end:]
    
    # Process and save each split
    print(f"Saving {len(train_seqs)} training sequences")
    save_sequences_batch(train_seqs, train_dir, label)
    
    print(f"Saving {len(val_seqs)} validation sequences")
    save_sequences_batch(val_seqs, val_dir, label)
    
    print(f"Saving {len(test_seqs)} test sequences")
    save_sequences_batch(test_seqs, test_dir, label)
    
    return train_seqs, val_seqs, test_seqs

def save_sequences_batch(sequences, output_dir, label):
    """Save a batch of sequences with their labels."""
    ensure_dir(output_dir)
    
    # Save sequences in batches to avoid memory issues
    seq_ids = []
    seq_texts = []
    labels = []
    
    for record in sequences:
        seq_ids.append(str(record.id))
        seq_texts.append(str(record.seq))
        labels.append(label)
    
    # Save in batches of 1000 to avoid memory issues
    batch_size = 1000
    for i in range(0, len(seq_ids), batch_size):
        batch_end = min(i + batch_size, len(seq_ids))
        batch_suffix = f"_{i//batch_size}"
        
        np.save(os.path.join(output_dir, f"sequence_ids{batch_suffix}.npy"), np.array(seq_ids[i:batch_end]))
        np.save(os.path.join(output_dir, f"full_sequences{batch_suffix}.npy"), np.array(seq_texts[i:batch_end]))
        np.save(os.path.join(output_dir, f"labels{batch_suffix}.npy"), np.array(labels[i:batch_end]))

def generate_and_save_contigs(sequences, output_dir, label, contig_lengths, split):
    """Generate contigs from sequences and save them."""
    ensure_dir(output_dir)
    
    # Process each contig length
    for contig_length in contig_lengths:
        length_dir = output_dir
        if split == "test":
            length_dir = os.path.join(output_dir, f"length_{contig_length}")
        ensure_dir(length_dir)
        
        contig_ids = []
        contig_seqs = []
        contig_labels = []
        
        # Generate contigs for each sequence
        for record in sequences:
            seq_str = str(record.seq)
            seq_id = record.id
            
            # Generate non-overlapping contigs
            num_contigs = len(seq_str) // contig_length
            for i in range(num_contigs):
                start = i * contig_length
                end = start + contig_length
                contig = seq_str[start:end]
                
                contig_ids.append(f"{seq_id}_{i}")
                contig_seqs.append(contig)
                contig_labels.append(label)
        
        # Save contigs in batches
        batch_size = 5000
        print(f"Generated {len(contig_seqs)} contigs of length {contig_length} for {split} split")
        
        for i in range(0, len(contig_ids), batch_size):
            batch_end = min(i + batch_size, len(contig_ids))
            batch_suffix = f"_{contig_length}_{i//batch_size}"
            
            np.save(os.path.join(length_dir, f"sequence_ids{batch_suffix}.npy"), np.array(contig_ids[i:batch_end]))
            np.save(os.path.join(length_dir, f"contigs{batch_suffix}.npy"), np.array(contig_seqs[i:batch_end]))
            np.save(os.path.join(length_dir, f"labels{batch_suffix}.npy"), np.array(contig_labels[i:batch_end]))

def main():
    # Set random seed for reproducibility
    random.seed(42)
    
    # Define paths
    POS_FILE = "/work/sgk270/dnabert2_task1/balanced_pos/positive.fna.gz"
    NEG_FILE = "/work/sgk270/dnabert2_task1/balanced_neg/negative.fna.gz"
    
    POS_CONTIGS_DIR = "/work/sgk270/dnabert2_task1/pos_contigs"
    NEG_CONTIGS_DIR = "/work/sgk270/dnabert2_task1/neg_contigs"
    POS_FULL_SEQS_DIR = "/work/sgk270/dnabert2_task1/pos_full_seqs"
    NEG_FULL_SEQS_DIR = "/work/sgk270/dnabert2_task1/neg_full_seqs"
    
    # Define contig lengths
    CONTIG_LENGTHS = [500, 1000, 3000, 5000, 10000]
    
    # Create base directories
    for directory in [POS_CONTIGS_DIR, NEG_CONTIGS_DIR, POS_FULL_SEQS_DIR, NEG_FULL_SEQS_DIR]:
        ensure_dir(directory)
    
    # Process positive sequences
    print("\n=== Processing positive sequences ===")
    pos_train, pos_val, pos_test = process_and_save_full_sequences(POS_FILE, POS_FULL_SEQS_DIR, label=1)
    
    # Generate contigs for positive sequences
    print("\n=== Generating positive contigs ===")
    generate_and_save_contigs(pos_train, os.path.join(POS_CONTIGS_DIR, "train"), label=1, contig_lengths=CONTIG_LENGTHS, split="train")
    generate_and_save_contigs(pos_val, os.path.join(POS_CONTIGS_DIR, "val"), label=1, contig_lengths=CONTIG_LENGTHS, split="val")
    generate_and_save_contigs(pos_test, os.path.join(POS_CONTIGS_DIR, "test"), label=1, contig_lengths=CONTIG_LENGTHS, split="test")
    
    # Process negative sequences
    print("\n=== Processing negative sequences ===")
    neg_train, neg_val, neg_test = process_and_save_full_sequences(NEG_FILE, NEG_FULL_SEQS_DIR, label=0)
    
    # Generate contigs for negative sequences
    print("\n=== Generating negative contigs ===")
    generate_and_save_contigs(neg_train, os.path.join(NEG_CONTIGS_DIR, "train"), label=0, contig_lengths=CONTIG_LENGTHS, split="train")
    generate_and_save_contigs(neg_val, os.path.join(NEG_CONTIGS_DIR, "val"), label=0, contig_lengths=CONTIG_LENGTHS, split="val")
    generate_and_save_contigs(neg_test, os.path.join(NEG_CONTIGS_DIR, "test"), label=0, contig_lengths=CONTIG_LENGTHS, split="test")
    
    print("\n=== Processing completed successfully ===")

if __name__ == "__main__":
    main()