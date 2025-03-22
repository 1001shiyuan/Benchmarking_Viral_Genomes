import numpy as np
import os
from collections import Counter
import shutil

def load_embeddings(filepath):
    """Load embeddings and labels from npz file."""
    data = np.load(filepath)
    return {
        'embeddings': data['embeddings'],
        'order_labels': data['order_labels'],
        'family_labels': data['family_labels'],
        'genus_labels': data['genus_labels'],
        'sequence_ids': data['sequence_ids'] if 'sequence_ids' in data else None,
        'order_text': data['order_text'] if 'order_text' in data else None,
        'family_text': data['family_text'] if 'family_text' in data else None,
        'genus_text': data['genus_text'] if 'genus_text' in data else None
    }

def get_all_full_seq_label_counts(full_seq_dir, label_type):
    """Get counts of labels across all full sequence splits combined."""
    all_counts = Counter()
    
    # Combine counts from train, val, and test splits
    for split in ['train', 'val', 'test']:
        filepath = os.path.join(full_seq_dir, f'{split}_embeddings.npz')
        if os.path.exists(filepath):
            data = load_embeddings(filepath)
            all_counts.update(Counter(data[f'{label_type}_labels']))
    
    return all_counts

def filter_data(data, valid_labels, label_type):
    """Filter data to keep only valid labels."""
    mask = np.isin(data[f'{label_type}_labels'], valid_labels)
    result = {
        'embeddings': data['embeddings'][mask],
        'order_labels': data['order_labels'][mask],
        'family_labels': data['family_labels'][mask],
        'genus_labels': data['genus_labels'][mask]
    }
    
    # Add optional fields if present in the data
    for field in ['sequence_ids', 'order_text', 'family_text', 'genus_text']:
        if field in data and data[field] is not None:
            result[field] = data[field][mask]
    
    return result

def save_filtered_data(data, filepath):
    """Save filtered data to npz file."""
    # Prepare the data dictionary for savez
    save_dict = {
        'embeddings': data['embeddings'],
        'order_labels': data['order_labels'],
        'family_labels': data['family_labels'],
        'genus_labels': data['genus_labels']
    }
    
    # Add optional fields if present
    for field in ['sequence_ids', 'order_text', 'family_text', 'genus_text']:
        if field in data and data[field] is not None:
            save_dict[field] = data[field]
    
    np.savez(filepath, **save_dict)

def filter_and_save_data(input_dir, output_dir, valid_labels, label_type):
    """Filter and save data based on valid labels."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy label encoders if they exist
    encoder_path = os.path.join(input_dir, 'label_encoders.npy')
    if os.path.exists(encoder_path):
        shutil.copy2(
            encoder_path,
            os.path.join(output_dir, 'label_encoders.npy')
        )
    
    # Count total samples before and after filtering
    total_samples = 0
    total_filtered_samples = 0
    
    # Process each split
    for split in ['train', 'val', 'test']:
        input_path = os.path.join(input_dir, f'{split}_embeddings.npz')
        if os.path.exists(input_path):
            data = load_embeddings(input_path)
            total_samples += len(data['embeddings'])
            filtered_data = filter_data(data, valid_labels, label_type)
            total_filtered_samples += len(filtered_data['embeddings'])
            
            # Save filtered data
            save_filtered_data(
                filtered_data,
                os.path.join(output_dir, f'{split}_embeddings.npz')
            )
            print(f"  Filtered {split} set - {len(filtered_data['embeddings'])} samples")
    
    return total_samples, total_filtered_samples

def process_full_sequences(full_seq_dir, thresholds, base_output_dir):
    """Process full sequences and determine valid labels based on thresholds."""
    valid_labels = {}
    
    for label_type in ['order', 'family', 'genus']:
        print(f"\nDetermining valid {label_type} labels from full sequences...")
        
        # Get counts of labels in full sequences
        label_counts = get_all_full_seq_label_counts(full_seq_dir, f'{label_type}')
        all_labels = np.array(list(label_counts.keys()))
        
        # Get valid labels (those with enough samples)
        threshold = thresholds[label_type]
        valid = np.array([label for label in all_labels 
                         if label_counts.get(label, 0) >= threshold])
        
        print(f"Total unique {label_type} classes: {len(all_labels)}")
        print(f"Classes with >= {threshold} samples: {len(valid)}")
        
        valid_labels[label_type] = valid
        
        # Create output directory
        output_dir = os.path.join(base_output_dir, f'{label_type}_full_embed')
        
        # Filter and save full sequence data
        print(f"Filtering full sequences for {label_type}...")
        total, filtered = filter_and_save_data(
            full_seq_dir, 
            output_dir, 
            valid, 
            f'{label_type}'
        )
        
        print(f"Full sequences - total: {total}, after filtering: {filtered}")
    
    return valid_labels

def process_contig_data(valid_labels, contig_dir, contig_lengths, base_output_dir):
    """Process contig data using valid labels from full sequences."""
    for label_type in ['order', 'family', 'genus']:
        print(f"\nFiltering contigs for {label_type}...")
        
        # Process combined train and val directories
        for split in ['train', 'val']:
            input_dir = contig_dir
            output_dir = os.path.join(base_output_dir, f'{label_type}_contig_embed')
            
            # Check if split file exists
            input_path = os.path.join(input_dir, f'{split}_embeddings.npz')
            if os.path.exists(input_path):
                print(f"Filtering {split} contigs...")
                
                # Create output directory
                os.makedirs(output_dir, exist_ok=True)
                
                # Load data
                data = load_embeddings(input_path)
                
                # Filter and save
                filtered_data = filter_data(data, valid_labels[label_type], f'{label_type}')
                save_filtered_data(
                    filtered_data,
                    os.path.join(output_dir, f'{split}_embeddings.npz')
                )
                
                print(f"  {split} contigs - total: {len(data['embeddings'])}, after filtering: {len(filtered_data['embeddings'])}")
        
        # Process test directories by length
        for length in contig_lengths:
            length_input_dir = os.path.join(contig_dir, f'length_{length}')
            length_output_dir = os.path.join(base_output_dir, f'{label_type}_contig_embed', f'length_{length}')
            
            # Check if directory exists
            if os.path.exists(length_input_dir):
                # Check if test file exists
                test_input_path = os.path.join(length_input_dir, 'test_embeddings.npz')
                if os.path.exists(test_input_path):
                    print(f"Filtering test contigs of length {length}...")
                    
                    # Create output directory
                    os.makedirs(length_output_dir, exist_ok=True)
                    
                    # Load data
                    data = load_embeddings(test_input_path)
                    
                    # Filter and save
                    filtered_data = filter_data(data, valid_labels[label_type], f'{label_type}')
                    save_filtered_data(
                        filtered_data,
                        os.path.join(length_output_dir, 'test_embeddings.npz')
                    )
                    
                    print(f"  test contigs (length {length}) - total: {len(data['embeddings'])}, after filtering: {len(filtered_data['embeddings'])}")
                
                # Copy label encoders
                encoder_path = os.path.join(length_input_dir, 'label_encoders.npy')
                if os.path.exists(encoder_path):
                    os.makedirs(length_output_dir, exist_ok=True)
                    shutil.copy2(
                        encoder_path,
                        os.path.join(length_output_dir, 'label_encoders.npy')
                    )

def main():
    # Define paths
    BASE_DIR = '/work/sgk270/genalm_task2'
    FULL_SEQ_DIR = os.path.join(BASE_DIR, 'full_embed')
    CONTIG_DIR = os.path.join(BASE_DIR, 'contig_embed')
    
    # Define contig lengths
    CONTIG_LENGTHS = [500, 1000, 3000, 5000, 10000]
    
    # Define thresholds for each level - using full sequences
    THRESHOLDS = {
        'order': 80,
        'family': 60,
        'genus': 40
    }
    
    # Copy main label encoders
    for label_type in ['order', 'family', 'genus']:
        output_dir = os.path.join(BASE_DIR, f'{label_type}_contig_embed')
        os.makedirs(output_dir, exist_ok=True)
        source_encoder = os.path.join(CONTIG_DIR, 'label_encoders.npy')
        if os.path.exists(source_encoder):
            shutil.copy2(
                source_encoder,
                os.path.join(output_dir, 'label_encoders.npy')
            )
    
    # Process full sequences to determine valid labels
    valid_labels = process_full_sequences(FULL_SEQ_DIR, THRESHOLDS, BASE_DIR)
    
    # Process contig data using the valid labels
    process_contig_data(valid_labels, CONTIG_DIR, CONTIG_LENGTHS, BASE_DIR)
    
    print("\nFiltering complete!")

if __name__ == "__main__":
    main()