import numpy as np
import os
import pandas as pd
from collections import Counter
import shutil
import datetime
from sklearn.preprocessing import LabelEncoder

def log_with_time(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)

def load_metadata():
    metadata_path = "/work/sgk270/dataset_for_benchmarking/combine3/final_dataset/combined_metadata.csv"
    log_with_time(f"Loading metadata from {metadata_path}")
    
    metadata = pd.read_csv(metadata_path, sep='\t')
    log_with_time(f"Loaded metadata with {len(metadata)} entries")
    
    order_dict = {}
    family_dict = {}
    genus_dict = {}

    for _, row in metadata.iterrows():
        seq_id = row['sequence_id']
        if pd.notna(row['Order']):
            order_dict[seq_id] = row['Order']
        if pd.notna(row['Family']):
            family_dict[seq_id] = row['Family']
        if pd.notna(row['Genus']):
            genus_dict[seq_id] = row['Genus']
    
    log_with_time(f"Prepared lookup dictionaries: Order ({len(order_dict)}), Family ({len(family_dict)}), Genus ({len(genus_dict)})")
    return order_dict, family_dict, genus_dict

def load_embeddings(filepath):
    log_with_time(f"Loading embeddings from {filepath}")
    data = np.load(filepath, allow_pickle=True)
    result = {
        'embeddings': data['embeddings'],
        'sequence_ids': data['sequence_ids']
    }
    log_with_time(f"Loaded {len(result['embeddings'])} embeddings with shape {result['embeddings'].shape}")
    return result

def add_taxonomic_labels(data, taxonomy_dicts, label_encoders):
    order_dict, family_dict, genus_dict = taxonomy_dicts
    sequence_ids = data['sequence_ids']

    def encode_labels(seq_ids, label_dict, encoder, label_type):
        texts = np.array([label_dict.get(seq_id, "NA") for seq_id in seq_ids], dtype=object)
        labels = np.full(len(seq_ids), -1, dtype=int)
        known_mask = texts != "NA"
        try:
            labels[known_mask] = encoder.transform(texts[known_mask])
        except Exception as e:
            log_with_time(f"Warning: label encoding failed for {label_type}: {e}")
        return labels, texts

    order_labels, order_text = encode_labels(sequence_ids, order_dict, label_encoders['order'], 'order')
    family_labels, family_text = encode_labels(sequence_ids, family_dict, label_encoders['family'], 'family')
    genus_labels, genus_text = encode_labels(sequence_ids, genus_dict, label_encoders['genus'], 'genus')

    data['order_labels'] = order_labels
    data['family_labels'] = family_labels
    data['genus_labels'] = genus_labels
    data['order_text'] = order_text
    data['family_text'] = family_text
    data['genus_text'] = genus_text

    log_with_time(f"Matched taxonomic labels: Order ({np.sum(order_labels >= 0)}), " +
                  f"Family ({np.sum(family_labels >= 0)}), " +
                  f"Genus ({np.sum(genus_labels >= 0)})")
    return data

def get_all_full_seq_label_counts(full_seq_dir, taxonomy_dicts, label_encoders, label_type):
    order_dict, family_dict, genus_dict = taxonomy_dicts
    taxonomy_dict = {'order': order_dict, 'family': family_dict, 'genus': genus_dict}[label_type]
    all_counts = Counter()

    for split in ['train', 'val', 'test']:
        filepath = os.path.join(full_seq_dir, f'{split}_embeddings.npz')
        if os.path.exists(filepath):
            data = load_embeddings(filepath)
            for i, seq_id in enumerate(data['sequence_ids']):
                if seq_id in taxonomy_dict:
                    try:
                        label = label_encoders[label_type].transform([taxonomy_dict[seq_id]])[0]
                        all_counts[label] += 1
                    except:
                        pass
    return all_counts

def filter_data(data, valid_labels, label_type):
    field_name = f'{label_type}_labels'
    if field_name not in data:
        log_with_time(f"Warning: {field_name} not found in data")
        return None

    labels = data[field_name]
    mask = (labels >= 0) & np.isin(labels, valid_labels)

    if not np.any(mask):
        log_with_time(f"Warning: No valid samples found after filtering for {label_type}")
        return None

    result = {}
    for field in data:
        if data[field] is not None:
            result[field] = data[field][mask]

    return result

def save_filtered_data(data, filepath):
    if data is None:
        log_with_time(f"No data to save to {filepath}")
        return

    log_with_time(f"Saving filtered data to {filepath} ({len(data['embeddings'])} samples)")
    np.savez(filepath, **data)

def filter_and_save_data(input_dir, output_dir, taxonomy_dicts, label_encoders, valid_labels, label_type):
    os.makedirs(output_dir, exist_ok=True)

    encoder_data = {
        'order': {'classes': label_encoders['order'].classes_},
        'family': {'classes': label_encoders['family'].classes_},
        'genus': {'classes': label_encoders['genus'].classes_}
    }
    np.save(os.path.join(output_dir, 'label_encoders.npy'), encoder_data)

    total_samples = 0
    total_filtered_samples = 0

    for split in ['train', 'val', 'test']:
        input_path = os.path.join(input_dir, f'{split}_embeddings.npz')
        if os.path.exists(input_path):
            data = load_embeddings(input_path)
            total_samples += len(data['embeddings'])
            data = add_taxonomic_labels(data, taxonomy_dicts, label_encoders)
            filtered_data = filter_data(data, valid_labels, label_type)
            if filtered_data is not None:
                total_filtered_samples += len(filtered_data['embeddings'])
                save_filtered_data(filtered_data, os.path.join(output_dir, f'{split}_embeddings.npz'))
                log_with_time(f"Filtered {split} set - {len(filtered_data['embeddings'])} samples (out of {len(data['embeddings'])})")

    return total_samples, total_filtered_samples

def create_label_encoders(taxonomy_dicts):
    order_dict, family_dict, genus_dict = taxonomy_dicts

    label_encoders = {}
    order_encoder = LabelEncoder()
    order_encoder.fit(list(set(order_dict.values())))
    label_encoders['order'] = order_encoder

    family_encoder = LabelEncoder()
    family_encoder.fit(list(set(family_dict.values())))
    label_encoders['family'] = family_encoder

    genus_encoder = LabelEncoder()
    genus_encoder.fit(list(set(genus_dict.values())))
    label_encoders['genus'] = genus_encoder

    log_with_time(f"Created label encoders: Order ({len(order_encoder.classes_)}), " +
                 f"Family ({len(family_encoder.classes_)}), " +
                 f"Genus ({len(genus_encoder.classes_)})")
    return label_encoders

def process_full_sequences(full_seq_dir, taxonomy_dicts, label_encoders, thresholds):
    valid_labels = {}

    for label_type in ['order', 'family', 'genus']:
        log_with_time(f"\nDetermining valid {label_type} labels from full sequences...")
        label_counts = get_all_full_seq_label_counts(full_seq_dir, taxonomy_dicts, label_encoders, label_type)
        all_labels = np.array(list(label_counts.keys()))
        threshold = thresholds[label_type]
        valid = np.array([label for label in all_labels if label_counts.get(label, 0) >= threshold])

        log_with_time(f"Total unique {label_type} classes: {len(all_labels)}")
        log_with_time(f"Classes with >= {threshold} samples: {len(valid)}")

        valid_labels[label_type] = valid
        output_dir = os.path.join("/work/sgk270/genalm_task2_new", f'{label_type}_full_embed')

        log_with_time(f"Filtering full sequences for {label_type}...")
        total, filtered = filter_and_save_data(full_seq_dir, output_dir, taxonomy_dicts, label_encoders, valid, label_type)
        log_with_time(f"Full sequences - total: {total}, after filtering: {filtered}")

    return valid_labels

def process_contig_data(valid_labels, contig_dir, taxonomy_dicts, label_encoders, contig_lengths):
    for label_type in ['order', 'family', 'genus']:
        log_with_time(f"\nFiltering contigs for {label_type}...")

        for split in ['train', 'val']:
            input_dir = os.path.join(contig_dir)
            output_dir = os.path.join("/work/sgk270/genalm_task2_new", f'{label_type}_contig_embed')

            input_path = os.path.join(input_dir, f'{split}_embeddings.npz')
            if os.path.exists(input_path):
                log_with_time(f"Filtering {split} contigs for {label_type}...")
                os.makedirs(output_dir, exist_ok=True)
                data = load_embeddings(input_path)
                data = add_taxonomic_labels(data, taxonomy_dicts, label_encoders)
                filtered_data = filter_data(data, valid_labels[label_type], label_type)
                if filtered_data is not None:
                    save_filtered_data(filtered_data, os.path.join(output_dir, f'{split}_embeddings.npz'))
                    log_with_time(f"  {split} contigs - total: {len(data['embeddings'])}, after filtering: {len(filtered_data['embeddings'])}")

        for length in contig_lengths:
            length_input_dir = os.path.join(contig_dir, f'length_{length}')
            length_output_dir = os.path.join("/work/sgk270/genalm_task2_new", f'{label_type}_contig_embed', f'length_{length}')

            if os.path.exists(length_input_dir):
                test_input_path = os.path.join(length_input_dir, 'test_embeddings.npz')
                if os.path.exists(test_input_path):
                    log_with_time(f"Filtering test contigs of length {length} for {label_type}...")
                    os.makedirs(length_output_dir, exist_ok=True)
                    data = load_embeddings(test_input_path)
                    data = add_taxonomic_labels(data, taxonomy_dicts, label_encoders)
                    filtered_data = filter_data(data, valid_labels[label_type], label_type)
                    if filtered_data is not None:
                        save_filtered_data(filtered_data, os.path.join(length_output_dir, 'test_embeddings.npz'))
                        log_with_time(f"  test contigs (length {length}) - total: {len(data['embeddings'])}, after filtering: {len(filtered_data['embeddings'])}")

                encoder_data = {
                    'order': {'classes': label_encoders['order'].classes_},
                    'family': {'classes': label_encoders['family'].classes_},
                    'genus': {'classes': label_encoders['genus'].classes_}
                }
                np.save(os.path.join(length_output_dir, 'label_encoders.npy'), encoder_data)

def main():
    log_with_time("Starting filtering process...")

    FULL_SEQ_DIR = '/work/sgk270/genalm_task2_new/full_embed'
    CONTIG_DIR = '/work/sgk270/genalm_task2_new/contig_embed'
    CONTIG_LENGTHS = [500, 1000, 3000, 5000, 10000]
##################################################################################################
    THRESHOLDS = {'order': 40, 'family': 40, 'genus': 40}
##################################################################################################

    taxonomy_dicts = load_metadata()
    label_encoders = create_label_encoders(taxonomy_dicts)

    for label_type in ['order', 'family', 'genus']:
        output_dir = os.path.join("/work/sgk270/genalm_task2_new", f'{label_type}_contig_embed')
        os.makedirs(output_dir, exist_ok=True)
        encoder_data = {
            'order': {'classes': label_encoders['order'].classes_},
            'family': {'classes': label_encoders['family'].classes_},
            'genus': {'classes': label_encoders['genus'].classes_}
        }
        np.save(os.path.join(output_dir, 'label_encoders.npy'), encoder_data)

    valid_labels = process_full_sequences(FULL_SEQ_DIR, taxonomy_dicts, label_encoders, THRESHOLDS)
    process_contig_data(valid_labels, CONTIG_DIR, taxonomy_dicts, label_encoders, CONTIG_LENGTHS)

    log_with_time("Filtering complete!")

if __name__ == "__main__":
    main()
