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

    order_dict, family_dict, genus_dict = {}, {}, {}
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
    return {'embeddings': data['embeddings'], 'sequence_ids': data['sequence_ids']}

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

    data['order_labels'], data['order_text'] = encode_labels(sequence_ids, order_dict, label_encoders['order'], 'order')
    data['family_labels'], data['family_text'] = encode_labels(sequence_ids, family_dict, label_encoders['family'], 'family')
    data['genus_labels'], data['genus_text'] = encode_labels(sequence_ids, genus_dict, label_encoders['genus'], 'genus')

    log_with_time(f"Matched taxonomic labels: Order ({np.sum(data['order_labels'] >= 0)}), Family ({np.sum(data['family_labels'] >= 0)}), Genus ({np.sum(data['genus_labels'] >= 0)})")
    return data

def get_all_full_seq_label_counts(full_seq_dir, taxonomy_dicts, label_encoders, label_type):
    taxonomy_dict = {'order': taxonomy_dicts[0], 'family': taxonomy_dicts[1], 'genus': taxonomy_dicts[2]}[label_type]
    label_encoder = label_encoders[label_type]
    counter = Counter()

    for split in ['train', 'val', 'test']:
        path = os.path.join(full_seq_dir, f'{split}_embeddings.npz')
        if os.path.exists(path):
            data = load_embeddings(path)
            for seq_id in data['sequence_ids']:
                if seq_id in taxonomy_dict:
                    try:
                        label = label_encoder.transform([taxonomy_dict[seq_id]])[0]
                        counter[label] += 1
                    except:
                        pass
    return counter

def filter_data(data, valid_labels, label_type):
    labels = data[f'{label_type}_labels']
    mask = (labels >= 0) & np.isin(labels, valid_labels)
    if not np.any(mask):
        log_with_time(f"Warning: No valid samples found after filtering for {label_type}")
        return None
    return {k: v[mask] for k, v in data.items() if v is not None}

def save_filtered_data(data, path):
    if data is None:
        log_with_time(f"No data to save to {path}")
        return
    log_with_time(f"Saving filtered data to {path} ({len(data['embeddings'])} samples)")
    np.savez(path, **data)

def filter_and_save_data(input_dir, output_dir, taxonomy_dicts, label_encoders, valid_labels, label_type):
    os.makedirs(output_dir, exist_ok=True)
    encoder_data = {lt: {'classes': le.classes_} for lt, le in label_encoders.items()}
    np.save(os.path.join(output_dir, 'label_encoders.npy'), encoder_data)

    for split in ['train', 'val', 'test']:
        path = os.path.join(input_dir, f'{split}_embeddings.npz')
        if os.path.exists(path):
            data = load_embeddings(path)
            data = add_taxonomic_labels(data, taxonomy_dicts, label_encoders)
            filtered = filter_data(data, valid_labels, label_type)
            if filtered:
                save_filtered_data(filtered, os.path.join(output_dir, f'{split}_embeddings.npz'))

def create_label_encoders(taxonomy_dicts):
    return {
        'order': LabelEncoder().fit(list(set(taxonomy_dicts[0].values()))),
        'family': LabelEncoder().fit(list(set(taxonomy_dicts[1].values()))),
        'genus': LabelEncoder().fit(list(set(taxonomy_dicts[2].values())))
    }

def process_contig_data(valid_labels, contig_dir, taxonomy_dicts, label_encoders, contig_lengths, output_base):
    for label_type in ['order', 'family', 'genus']:
        output_dir = os.path.join(output_base, f'{label_type}_contig_embed')
        os.makedirs(output_dir, exist_ok=True)
        encoder_data = {lt: {'classes': le.classes_} for lt, le in label_encoders.items()}
        np.save(os.path.join(output_dir, 'label_encoders.npy'), encoder_data)

        for split in ['train', 'val']:
            path = os.path.join(contig_dir, f'{split}_embeddings.npz')
            if os.path.exists(path):
                data = load_embeddings(path)
                data = add_taxonomic_labels(data, taxonomy_dicts, label_encoders)
                filtered = filter_data(data, valid_labels[label_type], label_type)
                if filtered:
                    save_filtered_data(filtered, os.path.join(output_dir, f'{split}_embeddings.npz'))

        for length in contig_lengths:
            length_dir = os.path.join(contig_dir, f'length_{length}')
            test_path = os.path.join(length_dir, 'test_embeddings.npz')
            if os.path.exists(test_path):
                output_length_dir = os.path.join(output_dir, f'length_{length}')
                os.makedirs(output_length_dir, exist_ok=True)
                data = load_embeddings(test_path)
                data = add_taxonomic_labels(data, taxonomy_dicts, label_encoders)
                filtered = filter_data(data, valid_labels[label_type], label_type)
                if filtered:
                    save_filtered_data(filtered, os.path.join(output_length_dir, 'test_embeddings.npz'))
                np.save(os.path.join(output_length_dir, 'label_encoders.npy'), encoder_data)

def main():
    log_with_time("Starting filtering process for ProkBERT Task 2...")
    BASE_DIR = '/work/sgk270/prokbert_task2_new'
    FULL_SEQ_DIR = os.path.join(BASE_DIR, 'full_embed')
    CONTIG_DIR = os.path.join(BASE_DIR, 'contig_embed')
    CONTIG_LENGTHS = [500, 1000, 3000, 5000, 10000]
##################################################################################################
    THRESHOLDS = {'order': 40, 'family': 40, 'genus': 40}
##################################################################################################

    taxonomy_dicts = load_metadata()
    label_encoders = create_label_encoders(taxonomy_dicts)

    valid_labels = {}
    for label_type in ['order', 'family', 'genus']:
        log_with_time(f"\nFiltering full sequences for {label_type}...")
        counts = get_all_full_seq_label_counts(FULL_SEQ_DIR, taxonomy_dicts, label_encoders, label_type)
        threshold = THRESHOLDS[label_type]
        valid = np.array([lbl for lbl, cnt in counts.items() if cnt >= threshold])
        valid_labels[label_type] = valid
        output_dir = os.path.join(BASE_DIR, f'{label_type}_full_embed')
        filter_and_save_data(FULL_SEQ_DIR, output_dir, taxonomy_dicts, label_encoders, valid, label_type)

    log_with_time("\nFiltering contig sequences...")
    process_contig_data(valid_labels, CONTIG_DIR, taxonomy_dicts, label_encoders, CONTIG_LENGTHS, BASE_DIR)

    log_with_time("Filtering complete for ProkBERT Task 2!")

if __name__ == "__main__":
    main()
