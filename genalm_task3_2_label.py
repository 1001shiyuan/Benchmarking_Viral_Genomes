import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import datetime

def log_with_time(msg):
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def load_metadata():
    path = "/work/sgk270/dataset_for_benchmarking/combine3/final_dataset/combined_metadata.csv"
    log_with_time(f"Loading metadata from {path}")
    metadata = pd.read_csv(path, sep='\t')
    lifestyle_dict = {row['sequence_id']: row['lifestyle'] for _, row in metadata.iterrows() if pd.notna(row['lifestyle'])}
    log_with_time(f"Loaded {len(lifestyle_dict)} lifestyle annotations")
    return lifestyle_dict

def load_embeddings(path):
    log_with_time(f"Loading embeddings from {path}")
    data = np.load(path, allow_pickle=True)
    return {
        'embeddings': data['embeddings'],
        'sequence_ids': data['sequence_ids']
    }

def add_lifestyle_labels(data, lifestyle_dict, label_encoder):
    sequence_ids = data['sequence_ids']
    lifestyles = np.array([lifestyle_dict.get(seq_id, 'NA') for seq_id in sequence_ids], dtype=object)
    mask = lifestyles != 'NA'

    labels = np.full(len(sequence_ids), -1, dtype=int)
    labels[mask] = label_encoder.transform(lifestyles[mask])

    data['lifestyle_labels'] = labels
    data['lifestyle_text'] = lifestyles

    log_with_time(f"Labeled {np.sum(mask)} out of {len(sequence_ids)} sequences with lifestyle labels")
    return data

def save_labeled_data(data, path):
    log_with_time(f"Saving labeled data to {path}")
    np.savez(path, **data)

def process_folder(input_dir, output_dir, lifestyle_dict, label_encoder):
    os.makedirs(output_dir, exist_ok=True)
    for split in ['train', 'val', 'test']:
        input_path = os.path.join(input_dir, f'{split}_embeddings.npz')
        output_path = os.path.join(output_dir, f'{split}_embeddings.npz')
        if os.path.exists(input_path):
            data = load_embeddings(input_path)
            data = add_lifestyle_labels(data, lifestyle_dict, label_encoder)
            save_labeled_data(data, output_path)

def process_length_dirs(base_input_dir, base_output_dir, lifestyle_dict, label_encoder, lengths):
    for length in lengths:
        subdir = f"length_{length}"
        input_path = os.path.join(base_input_dir, subdir, 'test_embeddings.npz')
        output_dir = os.path.join(base_output_dir, subdir)
        output_path = os.path.join(output_dir, 'test_embeddings.npz')
        if os.path.exists(input_path):
            os.makedirs(output_dir, exist_ok=True)
            data = load_embeddings(input_path)
            data = add_lifestyle_labels(data, lifestyle_dict, label_encoder)
            save_labeled_data(data, output_path)

def main():
    log_with_time("Starting lifestyle label annotation for GenaLM Task 3...")

    CONTIG_INPUT = "/work/sgk270/genalm_task3_new/contig_embed"
    FULL_INPUT = "/work/sgk270/genalm_task3_new/full_embed"
    CONTIG_OUTPUT = "/work/sgk270/genalm_task3_new/contig_embed_labeled"
    FULL_OUTPUT = "/work/sgk270/genalm_task3_new/full_embed_labeled"
    CONTIG_LENGTHS = [500, 1000, 3000, 5000, 10000]

    lifestyle_dict = load_metadata()

    lifestyle_encoder = LabelEncoder()
    lifestyle_encoder.fit(list(set(lifestyle_dict.values())))
    log_with_time(f"Found {len(lifestyle_encoder.classes_)} lifestyle classes:")
    for label in lifestyle_encoder.classes_:
        print(f"  - {label}", flush=True)

    encoder_data = {'classes': lifestyle_encoder.classes_}
    os.makedirs(CONTIG_OUTPUT, exist_ok=True)
    os.makedirs(FULL_OUTPUT, exist_ok=True)
    np.save(os.path.join(CONTIG_OUTPUT, 'label_encoder.npy'), encoder_data)
    np.save(os.path.join(FULL_OUTPUT, 'label_encoder.npy'), encoder_data)

    process_folder(CONTIG_INPUT, CONTIG_OUTPUT, lifestyle_dict, lifestyle_encoder)
    process_length_dirs(CONTIG_INPUT, CONTIG_OUTPUT, lifestyle_dict, lifestyle_encoder, CONTIG_LENGTHS)
    process_folder(FULL_INPUT, FULL_OUTPUT, lifestyle_dict, lifestyle_encoder)

    log_with_time("Lifestyle label annotation complete.")

if __name__ == "__main__":
    main()
