import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

def calculate_metrics(predictions, true_labels, probabilities):
    unique_classes = np.unique(true_labels)
    accuracy_micro = accuracy_score(true_labels, predictions)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        true_labels, predictions, average='macro', labels=unique_classes)
    accuracy_per_class = [
        np.mean(predictions[true_labels == class_idx] == class_idx)
        for class_idx in unique_classes
    ]
    accuracy_macro = np.mean(accuracy_per_class)
    roc_values = []
    for class_idx in unique_classes:
        class_true = (true_labels == class_idx).astype(int)
        class_probs = probabilities[:, class_idx]
        if len(np.unique(class_true)) > 1:
            roc_values.append(roc_auc_score(class_true, class_probs))
    roc_macro = np.mean(roc_values) if roc_values else 0.0
    return {
        'accuracy_micro': accuracy_micro,
        'accuracy_macro': accuracy_macro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'roc_macro': roc_macro,
        'valid_samples': len(true_labels),
        'valid_orders': len(unique_classes)
    }

def plot_combined_training_history(output_dir, label_type="viral"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    colors = ['#1f77b4', '#ff7f0e']
    contigs_file = os.path.join(output_dir, f"{label_type}_combined_training_history.csv")
    full_file = os.path.join(output_dir, f"{label_type}_full_training_history.csv")
    if os.path.exists(contigs_file):
        df = pd.read_csv(contigs_file)
        ax1.plot(df['epoch'], df['train_loss'], label='Contigs Train', color=colors[0], alpha=0.8)
        ax1.plot(df['epoch'], df['val_loss'], label='Contigs Val', color=colors[0], linestyle='--')
        ax2.plot(df['epoch'], df['train_macro_acc'], label='Contigs Train', color=colors[0], alpha=0.8)
        ax2.plot(df['epoch'], df['val_macro_acc'], label='Contigs Val', color=colors[0], linestyle='--')
    if os.path.exists(full_file):
        df = pd.read_csv(full_file)
        ax1.plot(df['epoch'], df['train_loss'], label='Full Train', color=colors[1], linewidth=2)
        ax1.plot(df['epoch'], df['val_loss'], label='Full Val', color=colors[1], linestyle='--')
        ax2.plot(df['epoch'], df['train_macro_acc'], label='Full Train', color=colors[1], linewidth=2)
        ax2.plot(df['epoch'], df['val_macro_acc'], label='Full Val', color=colors[1], linestyle='--')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax2.set_title('Training and Validation Macro Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Macro Accuracy')
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.3)
    fig.suptitle(f'{label_type.capitalize()} Classification Training Metrics', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, f'{label_type}_training_curves_dnabert2.png'), dpi=300)
    plt.close()

def plot_metrics_by_length(results_df, output_dir, label_type="viral"):
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    metrics = ['accuracy_micro', 'accuracy_macro', 'precision_macro', 'recall_macro', 'f1_macro', 'roc_macro']
    titles = ['Micro-averaged Accuracy', 'Macro-averaged Accuracy',
              'Macro-averaged Precision', 'Macro-averaged Recall',
              'Macro-averaged F1', 'Macro-averaged ROC AUC']
    ylabels = ['Accuracy', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC']
    contig_lengths = sorted([x for x in results_df['length_group'].unique() if x != 'full'])
    x_labels = [f"{x}bp" for x in contig_lengths] + ['Full']
    x = np.arange(len(x_labels))
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(x)))
    for ax, metric, title, ylabel in zip(axes.flat, metrics, titles, ylabels):
        for i, label in enumerate(x_labels):
            key = int(label[:-2]) if label != 'Full' else 'full'
            value = results_df[results_df['length_group'] == key][metric].values[0]
            ax.bar(x[i], value, color=colors[i])
            ax.text(x[i], value, f"{value:.3f}", ha='center', va='bottom', fontsize=10)
        ax.set_title(title)
        ax.set_xlabel('Sequence Length')
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45)
        if metric != 'roc_macro':
            ax.set_ylim(0, 1)
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    fig.suptitle(f'{label_type.capitalize()} Classification Performance by Sequence Length (Test Set)', y=1.02, fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{label_type}_metrics_by_length_dnabert2.png'), dpi=300)
    plt.close()

def main():
    label_type = "viral"
    output_dir = "viral_output"
    contig_lengths = [500, 1000, 3000, 5000, 10000]
    results = []
    for length in contig_lengths:
        path = os.path.join(output_dir, f"{label_type}_contigs_{length}_eval_results.npy")
        if os.path.exists(path):
            try:
                data = np.load(path, allow_pickle=True).item()
                m = calculate_metrics(data['predictions'], data['true_labels'], data['probabilities'])
                m['dataset'] = 'contigs'
                m['length_group'] = length
                results.append(m)
                print(f"Processed {length}bp contigs")
            except Exception as e:
                print(f"Error processing {length}bp contigs: {e}")
    full_path = os.path.join(output_dir, f"{label_type}_full_eval_results.npy")
    if os.path.exists(full_path):
        try:
            data = np.load(full_path, allow_pickle=True).item()
            m = calculate_metrics(data['predictions'], data['true_labels'], data['probabilities'])
            m['dataset'] = 'full'
            m['length_group'] = 'full'
            results.append(m)
            print("Processed full sequences")
        except Exception as e:
            print(f"Error processing full sequences: {e}")
    if results:
        df = pd.DataFrame(results)
        plot_combined_training_history(output_dir, label_type)
        plot_metrics_by_length(df, output_dir, label_type)
        print("\nNumerical Results:")
        print(df.to_string(index=False))
        print("\nNumber of samples per group:")
        print(df[['dataset', 'length_group', 'valid_samples']].to_string(index=False))
        print("\nNumber of classes per group:")
        print(df[['dataset', 'length_group', 'valid_orders']].to_string(index=False))
    else:
        print("No result files found.")

if __name__ == "__main__":
    main()
