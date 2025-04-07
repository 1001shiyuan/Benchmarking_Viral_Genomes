import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

def calculate_metrics(predictions, true_labels, probabilities):
    unique_classes = np.unique(true_labels)
    accuracy_micro = accuracy_score(true_labels, predictions)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        true_labels, predictions, average='macro', labels=unique_classes
    )

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
        'valid_genera': len(unique_classes)
    }

def plot_combined_training_history(output_dir, label_type="genus"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    colors = ['#1f77b4', '#ff7f0e']

    contigs_file = os.path.join(output_dir, f"{label_type}_combined_training_history.csv")
    if os.path.exists(contigs_file):
        df = pd.read_csv(contigs_file)
        ax1.plot(df['epoch'], df['train_loss'], label='Contigs Train', color=colors[0])
        ax1.plot(df['epoch'], df['val_loss'], label='Contigs Val', linestyle='--', color=colors[0])
        ax2.plot(df['epoch'], df['train_macro_acc'], label='Contigs Train', color=colors[0])
        ax2.plot(df['epoch'], df['val_macro_acc'], label='Contigs Val', linestyle='--', color=colors[0])

    full_file = os.path.join(output_dir, f"{label_type}_full_training_history.csv")
    if os.path.exists(full_file):
        df = pd.read_csv(full_file)
        ax1.plot(df['epoch'], df['train_loss'], label='Full Train', color=colors[1])
        ax1.plot(df['epoch'], df['val_loss'], label='Full Val', linestyle='--', color=colors[1])
        ax2.plot(df['epoch'], df['train_macro_acc'], label='Full Train', color=colors[1])
        ax2.plot(df['epoch'], df['val_macro_acc'], label='Full Val', linestyle='--', color=colors[1])

    ax1.set_title("Training and Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.legend()

    ax2.set_title("Training and Validation Macro Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Macro Accuracy")
    ax2.set_ylim(0, 1)
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.legend()

    fig.suptitle("Genus Classification Training Curves", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, f"{label_type}_training_curves_prokbert.png"), dpi=300)
    plt.close()

def plot_metrics_by_length(results_df, output_dir, label_type="genus"):
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    metrics = ['accuracy_micro', 'accuracy_macro', 'precision_macro', 'recall_macro', 'f1_macro', 'roc_macro']
    titles = ['Micro-averaged Accuracy', 'Macro-averaged Accuracy',
              'Macro-averaged Precision', 'Macro-averaged Recall',
              'Macro-averaged F1', 'Macro-averaged ROC AUC']
    ylabels = ['Accuracy', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC']

    contig_lengths = sorted([group for group in results_df['length_group'].unique() if group != 'full'])
    x_labels = [f"{length}bp" for length in contig_lengths] + ['Full']
    x_positions = np.arange(len(x_labels))
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(x_labels)))

    for ax, metric, title, ylabel in zip(axes.flat, metrics, titles, ylabels):
        for i, label in enumerate(x_labels):
            if label == 'Full':
                value = results_df[results_df['length_group'] == 'full'][metric].values[0]
            else:
                length = int(label.replace('bp', ''))
                value = results_df[results_df['length_group'] == length][metric].values[0]
            ax.bar(x_positions[i], value, color=colors[i], width=0.8, alpha=0.9)
            ax.text(x_positions[i], value, f"{value:.3f}", ha='center', va='bottom', fontsize=10)

        ax.set_title(title)
        ax.set_xlabel("Sequence Length")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=45)
        if metric != 'roc_macro':
            ax.set_ylim(0, 1)
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)

    fig.suptitle("Genus Classification Performance by Sequence Length", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{label_type}_metrics_by_length_prokbert.png"), dpi=300)
    plt.close()

def main():
    label_type = 'genus'
    output_dir = 'genus_output'  # Local folder with ProkBERT genus results
    contig_lengths = [500, 1000, 3000, 5000, 10000]
    results = []

    for length in contig_lengths:
        eval_file = os.path.join(output_dir, f"{label_type}_contigs_{length}_eval_results.npy")
        if os.path.exists(eval_file):
            data = np.load(eval_file, allow_pickle=True).item()
            metrics = calculate_metrics(data['predictions'], data['true_labels'], data['probabilities'])
            metrics['dataset'] = 'contigs'
            metrics['length_group'] = length
            results.append(metrics)
            print(f"Processed contigs {length}bp")
        else:
            print(f"Missing: {eval_file}")

    full_file = os.path.join(output_dir, f"{label_type}_full_eval_results.npy")
    if os.path.exists(full_file):
        data = np.load(full_file, allow_pickle=True).item()
        metrics = calculate_metrics(data['predictions'], data['true_labels'], data['probabilities'])
        metrics['dataset'] = 'full'
        metrics['length_group'] = 'full'
        results.append(metrics)
        print("Processed full sequences")
    else:
        print(f"Missing: {full_file}")

    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(output_dir, f"{label_type}_metrics_summary.csv"), index=False)
        plot_combined_training_history(output_dir, label_type)
        plot_metrics_by_length(results_df, output_dir, label_type)

        print("\nNumerical Results:")
        print(results_df.to_string(index=False))

        print("\nSample Counts by Group:")
        print(results_df[['dataset', 'length_group', 'valid_samples']].to_string(index=False))

        print("\nClass Count by Group:")
        print(results_df[['dataset', 'length_group', 'valid_genera']].to_string(index=False))
    else:
        print("No results to process.")

if __name__ == "__main__":
    main()
