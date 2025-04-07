import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

def calculate_metrics(predictions, true_labels, probabilities):
    """Calculate evaluation metrics for given predictions and true labels."""
    # Get all unique classes
    unique_classes = np.unique(true_labels)
    
    # Calculate metrics for all classes
    accuracy_micro = accuracy_score(true_labels, predictions)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        true_labels, predictions,
        average='macro',
        labels=unique_classes
    )
    accuracy_per_class = [
        np.mean(predictions[true_labels == class_idx] == class_idx)
        for class_idx in unique_classes
    ]
    accuracy_macro = np.mean(accuracy_per_class)

    # Calculate macro ROC AUC
    roc_values = []
    for class_idx in unique_classes:
        class_true = (true_labels == class_idx).astype(int)
        class_probs = probabilities[:, class_idx]
        if len(np.unique(class_true)) > 1:  # Avoid ROC calculation if only one class present
            roc_values.append(roc_auc_score(class_true, class_probs))
    roc_macro = np.mean(roc_values) if roc_values else 0.0

    # Get sample and class counts
    sample_count = len(true_labels)
    class_count = len(unique_classes)

    return {
        'accuracy_micro': accuracy_micro,
        'accuracy_macro': accuracy_macro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'roc_macro': roc_macro,
        'valid_samples': sample_count,
        'valid_families': class_count  # Changed from valid_orders to valid_families
    }

def plot_combined_training_history(output_dir, label_type="family"):
    """Plot training histories for combined contigs and full sequences in a single figure."""
    # Create a combined figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    # Colors for different datasets
    colors = ['#1f77b4', '#ff7f0e']  # Blue for contigs, orange for full
    
    # Load contigs combined training history
    contigs_history_file = os.path.join(output_dir, f"{label_type}_combined_training_history.csv")
    if os.path.exists(contigs_history_file):
        history_df = pd.read_csv(contigs_history_file)
        
        # Plot loss on the first subplot
        ax1.plot(history_df['epoch'], history_df['train_loss'], 
                label=f'Contigs Train',
                color=colors[0], 
                alpha=0.8)
        
        ax1.plot(history_df['epoch'], history_df['val_loss'], 
                label=f'Contigs Val',
                color=colors[0],
                alpha=0.8,
                linestyle='--')
        
        # Plot accuracy on the second subplot
        ax2.plot(history_df['epoch'], history_df['train_macro_acc'], 
                label=f'Contigs Train',
                color=colors[0], 
                alpha=0.8)
        
        ax2.plot(history_df['epoch'], history_df['val_macro_acc'], 
                label=f'Contigs Val',
                color=colors[0],
                alpha=0.8,
                linestyle='--')
    
    # Load full sequences training history
    full_history_file = os.path.join(output_dir, f"{label_type}_full_training_history.csv")
    if os.path.exists(full_history_file):
        full_history_df = pd.read_csv(full_history_file)
        
        # Plot loss on the first subplot
        ax1.plot(full_history_df['epoch'], full_history_df['train_loss'], 
                label=f'Full Train',
                color=colors[1], 
                linewidth=2)
        
        ax1.plot(full_history_df['epoch'], full_history_df['val_loss'], 
                label=f'Full Val',
                color=colors[1],
                linewidth=2,
                linestyle='--')
        
        # Plot accuracy on the second subplot
        ax2.plot(full_history_df['epoch'], full_history_df['train_macro_acc'], 
                label=f'Full Train',
                color=colors[1], 
                linewidth=2)
        
        ax2.plot(full_history_df['epoch'], full_history_df['val_macro_acc'], 
                label=f'Full Val',
                color=colors[1],
                linewidth=2,
                linestyle='--')
    
    # Customize the first subplot (Loss)
    ax1.set_title('Training and Validation Loss', fontsize=14)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # Customize the second subplot (Accuracy)
    ax2.set_title('Training and Validation Macro Accuracy', fontsize=14)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Macro Accuracy', fontsize=12)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    # Set y-axis limits for accuracy subplot
    ax2.set_ylim(0, 1)
    
    # Add a global title
    fig.suptitle(f'{label_type.capitalize()} Classification Training Metrics', fontsize=16)
    
    # Save the combined figure
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
    plt.savefig(os.path.join(output_dir, f'{label_type}_training_curves_dnabert2.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_metrics_by_length(results_df, output_dir, label_type="family"):
    """Create bar plots for metrics comparing different sequence lengths."""
    # Create bar plots for metrics
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # Metrics and labels
    metrics = ['accuracy_micro', 'accuracy_macro', 'precision_macro', 'recall_macro', 'f1_macro', 'roc_macro']
    titles = ['Micro-averaged Accuracy', 'Macro-averaged Accuracy', 
              'Macro-averaged Precision', 'Macro-averaged Recall', 
              'Macro-averaged F1', 'Macro-averaged ROC AUC']
    ylabels = ['Accuracy', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC']

    # Get x-axis labels based on available data
    contig_length_groups = sorted([group for group in results_df['length_group'].unique() if group != 'full'])
    x_labels = [f"{length}bp" for length in contig_length_groups] + ['Full']
    x_positions = np.arange(len(x_labels))
    
    # Generate color map
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(x_labels)))

    # Create each subplot for metrics
    for ax, metric, title, ylabel in zip(axes.flat, metrics, titles, ylabels):
        for i, length_group in enumerate(x_labels):
            if length_group == 'Full':
                # Get full sequences value
                if 'full' in results_df['length_group'].values:
                    value = results_df[results_df['length_group'] == 'full'][metric].values[0]
                    ax.bar(x_positions[i], value, width=0.8, color=colors[i], alpha=0.8)
                    ax.text(x_positions[i], value, f'{value:.3f}', ha='center', va='bottom', fontsize=10)
            else:
                # Get contig length value
                length = int(length_group[:-2])  # Remove "bp" suffix
                if length in results_df['length_group'].values:
                    value = results_df[results_df['length_group'] == length][metric].values[0]
                    ax.bar(x_positions[i], value, width=0.8, color=colors[i], alpha=0.8)
                    ax.text(x_positions[i], value, f'{value:.3f}', ha='center', va='bottom', fontsize=10)
                    
        # Customize plot
        ax.set_title(title, fontsize=14, pad=10)
        ax.set_xlabel('Sequence Length', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=45, ha='center')
        if metric != 'roc_macro':
            ax.set_ylim(0, 1)
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)

    # Adjust layout and save plot
    plt.tight_layout()
    fig.suptitle(f'{label_type.capitalize()} Classification Performance by Sequence Length (Test Set)', y=1.02, fontsize=16)
    plt.savefig(os.path.join(output_dir, f'{label_type}_metrics_by_length_dnabert2.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Configure parameters
    label_type = 'family'  # Changed from 'order' to 'family'
    output_dir = 'family_output'  # Changed from 'order_output' to 'family_output'
    contig_lengths = [500, 1000, 3000, 5000, 10000]
    
    results = []
    
    # Process contig lengths
    for length in contig_lengths:
        eval_file = os.path.join(output_dir, f'{label_type}_contigs_{length}_eval_results.npy')
        
        if os.path.exists(eval_file):
            try:
                eval_results = np.load(eval_file, allow_pickle=True).item()
                metrics = calculate_metrics(
                    eval_results['predictions'], 
                    eval_results['true_labels'], 
                    eval_results['probabilities']
                )
                if metrics:
                    metrics['dataset'] = 'contigs'
                    metrics['length_group'] = length
                    results.append(metrics)
                    print(f"Processed {length}bp contigs")
                else:
                    print(f"No metrics calculated for {length}bp contigs")
            except Exception as e:
                print(f"Error processing {length}bp contigs: {e}")
        else:
            print(f"Evaluation file not found for {length}bp contigs: {eval_file}")
    
    # Process full sequences
    full_eval_file = os.path.join(output_dir, f'{label_type}_full_eval_results.npy')
    if os.path.exists(full_eval_file):
        try:
            eval_results = np.load(full_eval_file, allow_pickle=True).item()
            metrics = calculate_metrics(
                eval_results['predictions'], 
                eval_results['true_labels'], 
                eval_results['probabilities']
            )
            if metrics:
                metrics['dataset'] = 'full'
                metrics['length_group'] = 'full'
                results.append(metrics)
                print("Processed full sequences")
            else:
                print("No metrics calculated for full sequences")
        except Exception as e:
            print(f"Error processing full sequences: {e}")
    else:
        print(f"Evaluation file not found for full sequences: {full_eval_file}")
    
    # Convert results to DataFrame
    if results:
        results_df = pd.DataFrame(results)
        
        # Generate plots
        plot_combined_training_history(output_dir, label_type)
        plot_metrics_by_length(results_df, output_dir, label_type)
        
        # Print numerical results
        print("\nNumerical Results:")
        print(results_df.to_string(index=False))
        
        # Print sample distribution for each length group
        print("\nNumber of samples in each length group:")
        print(results_df[['dataset', 'length_group', 'valid_samples']].to_string(index=False))
        
        # Print number of families for each length group
        print("\nNumber of families in each length group:")
        print(results_df[['dataset', 'length_group', 'valid_families']].to_string(index=False))
    else:
        print("No results found to analyze")

if __name__ == "__main__":
    main()