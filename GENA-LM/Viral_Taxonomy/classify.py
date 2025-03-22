import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import os
import numpy as np
import pandas as pd
import sys
import traceback
from argparse import ArgumentParser
import tempfile

class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes, dropout_prob=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.dropout(x)
        return self.classifier(x)

def evaluate_model(model, dataset, batch_size, classes):
    """Evaluate model and compute both micro and macro accuracies."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(next(model.parameters()).device), labels.to(next(model.parameters()).device)

            logits = model(features)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate micro (overall) accuracy
    avg_loss = total_loss / len(dataloader)  # Average loss across all batches
    micro_accuracy = (all_preds == all_labels).mean()
    
    # Calculate macro accuracy (mean of per-class accuracies)
    unique_classes = np.unique(all_labels)
    class_accuracies = []
    for cls in unique_classes:
        mask = all_labels == cls
        if np.sum(mask) > 0:  # Only if class is present
            class_acc = (all_preds[mask] == all_labels[mask]).mean()
            class_accuracies.append(class_acc)
    macro_accuracy = np.mean(class_accuracies)
    
    results = {
        'predictions': all_preds,
        'true_labels': all_labels,
        'probabilities': np.array(all_probs),
        'n_samples': len(all_labels),
        'classes': classes,
        'macro_accuracy': macro_accuracy,
        'class_accuracies': dict(zip(unique_classes, class_accuracies))
    }

    return results, avg_loss, micro_accuracy, macro_accuracy

def load_embeddings(embeddings_dir, filename):
    """Load dataset and handle missing files."""
    file_path = os.path.join(embeddings_dir, filename)
    if not os.path.exists(file_path):
        print(f"ERROR: Missing {file_path}")
        sys.exit(1)
    
    data = np.load(file_path)
    
    # Select the appropriate label type based on the embeddings_dir
    label_type = 'order'
    if 'family' in embeddings_dir.lower():
        label_type = 'family'
    elif 'genus' in embeddings_dir.lower():
        label_type = 'genus'
    
    return torch.FloatTensor(data['embeddings']), torch.LongTensor(data[f'{label_type}_labels'])

def get_all_unique_labels(embeddings_dir, train_file, val_file):
    """Get unique labels from train and validation splits."""
    all_labels = []
    
    # Get labels from train and val
    _, train_labels = load_embeddings(embeddings_dir, train_file)
    all_labels.extend(train_labels.numpy())
    _, val_labels = load_embeddings(embeddings_dir, val_file)
    all_labels.extend(val_labels.numpy())
    
    unique_labels = np.unique(all_labels)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    return unique_labels, label_map

def add_test_labels_to_map(length_test_dirs, label_map, label_type, unique_labels):
    """Add test labels from length-specific directories to the label map."""
    for test_dir in length_test_dirs:
        test_path = os.path.join(test_dir, "test_embeddings.npz")
        if os.path.exists(test_path):
            data = np.load(test_path)
            test_labels = data[f'{label_type}_labels']
            for label in test_labels:
                if label not in label_map:
                    new_idx = len(label_map)
                    label_map[label] = new_idx
                    unique_labels = np.append(unique_labels, label)
    
    return unique_labels, label_map

def remap_labels(labels, label_map):
    """Remap labels using the consistent label map."""
    return torch.tensor([label_map[label.item()] for label in labels])

def save_checkpoint(model, optimizer, epoch, val_loss, temp_dir, model_prefix):
    """Save model checkpoint to temporary directory."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'val_loss': val_loss
    }
    torch.save(checkpoint, os.path.join(temp_dir, f'{model_prefix}_checkpoint.pt'))

def load_checkpoint(model, optimizer, temp_dir, model_prefix):
    """Load model checkpoint from temporary directory."""
    checkpoint = torch.load(os.path.join(temp_dir, f'{model_prefix}_checkpoint.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['val_loss']

def train_model(model, optimizer, train_loader, val_dataset, device, num_epochs, early_stop_patience, batch_size, temp_dir, model_prefix, valid_classes):
    """Train model and return best model and performance metrics."""
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float('inf')
    no_improve_epochs = 0
    train_losses, val_losses = [], []
    train_micro_accs, val_micro_accs = [], []
    train_macro_accs, val_macro_accs = [], []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        all_preds, all_labels = [], []

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # Calculate average loss across all batches
        avg_train_loss = total_loss / len(train_loader)
        
        # Calculate train accuracies
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        train_micro_acc = (all_preds == all_labels).mean()
        
        unique_classes = np.unique(all_labels)
        class_accuracies = []
        for cls in unique_classes:
            mask = all_labels == cls
            if np.sum(mask) > 0:
                class_acc = (all_preds[mask] == all_labels[mask]).mean()
                class_accuracies.append(class_acc)
        train_macro_acc = np.mean(class_accuracies)

        train_losses.append(avg_train_loss)
        train_micro_accs.append(train_micro_acc)
        train_macro_accs.append(train_macro_acc)

        val_results, val_loss, val_micro_acc, val_macro_acc = evaluate_model(model, val_dataset, batch_size, valid_classes)
        val_losses.append(val_loss)
        val_micro_accs.append(val_micro_acc)
        val_macro_accs.append(val_macro_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epochs = 0
            save_checkpoint(model, optimizer, epoch, val_loss, temp_dir, model_prefix)
        else:
            no_improve_epochs += 1

        # Use provided early stopping patience
        if no_improve_epochs >= early_stop_patience:
            break

    # Load best model
    best_epoch, best_val_loss = load_checkpoint(model, optimizer, temp_dir, model_prefix)
    
    training_history = {
        'epoch': range(1, len(train_losses) + 1),
        'train_loss': train_losses,
        'train_macro_acc': train_macro_accs,
        'val_loss': val_losses,
        'val_macro_acc': val_macro_accs
    }
    
    return model, best_epoch, best_val_loss, training_history

def train_and_evaluate(embeddings_dir, output_dir, model_prefix, batch_size, learning_rate, num_epochs, early_stop_patience, label_type="order", length_test_dirs=None):
    """Train and evaluate a model for a given dataset type."""
    os.makedirs(output_dir, exist_ok=True)
    temp_dir = tempfile.mkdtemp()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining model for {model_prefix} using device: {device}")

    print("\nLoading training data...")
    train_embeddings, train_labels = load_embeddings(embeddings_dir, "train_embeddings.npz")
    
    label_info_path = os.path.join(embeddings_dir, 'label_encoders.npy')
    if not os.path.exists(label_info_path):
        print(f"WARNING: Missing {label_info_path}, looking in parent directory")
        parent_dir = os.path.dirname(embeddings_dir)
        label_info_path = os.path.join(parent_dir, 'label_encoders.npy')
        if not os.path.exists(label_info_path):
            print(f"ERROR: Missing {label_info_path}")
            sys.exit(1)
    
    label_encoders = np.load(label_info_path, allow_pickle=True).item()
    
    # Get unique labels from train and val
    unique_labels, label_map = get_all_unique_labels(embeddings_dir, "train_embeddings.npz", "val_embeddings.npz")
    
    # If length_test_dirs is provided, add labels from test sets
    if length_test_dirs is not None:
        unique_labels, label_map = add_test_labels_to_map(length_test_dirs, label_map, label_type, unique_labels)
    
    valid_classes = label_encoders[label_type.capitalize()]['classes'][unique_labels]
    num_classes = len(unique_labels)
    print(f"Number of unique classes for {model_prefix}: {num_classes}")

    train_labels_mapped = remap_labels(train_labels, label_map)

    print("\nLoading validation data...")
    val_embeddings, val_labels = load_embeddings(embeddings_dir, "val_embeddings.npz")
    val_labels_mapped = remap_labels(val_labels, label_map)

    input_dim = train_embeddings.shape[1]
    
    # Create model and optimizer with fixed hyperparameters
    model = Classifier(input_dim, num_classes, dropout_prob=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(train_embeddings, train_labels_mapped)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(val_embeddings, val_labels_mapped)
    
    # Train the model
    model, best_epoch, best_val_loss, training_history = train_model(
        model, optimizer, train_loader, val_dataset, device, 
        num_epochs, early_stop_patience, batch_size, 
        temp_dir, model_prefix, valid_classes
    )
    
    print(f"Best epoch: {best_epoch+1}, Validation loss: {best_val_loss:.4f}")
    
    # Save the training history with streamlined output
    pd.DataFrame({
        'epoch': training_history['epoch'],
        'train_loss': training_history['train_loss'],
        'train_macro_acc': training_history['train_macro_acc'],
        'val_loss': training_history['val_loss'],
        'val_macro_acc': training_history['val_macro_acc']
    }).to_csv(os.path.join(output_dir, f'{label_type}_{model_prefix}_training_history.csv'), index=False)
    
    # If there are length-specific test directories, evaluate on each
    if length_test_dirs:
        for test_dir in length_test_dirs:
            length_name = os.path.basename(test_dir)
            test_file = os.path.join(test_dir, "test_embeddings.npz")
            
            if os.path.exists(test_file):
                print(f"\nEvaluating {model_prefix} on test set from {length_name}...")
                
                test_embeddings, test_labels = load_embeddings(test_dir, "test_embeddings.npz")
                test_labels_mapped = remap_labels(test_labels, label_map)
                test_dataset = TensorDataset(test_embeddings, test_labels_mapped)
                
                test_results, test_loss, test_micro_acc, test_macro_acc = evaluate_model(
                    model, test_dataset, batch_size, valid_classes
                )
                print(f"Test ({length_name}): Loss: {test_loss:.4f}, Micro Acc: {test_micro_acc:.4f}, Macro Acc: {test_macro_acc:.4f}")
                
                # Save only the necessary data for visualization
                simplified_results = {
                    'predictions': test_results['predictions'],
                    'true_labels': test_results['true_labels'],
                    'probabilities': test_results['probabilities'],
                    'classes': test_results['classes']
                }
                
                length_id = int(length_name.split('_')[1])
                np.save(os.path.join(output_dir, f'{label_type}_contigs_{length_id}_eval_results.npy'), simplified_results)
    else:
        # For full sequences, test data is in the same directory
        test_file = os.path.join(embeddings_dir, "test_embeddings.npz")
        if os.path.exists(test_file):
            print(f"\nEvaluating {model_prefix} test set...")
            
            test_embeddings, test_labels = load_embeddings(embeddings_dir, "test_embeddings.npz")
            test_labels_mapped = remap_labels(test_labels, label_map)
            test_dataset = TensorDataset(test_embeddings, test_labels_mapped)
            
            test_results, test_loss, test_micro_acc, test_macro_acc = evaluate_model(
                model, test_dataset, batch_size, valid_classes
            )
            print(f"Test ({model_prefix}): Loss: {test_loss:.4f}, Micro Acc: {test_micro_acc:.4f}, Macro Acc: {test_macro_acc:.4f}")
            
            # Save only the necessary data for visualization
            simplified_results = {
                'predictions': test_results['predictions'],
                'true_labels': test_results['true_labels'],
                'probabilities': test_results['probabilities'],
                'classes': test_results['classes']
            }
            
            np.save(os.path.join(output_dir, f'{label_type}_full_eval_results.npy'), simplified_results)

    import shutil
    shutil.rmtree(temp_dir)
    print(f"\nEvaluation for {model_prefix} completed, results saved!")
    
    return model, valid_classes

def process_contig_lengths(args, label_type, early_stop_patience, batch_size, learning_rate):
    """Process contigs with combined training and length-specific testing."""
    contig_lengths = [500, 1000, 3000, 5000, 10000]
    combined_dir = os.path.join(args.base_dir, f"{label_type}_contig_embed")
    
    # Check if combined directory exists with train and val data
    if not os.path.exists(combined_dir) or not os.path.exists(os.path.join(combined_dir, "train_embeddings.npz")):
        print(f"ERROR: Missing combined {combined_dir} directory or training data")
        return
    
    # Find all length-specific test directories
    length_test_dirs = []
    for length in contig_lengths:
        length_dir = os.path.join(combined_dir, f"length_{length}")
        if os.path.exists(length_dir) and os.path.exists(os.path.join(length_dir, "test_embeddings.npz")):
            length_test_dirs.append(length_dir)
    
    if not length_test_dirs:
        print(f"WARNING: No length-specific test directories found for {label_type}")
    
    # Train on combined data and evaluate on each length-specific test set
    print(f"\n{'='*50}")
    print(f"Training on combined contigs for {label_type}")
    print(f"{'='*50}")
    
    train_and_evaluate(
        combined_dir,
        args.output_dir,
        "combined",
        batch_size,
        learning_rate,
        args.num_train_epochs,
        early_stop_patience,
        label_type,
        length_test_dirs
    )

def main():
    parser = ArgumentParser()
    parser.add_argument("--num_train_epochs", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default="order_output")
    parser.add_argument("--label_type", type=str, default="order", choices=["order", "family", "genus"])
    parser.add_argument("--early_stop_patience", type=int, default=15)
    parser.add_argument("--base_dir", type=str, default="/work/sgk270/genalm_task2")
    args = parser.parse_args()

    # Fixed hyperparameters
    batch_size = 64
    learning_rate = 0.001

    print(f"\nRunning classification for {args.label_type}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process contigs
    process_contig_lengths(args, args.label_type, args.early_stop_patience, batch_size, learning_rate)
    
    # Process full sequences
    full_embeddings_dir = os.path.join(args.base_dir, f"{args.label_type}_full_embed")
    if os.path.exists(full_embeddings_dir):
        print(f"\n{'='*50}")
        print(f"Processing full sequences")
        print(f"{'='*50}")
        try:
            train_and_evaluate(
                full_embeddings_dir,
                args.output_dir,
                "full",
                batch_size,
                learning_rate,
                args.num_train_epochs,
                args.early_stop_patience,
                args.label_type
            )
        except Exception as e:
            print(f"Error processing full sequences: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\nError in main execution:", flush=True)
        print(traceback.format_exc(), flush=True)
        sys.exit(1)