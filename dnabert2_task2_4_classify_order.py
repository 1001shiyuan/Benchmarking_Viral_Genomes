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
    avg_loss = total_loss / len(dataloader)
    micro_accuracy = (all_preds == all_labels).mean()

    class_accuracies = []
    for cls in np.unique(all_labels):
        mask = all_labels == cls
        class_acc = (all_preds[mask] == all_labels[mask]).mean()
        class_accuracies.append(class_acc)
    macro_accuracy = np.mean(class_accuracies)

    return {
        'predictions': all_preds,
        'true_labels': all_labels,
        'probabilities': np.array(all_probs),
        'n_samples': len(all_labels),
        'classes': classes,
        'macro_accuracy': macro_accuracy,
        'class_accuracies': dict(zip(np.unique(all_labels), class_accuracies))
    }, avg_loss, micro_accuracy, macro_accuracy

def load_embeddings(embeddings_dir, filename):
    file_path = os.path.join(embeddings_dir, filename)
    if not os.path.exists(file_path):
        print(f"ERROR: Missing {file_path}")
        sys.exit(1)
    data = np.load(file_path)
    label_type = 'order'
    if 'family' in embeddings_dir.lower():
        label_type = 'family'
    elif 'genus' in embeddings_dir.lower():
        label_type = 'genus'
    return torch.FloatTensor(data['embeddings']), torch.LongTensor(data[f'{label_type}_labels'])

def get_all_unique_labels(embeddings_dir, train_file, val_file):
    all_labels = []
    _, train_labels = load_embeddings(embeddings_dir, train_file)
    all_labels.extend(train_labels.numpy())
    _, val_labels = load_embeddings(embeddings_dir, val_file)
    all_labels.extend(val_labels.numpy())
    unique_labels = np.unique(all_labels)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    return unique_labels, label_map

def add_test_labels_to_map(length_test_dirs, label_map, label_type, unique_labels):
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
    return torch.tensor([label_map[label.item()] for label in labels])

def save_checkpoint(model, optimizer, epoch, val_loss, temp_dir, model_prefix):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'val_loss': val_loss
    }
    torch.save(checkpoint, os.path.join(temp_dir, f'{model_prefix}_checkpoint.pt'))

def load_checkpoint(model, optimizer, temp_dir, model_prefix):
    checkpoint = torch.load(os.path.join(temp_dir, f'{model_prefix}_checkpoint.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['val_loss']

def train_model(model, optimizer, train_loader, val_dataset, device, num_epochs, early_stop_patience, batch_size, temp_dir, model_prefix, valid_classes):
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float('inf')
    no_improve_epochs = 0
    train_losses = []
    val_losses = []
    train_macro_accs = []
    val_macro_accs = []

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

        avg_train_loss = total_loss / len(train_loader)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        class_accuracies = []
        for cls in np.unique(all_labels):
            mask = all_labels == cls
            class_accuracies.append((all_preds[mask] == all_labels[mask]).mean())
        train_macro_acc = np.mean(class_accuracies)

        train_losses.append(avg_train_loss)
        train_macro_accs.append(train_macro_acc)

        val_results, val_loss, _, val_macro_acc = evaluate_model(model, val_dataset, batch_size, valid_classes)
        val_losses.append(val_loss)
        val_macro_accs.append(val_macro_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epochs = 0
            save_checkpoint(model, optimizer, epoch, val_loss, temp_dir, model_prefix)
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= early_stop_patience:
            break

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
    os.makedirs(output_dir, exist_ok=True)
    temp_dir = tempfile.mkdtemp()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining model for {model_prefix} using device: {device}")

    train_embeddings, train_labels = load_embeddings(embeddings_dir, "train_embeddings.npz")

    label_info_path = os.path.join(embeddings_dir, 'label_encoders.npy')
    if not os.path.exists(label_info_path):
        label_info_path = os.path.join(os.path.dirname(embeddings_dir), 'label_encoders.npy')
        if not os.path.exists(label_info_path):
            print(f"ERROR: Missing label encoder at {label_info_path}")
            sys.exit(1)
    label_encoders = np.load(label_info_path, allow_pickle=True).item()

    unique_labels, label_map = get_all_unique_labels(embeddings_dir, "train_embeddings.npz", "val_embeddings.npz")

    if length_test_dirs is not None:
        unique_labels, label_map = add_test_labels_to_map(length_test_dirs, label_map, label_type, unique_labels)

    valid_classes = label_encoders[label_type]['classes'][unique_labels]
    print(f"\nNumber of unique classes for {model_prefix}: {len(unique_labels)}")

    num_classes = len(unique_labels)
    train_labels_mapped = remap_labels(train_labels, label_map)

    val_embeddings, val_labels = load_embeddings(embeddings_dir, "val_embeddings.npz")
    val_labels_mapped = remap_labels(val_labels, label_map)

    model = Classifier(train_embeddings.shape[1], num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    train_dataset = TensorDataset(train_embeddings, train_labels_mapped)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(val_embeddings, val_labels_mapped)

    model, best_epoch, best_val_loss, training_history = train_model(
        model, optimizer, train_loader, val_dataset, device,
        num_epochs, early_stop_patience, batch_size,
        temp_dir, model_prefix, valid_classes
    )

    print(f"\nBest epoch: {best_epoch+1}, Validation loss: {best_val_loss:.4f}")
    pd.DataFrame(training_history).to_csv(os.path.join(output_dir, f'{label_type}_{model_prefix}_training_history.csv'), index=False)

    if length_test_dirs:
        for test_dir in length_test_dirs:
            length_name = os.path.basename(test_dir)
            test_embeddings, test_labels = load_embeddings(test_dir, "test_embeddings.npz")
            test_labels_mapped = remap_labels(test_labels, label_map)
            test_dataset = TensorDataset(test_embeddings, test_labels_mapped)
            results, loss, micro_acc, macro_acc = evaluate_model(model, test_dataset, batch_size, valid_classes)
            print(f"\nTest ({length_name}): Loss: {loss:.4f}, Micro Acc: {micro_acc:.4f}, Macro Acc: {macro_acc:.4f}")
            np.save(os.path.join(output_dir, f'{label_type}_contigs_{length_name.split("_")[1]}_eval_results.npy'), {
                'predictions': results['predictions'],
                'true_labels': results['true_labels'],
                'probabilities': results['probabilities'],
                'classes': results['classes']
            })
    else:
        test_embeddings, test_labels = load_embeddings(embeddings_dir, "test_embeddings.npz")
        test_labels_mapped = remap_labels(test_labels, label_map)
        test_dataset = TensorDataset(test_embeddings, test_labels_mapped)
        results, loss, micro_acc, macro_acc = evaluate_model(model, test_dataset, batch_size, valid_classes)
        print(f"\nTest ({model_prefix}): Loss: {loss:.4f}, Micro Acc: {micro_acc:.4f}, Macro Acc: {macro_acc:.4f}")
        np.save(os.path.join(output_dir, f'{label_type}_full_eval_results.npy'), {
            'predictions': results['predictions'],
            'true_labels': results['true_labels'],
            'probabilities': results['probabilities'],
            'classes': results['classes']
        })

    import shutil
    shutil.rmtree(temp_dir)
    print(f"\nEvaluation for {model_prefix} completed, results saved!")

    return model, valid_classes

def process_contig_lengths(args, label_type, early_stop_patience, batch_size, learning_rate):
    contig_lengths = [500, 1000, 3000, 5000, 10000]
    combined_dir = f"{label_type}_contig_embed"
    if not os.path.exists(combined_dir):
        print(f"ERROR: Missing {combined_dir}")
        return
    length_test_dirs = []
    for length in contig_lengths:
        length_dir = os.path.join(combined_dir, f"length_{length}")
        if os.path.exists(os.path.join(length_dir, "test_embeddings.npz")):
            length_test_dirs.append(length_dir)
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
    args = parser.parse_args()

    batch_size = 64
    learning_rate = 0.001

    os.makedirs(args.output_dir, exist_ok=True)
    process_contig_lengths(args, args.label_type, args.early_stop_patience, batch_size, learning_rate)

    full_dir = f"{args.label_type}_full_embed"
    if os.path.exists(full_dir):
        train_and_evaluate(
            full_dir,
            args.output_dir,
            "full",
            batch_size,
            learning_rate,
            args.num_train_epochs,
            args.early_stop_patience,
            args.label_type
        )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\nError in main execution:", flush=True)
        print(traceback.format_exc(), flush=True)
        sys.exit(1)
