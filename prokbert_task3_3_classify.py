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
import shutil

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
    return torch.FloatTensor(data['embeddings']), torch.LongTensor(data['lifestyle_labels'])

def filter_nan_embeddings(embeddings, labels):
    mask = ~(torch.isnan(embeddings).any(dim=1) | torch.isinf(embeddings).any(dim=1))
    return embeddings[mask], labels[mask]

def save_checkpoint(model, optimizer, epoch, val_loss, temp_dir):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'val_loss': val_loss
    }
    torch.save(checkpoint, os.path.join(temp_dir, 'checkpoint.pt'))

def load_checkpoint(model, optimizer, temp_dir):
    checkpoint = torch.load(os.path.join(temp_dir, 'checkpoint.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['val_loss']

def train_model(model, optimizer, train_loader, val_dataset, device, num_epochs, early_stop_patience, batch_size, temp_dir, classes):
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

        val_results, val_loss, _, val_macro_acc = evaluate_model(model, val_dataset, batch_size, classes)
        val_losses.append(val_loss)
        val_macro_accs.append(val_macro_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epochs = 0
            save_checkpoint(model, optimizer, epoch, val_loss, temp_dir)
        else:
            no_improve_epochs += 1

        print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}, val_macro_acc={val_macro_acc:.4f}")

        if no_improve_epochs >= early_stop_patience:
            break

    best_epoch, best_val_loss = load_checkpoint(model, optimizer, temp_dir)
    training_history = {
        'epoch': range(1, len(train_losses) + 1),
        'train_loss': train_losses,
        'train_macro_acc': train_macro_accs,
        'val_loss': val_losses,
        'val_macro_acc': val_macro_accs
    }
    return model, best_epoch, best_val_loss, training_history

def train_and_evaluate(embeddings_dir, output_dir, model_prefix, batch_size, learning_rate, num_epochs, early_stop_patience, length_test_dirs):
    os.makedirs(output_dir, exist_ok=True)
    temp_dir = tempfile.mkdtemp()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining model for {model_prefix} using device: {device}")

    train_embeddings, train_labels = load_embeddings(embeddings_dir, "train_embeddings.npz")
    val_embeddings, val_labels = load_embeddings(embeddings_dir, "val_embeddings.npz")

    train_embeddings, train_labels = filter_nan_embeddings(train_embeddings, train_labels)
    val_embeddings, val_labels = filter_nan_embeddings(val_embeddings, val_labels)

    if train_embeddings.shape[0] == 0:
        print("No training samples after filtering. Skipping training.")
        return

    class_names_path = os.path.join(embeddings_dir, 'label_encoder.npy')
    classes = np.load(class_names_path, allow_pickle=True).item()['classes']

    print(f"\nNumber of lifestyle classes: {len(classes)}")
    print(f"Class labels: {list(classes)}")

    model = Classifier(train_embeddings.shape[1], len(classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    train_dataset = TensorDataset(train_embeddings, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(val_embeddings, val_labels)

    model, best_epoch, best_val_loss, training_history = train_model(
        model, optimizer, train_loader, val_dataset, device,
        num_epochs, early_stop_patience, batch_size,
        temp_dir, classes
    )

    if best_epoch is not None:
        print(f"\nBest epoch: {best_epoch + 1}, Validation loss: {best_val_loss:.4f}")
    else:
        print(f"\nValidation did not improve. Using final model.")

    pd.DataFrame(training_history).to_csv(os.path.join(output_dir, f'lifestyle_{model_prefix}_training_history.csv'), index=False)

    if model_prefix == "full":
        results, loss, micro_acc, macro_acc = evaluate_model(model, val_dataset, batch_size, classes)
        final_path = os.path.join(output_dir, f'lifestyle_full_eval_results.npy')
        np.save(final_path, {
            'predictions': results['predictions'],
            'true_labels': results['true_labels'],
            'probabilities': results['probabilities'],
            'classes': results['classes']
        })

    for test_dir in length_test_dirs:
        length_name = os.path.basename(test_dir)
        test_embeddings, test_labels = load_embeddings(test_dir, "test_embeddings.npz")
        test_embeddings, test_labels = filter_nan_embeddings(test_embeddings, test_labels)
        test_dataset = TensorDataset(test_embeddings, test_labels)
        results, loss, micro_acc, macro_acc = evaluate_model(model, test_dataset, batch_size, classes)
        print(f"\nTest ({length_name}): Loss: {loss:.4f}, Micro Acc: {micro_acc:.4f}, Macro Acc: {macro_acc:.4f}")
        result_path = os.path.join(output_dir, f'lifestyle_contigs_{length_name.split("_")[1]}_eval_results.npy')
        np.save(result_path, {
            'predictions': results['predictions'],
            'true_labels': results['true_labels'],
            'probabilities': results['probabilities'],
            'classes': results['classes']
        })

    shutil.rmtree(temp_dir)
    print(f"\nEvaluation for {model_prefix} completed, results saved!")

def main():
    parser = ArgumentParser()
    parser.add_argument("--num_train_epochs", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default="/work/sgk270/prokbert_task3_new/lifestyle_output")
    parser.add_argument("--early_stop_patience", type=int, default=15)
    args = parser.parse_args()

    batch_size = 64
    learning_rate = 0.001

    base_dir = "/work/sgk270/prokbert_task3_new/contig_embed_labeled"
    contig_lengths = [500, 1000, 3000, 5000, 10000]
    length_test_dirs = [os.path.join(base_dir, f"length_{l}") for l in contig_lengths if os.path.exists(os.path.join(base_dir, f"length_{l}", "test_embeddings.npz"))]

    train_and_evaluate(base_dir, args.output_dir, "contigs", batch_size, learning_rate, args.num_train_epochs, args.early_stop_patience, length_test_dirs)

    full_dir = "/work/sgk270/prokbert_task3_new/full_embed_labeled"
    if os.path.exists(full_dir):
        train_and_evaluate(full_dir, args.output_dir, "full", batch_size, learning_rate, args.num_train_epochs, args.early_stop_patience, [])

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\nError in main execution:")
        print(traceback.format_exc())
        sys.exit(1)
