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
            features, labels = features.to(model.device), labels.to(model.device)
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
        class_accuracies.append((all_preds[mask] == all_labels[mask]).mean())
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

def load_embeddings(embeddings_dir, filename, label_type):
    path = os.path.join(embeddings_dir, filename)
    if not os.path.exists(path):
        print(f"ERROR: Missing {path}")
        sys.exit(1)
    data = np.load(path)
    embeddings = data['embeddings']
    labels = data[f'{label_type}_labels']
    embeddings = torch.FloatTensor(embeddings)
    labels = torch.LongTensor(labels)
    mask = ~(torch.isnan(embeddings).any(dim=1) | torch.isinf(embeddings).any(dim=1))
    return embeddings[mask], labels[mask]

def save_eval_results(output_path, results):
    np.save(output_path, {
        'predictions': results['predictions'],
        'true_labels': results['true_labels'],
        'probabilities': results['probabilities'],
        'classes': results['classes']
    })

def train_model(model, optimizer, train_loader, val_dataset, device, num_epochs, early_stop_patience, batch_size, temp_dir, model_prefix, classes):
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float('inf')
    no_improve_epochs = 0
    train_losses, val_losses = [], []
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
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        train_loss = total_loss / len(train_loader)
        train_macro_acc = np.mean([
            (np.array(all_preds)[np.array(all_labels) == c] == c).mean()
            for c in np.unique(all_labels)
        ])

        val_results, val_loss, _, val_macro_acc = evaluate_model(model, val_dataset, batch_size, classes)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_macro_accs.append(train_macro_acc)
        val_macro_accs.append(val_macro_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epochs = 0
            torch.save(model.state_dict(), os.path.join(temp_dir, f"{model_prefix}_best.pt"))
        else:
            no_improve_epochs += 1

        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_macro_acc={val_macro_acc:.4f}")
        if no_improve_epochs >= early_stop_patience:
            break

    model.load_state_dict(torch.load(os.path.join(temp_dir, f"{model_prefix}_best.pt")))
    return model, {
        'epoch': list(range(1, len(train_losses)+1)),
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_macro_acc': train_macro_accs,
        'val_macro_acc': val_macro_accs
    }

def train_and_evaluate(embeddings_dir, output_dir, model_prefix, batch_size, learning_rate, num_epochs, early_stop_patience, label_type, length_test_dirs):
    os.makedirs(output_dir, exist_ok=True)
    temp_dir = tempfile.mkdtemp()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining on {device}")

    train_X, train_y = load_embeddings(embeddings_dir, "train_embeddings.npz", label_type)
    val_X, val_y = load_embeddings(embeddings_dir, "val_embeddings.npz", label_type)

    encoder_path = os.path.join(embeddings_dir, "label_encoders.npy")
    if not os.path.exists(encoder_path):
        encoder_path = os.path.join(os.path.dirname(embeddings_dir), "label_encoders.npy")
    label_classes = np.load(encoder_path, allow_pickle=True).item()[label_type]['classes']
    print(f"\nNumber of {label_type} classes: {len(label_classes)}")
    print("Class labels:")
    for c in label_classes:
        print(f"  - {c}")

    model = Classifier(train_X.shape[1], len(label_classes)).to(device)
    model.device = device
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(val_X, val_y)

    model, history = train_model(model, optimizer, train_loader, val_dataset, device,
                                 num_epochs, early_stop_patience, batch_size,
                                 temp_dir, model_prefix, label_classes)

    pd.DataFrame(history).to_csv(os.path.join(output_dir, f"{label_type}_{model_prefix}_training_history.csv"), index=False)

    if model_prefix == "full":
        test_X, test_y = load_embeddings(embeddings_dir, "test_embeddings.npz", label_type)
        test_dataset = TensorDataset(test_X, test_y)
        results, loss, micro_acc, macro_acc = evaluate_model(model, test_dataset, batch_size, label_classes)
        print(f"Test (full) - Loss: {loss:.4f}, Micro: {micro_acc:.4f}, Macro: {macro_acc:.4f}")
        save_eval_results(os.path.join(output_dir, f"{label_type}_full_eval_results.npy"), results)

    for test_dir in length_test_dirs:
        name = os.path.basename(test_dir)
        test_X, test_y = load_embeddings(test_dir, "test_embeddings.npz", label_type)
        test_dataset = TensorDataset(test_X, test_y)
        results, loss, micro_acc, macro_acc = evaluate_model(model, test_dataset, batch_size, label_classes)
        print(f"Test ({name}) - Loss: {loss:.4f}, Micro: {micro_acc:.4f}, Macro: {macro_acc:.4f}")
        save_eval_results(os.path.join(output_dir, f"{label_type}_contigs_{name.split('_')[1]}_eval_results.npy"), results)

    shutil.rmtree(temp_dir)
    print("Done.")

def main():
    parser = ArgumentParser()
    parser.add_argument("--num_train_epochs", type=int, default=500)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--label_type", type=str, default="order", choices=["order", "family", "genus"])
    parser.add_argument("--early_stop_patience", type=int, default=15)
    args = parser.parse_args()

    batch_size = 64
    learning_rate = 0.001

    base_dir = "/work/sgk270/prokbert_task2_new"
    contig_dir = os.path.join(base_dir, f"{args.label_type}_contig_embed")
    full_dir = os.path.join(base_dir, f"{args.label_type}_full_embed")

    contig_lengths = [500, 1000, 3000, 5000, 10000]
    length_test_dirs = [
        os.path.join(contig_dir, f"length_{l}")
        for l in contig_lengths
        if os.path.exists(os.path.join(contig_dir, f"length_{l}", "test_embeddings.npz"))
    ]

    train_and_evaluate(contig_dir, args.output_dir, "combined", batch_size, learning_rate,
                       args.num_train_epochs, args.early_stop_patience, args.label_type, length_test_dirs)

    if os.path.exists(full_dir):
        train_and_evaluate(full_dir, args.output_dir, "full", batch_size, learning_rate,
                           args.num_train_epochs, args.early_stop_patience, args.label_type, [])

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\nError in main execution:")
        print(traceback.format_exc())
        sys.exit(1)
