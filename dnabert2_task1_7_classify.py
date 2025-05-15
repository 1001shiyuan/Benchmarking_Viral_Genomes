#!/usr/bin/env python3
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import torch.nn.functional as F
from argparse import ArgumentParser
import traceback
import tempfile
import pandas as pd
import shutil
import sys

class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes=2, dropout_prob=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(self.dropout(x))

def load_embeddings(path):
    data = np.load(path)
    return torch.FloatTensor(data['embeddings']), torch.LongTensor(data['labels'])

def evaluate_model(model, dataset, batch_size):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_preds, all_labels, all_probs = [], [], []
    loss_total = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.cuda(), labels.cuda()
            logits = model(features)
            loss = criterion(logits, labels)
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            loss_total += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    micro_acc = (np.array(all_preds) == np.array(all_labels)).mean()
    return {
        'predictions': np.array(all_preds),
        'true_labels': np.array(all_labels),
        'probabilities': np.array(all_probs)
    }, loss_total / len(dataloader), micro_acc

def train_model(model, optimizer, train_loader, val_dataset, batch_size, num_epochs, patience, checkpoint_path):
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float('inf')
    no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for features, labels in train_loader:
            features, labels = features.cuda(), labels.cuda()
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        val_results, val_loss, val_acc = evaluate_model(model, val_dataset, batch_size)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(torch.load(checkpoint_path))
    return model, history

def run_classifier(emb_dir, out_dir, prefix, batch_size, lr, epochs, patience, is_contig=False):
    print(f"\nRunning classifier for {prefix}")
    train_path = os.path.join(emb_dir, "train_embeddings.npz")
    val_path = os.path.join(emb_dir, "val_embeddings.npz")
    test_path = os.path.join(emb_dir, "test_embeddings.npz")

    train_X, train_y = load_embeddings(train_path)
    val_X, val_y = load_embeddings(val_path)
    test_X, test_y = load_embeddings(test_path)

    input_dim = train_X.shape[1]
    model = Classifier(input_dim).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(val_X, val_y)

    with tempfile.TemporaryDirectory() as tmpdir:
        model, history = train_model(
            model, optimizer, train_loader, val_dataset,
            batch_size, epochs, patience,
            os.path.join(tmpdir, "checkpoint.pt")
        )

    pd.DataFrame(history).to_csv(os.path.join(out_dir, f"{prefix}_training_history.csv"), index=False)

    results, loss, acc = evaluate_model(model, TensorDataset(test_X, test_y), batch_size)
    print(f"{prefix} Test Accuracy: {acc:.4f}")
    out_file = f"{prefix}_full_eval_results.npy" if not is_contig else f"{prefix}_contigs_eval_results.npy"
    np.save(os.path.join(out_dir, out_file), results)

def main():
    parser = ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="viral_output")
    parser.add_argument("--num_train_epochs", type=int, default=500)
    parser.add_argument("--early_stop_patience", type=int, default=15)
    args = parser.parse_args()

    base_path = "/work/sgk270/dnabert2_task1_new"
    contig_embed_dir = os.path.join(base_path, "pos_contig_embed")
    full_embed_dir = os.path.join(base_path, "pos_full_embed")
    neg_contig_embed_dir = os.path.join(base_path, "neg_contig_embed")
    neg_full_embed_dir = os.path.join(base_path, "neg_full_embed")

    # Merge positive and negative embeddings into temporary train/val/test sets with binary labels
    def merge_embeddings(pos_dir, neg_dir, split, output_dir):
        pos_npz = np.load(os.path.join(pos_dir, f"{split}_embeddings.npz"))
        neg_npz = np.load(os.path.join(neg_dir, f"{split}_embeddings.npz"))
        X = np.concatenate([pos_npz['embeddings'], neg_npz['embeddings']], axis=0)
        y = np.array([1] * len(pos_npz['embeddings']) + [0] * len(neg_npz['embeddings']))
        np.savez(os.path.join(output_dir, f"{split}_embeddings.npz"), embeddings=X, labels=y)

    os.makedirs(args.output_dir, exist_ok=True)
    tmp_full = os.path.join(args.output_dir, "full")
    tmp_contig = os.path.join(args.output_dir, "contig")
    os.makedirs(tmp_full, exist_ok=True)
    os.makedirs(tmp_contig, exist_ok=True)

    for split in ['train', 'val', 'test']:
        merge_embeddings(os.path.join(full_embed_dir, split), os.path.join(neg_full_embed_dir, split), split, tmp_full)
        merge_embeddings(os.path.join(contig_embed_dir, split), os.path.join(neg_contig_embed_dir, split), split, tmp_contig)

    batch_size = 64
    lr = 0.001

    run_classifier(tmp_full, args.output_dir, "lifestyle_full", batch_size, lr, args.num_train_epochs, args.early_stop_patience, is_contig=False)
    run_classifier(tmp_contig, args.output_dir, "lifestyle_contig", batch_size, lr, args.num_train_epochs, args.early_stop_patience, is_contig=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error in main execution:", flush=True)
        print(traceback.format_exc(), flush=True)
        sys.exit(1)
