# auth_models.py
"""
Real implementation for biometric auth models:
- SimpleCNN: image-based biometric classifier (train/val/test)
- SimpleRNN: sequence-based biometric classifier
Features:
- deterministic seeding
- train/val split
- metrics: accuracy, precision, recall, F1, AUC (binary)
- model checkpoint saving
- small synthetic dataset generator (replaceable)
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from tqdm import tqdm

SEED = 42
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

set_seed()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleCNN(nn.Module):
    def __init__(self, in_channels=1, n_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.classifier = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class SimpleRNN(nn.Module):
    def __init__(self, input_dim=16, hidden=64, n_classes=2, num_layers=1):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden, num_layers=num_layers, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden, n_classes)
    def forward(self, x):
        # x: (B, T, F)
        out, hn = self.rnn(x)
        # use last hidden state
        last = out[:, -1, :]
        return self.fc(last)

# Synthetic dataset generators (for testing). Replace with real biometric datasets later.
def synthetic_image_dataset(n=2000, size=64):
    # produce two classes with slightly different spatial patterns
    X = np.random.rand(n, size, size).astype(np.float32)
    centers = np.random.randint(size//4, 3*size//4, size=(n,2))
    for i, c in enumerate(centers):
        rr, cc = np.ogrid[:size, :size]
        mask = ((rr - c[0])**2 + (cc - c[1])**2) < (size//8)**2
        X[i][mask] += (np.random.rand()*0.3 + 0.2)  # add blob
    X = np.clip(X, 0.0, 1.0)
    y = (np.mean(X, axis=(1,2)) > X.mean()).astype(np.int64)  # arbitrary split
    X = X[:, None, :, :]  # (N,1,H,W)
    return torch.tensor(X), torch.tensor(y)

def synthetic_sequence_dataset(n=2000, seq_len=20, feat=16):
    X = np.random.randn(n, seq_len, feat).astype(np.float32)
    # introduce signal pattern for class 1
    for i in range(n//2):
        X[i, :seq_len//4, :feat//2] += 0.8
    y = np.zeros(n,dtype=np.int64)
    y[:n//2] = 1
    # shuffle
    idx = np.random.permutation(n)
    return torch.tensor(X[idx]), torch.tensor(y[idx])

# Training utilities
def train_epoch(model, loader, opt, loss_fn):
    model.train()
    running_loss = 0.0
    preds = []
    trues = []
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        out = model(xb)
        loss = loss_fn(out, yb)
        loss.backward()
        opt.step()
        running_loss += loss.item() * xb.size(0)
        preds.append(out.detach().cpu())
        trues.append(yb.detach().cpu())
    preds = torch.cat(preds)
    trues = torch.cat(trues).numpy()
    probs = nn.functional.softmax(preds, dim=1)[:,1].numpy()
    pred_labels = probs >= 0.5
    loss_avg = running_loss / len(loader.dataset)
    return loss_avg, pred_labels, probs, trues

def eval_epoch(model, loader, loss_fn):
    model.eval()
    running_loss = 0.0
    preds = []
    trues = []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out = model(xb)
            loss = loss_fn(out, yb)
            running_loss += loss.item() * xb.size(0)
            preds.append(out.cpu())
            trues.append(yb.cpu())
    preds = torch.cat(preds)
    trues = torch.cat(trues).numpy()
    probs = nn.functional.softmax(preds, dim=1)[:,1].numpy()
    pred_labels = probs >= 0.5
    loss_avg = running_loss / len(loader.dataset)
    return loss_avg, pred_labels, probs, trues

def compute_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float('nan')
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc}

def fit_model(model, X, y, epochs=15, batch=64, lr=1e-3, save_path="model.pt", val_frac=0.2):
    dataset = TensorDataset(X, y)
    n_val = int(len(dataset) * val_frac)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False)
    model = model.to(DEVICE)
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    best_val_f1 = -1.0
    history = {"train_loss":[], "val_loss":[], "val_f1":[], "val_auc":[]}
    for ep in range(1, epochs+1):
        train_loss, _, _, _ = train_epoch(model, train_loader, opt, loss_fn)
        val_loss, val_pred_labels, val_probs, val_trues = eval_epoch(model, val_loader, loss_fn)
        metrics = compute_metrics(val_trues, val_pred_labels, val_probs)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(metrics["f1"])
        history["val_auc"].append(metrics["auc"])
        print(f"Epoch {ep}/{epochs} | train_loss {train_loss:.4f} val_loss {val_loss:.4f} val_f1 {metrics['f1']:.4f} val_auc {metrics['auc']:.4f}")
        # checkpoint best
        if metrics["f1"] > best_val_f1:
            best_val_f1 = metrics["f1"]
            torch.save({"model_state": model.state_dict(), "metrics": metrics, "epoch": ep}, save_path)
    print("Best val F1:", best_val_f1, "saved to", save_path)
    return history

# CLI-like entry points
def train_cnn_demo(out_path="cnn_checkpoint.pt", epochs=12):
    X, y = synthetic_image_dataset(n=2000, size=64)
    model = SimpleCNN(in_channels=1, n_classes=2)
    hist = fit_model(model, X, y, epochs=epochs, batch=64, lr=1e-3, save_path=out_path)
    return hist

def train_rnn_demo(out_path="rnn_checkpoint.pt", epochs=12):
    X, y = synthetic_sequence_dataset(n=2000, seq_len=20, feat=16)
    model = SimpleRNN(input_dim=16, hidden=64, n_classes=2)
    hist = fit_model(model, X, y, epochs=epochs, batch=64, lr=1e-3, save_path=out_path)
    return hist

if __name__ == "__main__":
    print("Train CNN demo")
    train_cnn_demo(epochs=10)
    print("Train RNN demo")
    train_rnn_demo(epochs=10)
