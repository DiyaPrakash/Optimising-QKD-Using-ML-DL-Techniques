# gnn_opt.py
"""
Small Graph Neural Network implemented from scratch with PyTorch:
- Node features + adjacency matrix
- Message passing with learnable weights to predict path reliability (binary)
This is intended for small graphs; it's educational and runnable without extra libs.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_graph(num_nodes=12, p=0.25):
    G = nx.erdos_renyi_graph(num_nodes, p)
    while not nx.is_connected(G):
        G = nx.erdos_renyi_graph(num_nodes, p)
    # assign random fidelity to edges
    for u,v in G.edges():
        G.edges[u,v]['fidelity'] = float(max(0.05, np.random.rand()))
    return G

def graph_to_tensors(G):
    n = G.number_of_nodes()
    # adjacency matrix (n,n)
    A = np.zeros((n,n), dtype=np.float32)
    edge_feat = np.zeros((n,n), dtype=np.float32)
    for u,v in G.edges():
        A[u,v] = 1.0
        A[v,u] = 1.0
        f = G.edges[u,v]['fidelity']
        edge_feat[u,v] = f
        edge_feat[v,u] = f
    # node features can be degree + zeros
    deg = np.array([G.degree(i) for i in range(n)], dtype=np.float32).reshape(n,1)
    X = np.concatenate([deg, np.zeros((n,3), dtype=np.float32)], axis=1)  # (n,4)
    return torch.tensor(A), torch.tensor(edge_feat), torch.tensor(X)

class SimpleMPNN(nn.Module):
    """
    Message passing style network to compute node embeddings and predict reliability score for a (s,t) pair.
    """
    def __init__(self, in_dim=4, hidden=32):
        super().__init__()
        # Encode raw node features -> hidden size once
        self.enc = nn.Linear(in_dim, hidden)
        # MESSAGE MLP now matches what we actually pass: [neigh_hidden, edge_fidelity] => hidden + 1
        self.msg = nn.Linear(hidden + 1, hidden)
        self.update = nn.GRUCell(hidden, hidden)
        self.readout = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, A, E, X, source, target, steps=3):
        # A: (n,n) adjacency (0/1), E: (n,n) edge fidelity, X: (n,in_dim)
        A = A.to(DEVICE).float()
        E = E.to(DEVICE).float()
        X = X.to(DEVICE).float()

        n = X.size(0)
        # Initial hidden states from node features
        h = torch.tanh(self.enc(X))  # (n, hidden)

        for _ in range(steps):
            msgs = torch.zeros((n, h.size(1)), device=DEVICE)  # (n, hidden)
            for u in range(n):
                nbrs = (A[u] > 0).nonzero(as_tuple=True)[0]
                if nbrs.numel() == 0:
                    continue
                neigh_h = h[nbrs]                         # (k, hidden)
                edge_f = E[u, nbrs].unsqueeze(1)          # (k, 1)
                m_in = torch.cat([neigh_h, edge_f], dim=1)  # (k, hidden+1)
                m = torch.tanh(self.msg(m_in))            # (k, hidden)
                msgs[u] = m.mean(dim=0)                   # aggregate

            # GRUCell: new h from messages + previous h
            h = self.update(msgs, h)

        # Readout on (source, target)
        src_h = h[int(source)]
        tgt_h = h[int(target)]
        out = self.readout(torch.cat([src_h, tgt_h], dim=0).unsqueeze(0))  # (1,1)
        return torch.sigmoid(out).squeeze(0)  # scalar in [0,1]


# Synthetic dataset: sample many (G, s, t) triples and label whether path exists with min-fidelity > threshold
def generate_dataset(num_graphs=400, nodes=12, threshold=0.3):
    graphs = []
    labels = []
    for _ in range(num_graphs):
        G = build_graph(nodes, p=0.25)
        A, E, X = graph_to_tensors(G)
        # sample random pair
        s, t = np.random.choice(nodes, size=2, replace=False)
        # compute best path (maximize min-edge-fidelity)
        paths = list(nx.all_simple_paths(G, s, t, cutoff=6))
        best_score = 0.0
        for pth in paths:
            fidelities = [G.edges[pth[i], pth[i+1]]['fidelity'] for i in range(len(pth)-1)]
            score = min(fidelities) if fidelities else 0.0
            best_score = max(best_score, score)
        label = 1 if best_score >= threshold else 0
        graphs.append((A, E, X, s, t))
        labels.append(label)
    return graphs, np.array(labels)

def train_gnn_demo(epochs=20):
    graphs, labels = generate_dataset(num_graphs=600, nodes=12, threshold=0.35)
    # split indices
    idx = np.arange(len(graphs))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=SEED)
    model = SimpleMPNN(in_dim=4, hidden=32).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    for ep in range(1, epochs+1):
        model.train()
        losses = []
        preds = []
        trues = []
        for i in train_idx:
            A, E, X, s, t = graphs[i]
            A_t = A.to(DEVICE)
            E_t = E.to(DEVICE)
            X_t = X.to(DEVICE)
            out = model(A_t, E_t, X_t, int(s), int(t))
            loss = criterion(out, torch.tensor([labels[i]], dtype=torch.float32, device=DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        # eval
        model.eval()
        with torch.no_grad():
            for i in test_idx:
                A, E, X, s, t = graphs[i]
                out = model(A.to(DEVICE), E.to(DEVICE), X.to(DEVICE), int(s), int(t)).cpu().item()
                preds.append(1 if out >= 0.5 else 0)
                trues.append(labels[i])
        acc = accuracy_score(trues, preds)
        print(f"Epoch {ep}/{epochs} train_loss {np.mean(losses):.4f} test_acc {acc:.4f}")
    return model

if __name__ == "__main__":
    print("Training GNN demo")
    train_gnn_demo(epochs=18)
