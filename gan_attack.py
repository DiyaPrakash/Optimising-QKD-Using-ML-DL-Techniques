# gan_attack.py
"""
GAN training and noise generation module.
Produces vectors in [-1,1] that we map to noise model parameters.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import trange
import os
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LATENT_DIM = 32
HIDDEN = 256
OUT_DIM = 128   # adversarial vector length (tweakable)

class Generator(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, out_dim=OUT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, HIDDEN),
            nn.LeakyReLU(0.2),
            nn.Linear(HIDDEN, HIDDEN),
            nn.LeakyReLU(0.2),
            nn.Linear(HIDDEN, out_dim),
            nn.Tanh()
        )
    def forward(self, z): return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, in_dim=OUT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, HIDDEN),
            nn.LeakyReLU(0.2),
            nn.Linear(HIDDEN, HIDDEN//2),
            nn.LeakyReLU(0.2),
            nn.Linear(HIDDEN//2, 1),
            nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

def create_real_dataset(n=4000, dim=OUT_DIM):
    X = np.random.randint(0,2,(n,dim)).astype(np.float32)*2 - 1
    for i in range(0,n,7):
        X[i, :dim//10] += 0.4
    X = np.clip(X, -1.0, 1.0)
    return torch.tensor(X)

def train_gan(epochs=60, batch_size=256, lr=2e-4, save_dir="gan_ckpt"):
    os.makedirs(save_dir, exist_ok=True)
    gen = Generator().to(DEVICE)
    dis = Discriminator().to(DEVICE)
    optG = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5,0.999))
    optD = torch.optim.Adam(dis.parameters(), lr=lr, betas=(0.5,0.999))
    real = create_real_dataset(n=4000, dim=OUT_DIM)
    loader = DataLoader(TensorDataset(real), batch_size=batch_size, shuffle=True, drop_last=True)
    criterion = nn.BCELoss()
    for ep in range(1, epochs+1):
        for (rb,) in loader:
            rb = rb.to(DEVICE)
            bs = rb.size(0)
            # Discriminator
            optD.zero_grad()
            z = torch.randn(bs, LATENT_DIM, device=DEVICE)
            fake = gen(z).detach()
            lossD = criterion(dis(rb), torch.ones((bs,1), device=DEVICE)*0.9) + criterion(dis(fake), torch.zeros((bs,1), device=DEVICE))
            lossD.backward(); optD.step()
            # Generator
            optG.zero_grad()
            z2 = torch.randn(bs, LATENT_DIM, device=DEVICE)
            fake2 = gen(z2)
            lossG = criterion(dis(fake2), torch.ones((bs,1), device=DEVICE))
            lossG.backward(); optG.step()
        if ep % 10 == 0 or ep==epochs:
            torch.save({"G":gen.state_dict(), "D":dis.state_dict(), "epoch":ep}, f"{save_dir}/gan_ep{ep}.pt")
            print(f"Saved GAN epoch {ep}")
    return gen

def generate_noise(generator, n_samples=8):
    generator.eval()
    z = torch.randn(n_samples, LATENT_DIM, device=DEVICE)
    with torch.no_grad():
        out = generator(z).cpu().numpy()
    return out  # in [-1,1] shape (n_samples, OUT_DIM)

if __name__ == "__main__":
    print("Training GAN (this will take a while)...")
    g = train_gan(epochs=40)
    samples = generate_noise(g, 4)
    print("Sample shape:", samples.shape)
