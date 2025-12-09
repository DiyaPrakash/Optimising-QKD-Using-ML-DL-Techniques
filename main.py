# main.py
"""
Orchestrates:
1. Train GAN (or load checkpoint)
2. Use GAN generator to create adversarial vectors
3. Train DQN agent that uses Cirq BB84 + GAN noise to learn threshold
"""

import os
from gan_attack import train_gan, generate_noise, Generator
from rl_optimizer_qkd import train_agent, evaluate_agent
import torch

def run_pipeline(train_gan_first=True, gan_epochs=40, rl_episodes=120):
    gen_ckpt = "gan_ckpt/gan_ep40.pt"
    generator = None
    if train_gan_first or not os.path.exists(gen_ckpt):
        print("Training GAN...")
        generator = train_gan(epochs=gan_epochs, batch_size=256)
    else:
        print("Loading pretrained GAN...")
        ck = torch.load(gen_ckpt)
        generator = Generator()
        generator.load_state_dict(ck["G"])
    sample = generate_noise(generator, n_samples=2)
    print("Sample noise means:", [float(n.mean()) for n in sample])
    print("Training RL agent for QKD threshold tuning (Cirq)...")
    agent = train_agent(generator, episodes=rl_episodes, n_bits=1024, out="dqn_qkd_cirq.pt")
    return generator, agent

if __name__ == "__main__":
    gen, agent = run_pipeline(train_gan_first=True, gan_epochs=40, rl_episodes=80)
    print("Pipeline finished.")

    # ---- NEW: evaluation summary ----
    print("\n=== Evaluation Summary ===")
    metrics = evaluate_agent(agent, gan_generator=gen, episodes=50, n_bits=512)
    if metrics is None:
        print("No evaluation metrics were produced.")
    else:
        print(f"Episodes:           {metrics['episodes']}")
        print(f"Accept rate:        {metrics['accept_rate']:.3f}")
        print(f"Avg QBER:           {metrics['avg_qber']:.4f}")
        print(f"Avg sift fraction:  {metrics['avg_sift_frac']:.4f}")
        print(f"Avg flip (noise):   {metrics['avg_flip']:.4f}")
        print(f"Avg key rate (est): {metrics['avg_key_rate']:.4f}")
        print(f"Avg reward:         {metrics['avg_reward']:.2f}")
        print(f"Threshold mean±std: {metrics['threshold_mean']:.4f} ± {metrics['threshold_std']:.4f}")
