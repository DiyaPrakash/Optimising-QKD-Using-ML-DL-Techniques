# Hybrid Quantum Security & Machine Learning Framework  
### BB84 QKD • GAN Adversarial Noise • RL Threshold Optimization • Quantum ML • Classical ML Models

This project is an integrated research framework combining **quantum cryptography**, **adversarial machine learning**, **reinforcement learning**, and **classical/quantum ML models**.  
The system simulates the **BB84 quantum key distribution protocol**, generates adversarial noise via a **GAN**, and trains a **Deep Q-Network (DQN)** RL agent to learn optimal security thresholds.

The repository also includes:  
- A Variational Quantum Classifier (VQC)  
- Classical ML demos: CNN, RNN, GNN  
- Synthetic dataset generators (no real data needed)

---

## Project Highlights

- Full BB84 QKD simulation using Cirq  
- GAN generates adversarial noise patterns for QKD  
- RL agent learns optimal QBER acceptance thresholds  
- Quantum ML demo using Cirq  
- Classical ML demos using PyTorch  
- All ML modules include synthetic dataset generation  
- Complete pipeline orchestration (GAN → QKD → RL)

---

## Repository Structure

```
├── main.py
├── qkd_cirq.py
├── rl_optimizer_qkd.py
├── gan_attack.py
├── auth_models.py
├── gnn_opt.py
├── qnn_cirq.py
```

---

## File-by-File Explanation

### 1. qkd_cirq.py — Quantum Key Distribution (BB84)
Implements a full BB84 QKD simulation:

- Alice prepares qubits in random bases  
- Optional Eve intercepts and resends  
- Bob measures with random bases  
- Channel noise simulated via bit flips  
- Outputs:
  - QBER  
  - Sifted key length  
  - Interception statistics  

Used as the RL environment for secure key acceptance.

---

### 2. gan_attack.py — Adversarial Noise Generation (GAN)
Implements a lightweight GAN:

- Generator: produces adversarial noise vectors  
- Discriminator: classifies real/fake noise  
- Synthetic noise dataset automatically created  
- Generator outputs mapped to QKD noise probability  

Adds realism to the noisy quantum channel model.

---

### 3. rl_optimizer_qkd.py — Reinforcement Learning for QKD Thresholds
Contains:

- QKDEnv: the RL environment wrapping BB84  
- DQNAgent: Deep Q-Network that learns QBER acceptance thresholds  
- Reward depends on key usability and noise risk  
- Integrates GAN-generated noise  

Outputs metrics:

- Accept rate  
- QBER statistics  
- Sift fraction  
- Key rate estimate  
- Learned threshold distribution  

---

### 4. main.py — Full Pipeline Orchestrator
Controls the entire workflow:

1. Train or load GAN  
2. Generate noise samples  
3. Train RL agent with noisy QKD simulation  
4. Print evaluation summary  

This is the entry point for running the entire system.

---

### 5. auth_models.py — CNN & RNN Demos
Contains:

- SimpleCNN (image classification)  
- SimpleRNN using GRU (sequence classification)  
- Synthetic dataset generators  
- Metrics: accuracy, precision, recall, F1, AUC  
- Model checkpoint saving  

Standalone ML demonstrations.

---

### 6. gnn_opt.py — Graph Neural Network for Path Reliability
Implements a Message Passing Neural Network (MPNN):

- Generates random graphs  
- Edge fidelity values represent “channel quality”  
- Predicts whether a reliable path exists between two nodes  

Useful analog for quantum network routing.

---

### 7. qnn_cirq.py — Variational Quantum Classifier (VQC)
Implements:

- Angle encoding  
- Parameterized ansatz of Ry, Rz, CNOT  
- Expectation value computation  
- Optimization using COBYLA  
- Synthetic dataset for quick testing  

Demonstrates hybrid quantum–classical ML.

---

## How the Full System Works

1. **GAN** learns to generate adversarial noise vectors.  
2. **BB84 QKD simulation** uses this noise to simulate realistic errors.  
3. **RL agent** learns optimal QBER thresholds for secure key acceptance.  
4. Optional ML modules (CNN, RNN, GNN, VQC) extend the research capabilities.  

This forms a **research-grade hybrid architecture** combining quantum security and advanced machine learning.

---

## Running the Pipeline

### Install dependencies
```
pip install torch numpy networkx cirq tqdm sklearn scipy
```

### Run the orchestrator
```
python main.py
```

This will:

- Train/load GAN  
- Train RL agent  
- Print evaluation metrics  

---

## Output Metrics

Includes:

- Mean reward  
- Average QBER  
- Sift fraction  
- Noise flip probability  
- Accept rate  
- Estimated key rate  
- Threshold mean ± std  

---

## License

MIT License
