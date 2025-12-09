# qnn_cirq.py
"""
Variational Quantum Classifier implemented with Cirq.
- Angle encoding (Ry) of features
- Parameterized ansatz (Ry, Rz) layers with CNOT ladder entanglement
- Uses Cirq simulator state vector to compute <Z> on qubit 0 and maps via sigmoid to probability
- Trains with scipy.optimize.minimize (COBYLA)
"""

import cirq
import numpy as np
from scipy.optimize import minimize
from typing import Tuple

def build_vqc_circuit(n_qubits:int, x:np.ndarray, params:np.ndarray, layers:int=2):
    """
    Construct cirq.Circuit for data x and flattened params.
    params shape: layers * n_qubits * 2  (ry, rz per qubit per layer)
    """
    qubits = [cirq.LineQubit(i) for i in range(n_qubits)]
    circuit = cirq.Circuit()
    # angle encoding: use Ry on each qubit (cirq.ry is rotation around Y via `cirq.ry` is not present; use cirq.ops.rx / rz? We'll use rotation gates via cirq.ry)
    # Cirq has ops: cirq.ry(angle)(qubit)
    for i in range(n_qubits):
        angle = float(x[i % len(x)])
        circuit.append(cirq.ry(angle)(qubits[i]))
    exp_len = layers * n_qubits * 2
    assert params.size == exp_len, f"params length {params.size} != expected {exp_len}"
    p = params.reshape((layers, n_qubits, 2))
    for l in range(layers):
        # single-qubit rotations
        for q in range(n_qubits):
            circuit.append(cirq.ry(float(p[l,q,0]))(qubits[q]))
            circuit.append(cirq.rz(float(p[l,q,1]))(qubits[q]))
        # entangling ladder
        for q in range(n_qubits-1):
            circuit.append(cirq.CNOT(qubits[q], qubits[q+1]))
    return circuit, qubits

def statevector_expectation(circuit:cirq.Circuit, qubits, qubit_idx:int=0):
    sim = cirq.Simulator()
    res = sim.simulate(circuit)
    sv = res.final_state_vector
    n = len(qubits)
    exp = 0.0
    # enumerate basis states
    dim = 2**n
    for idx in range(dim):
        amp = sv[idx]
        prob = np.abs(amp)**2
        bit = (idx >> qubit_idx) & 1
        val = 1.0 if bit == 0 else -1.0
        exp += prob * val
    return float(exp)

def predict_proba(params:np.ndarray, X:np.ndarray, n_qubits:int, layers:int=2):
    probs = []
    for x in X:
        qc, qubits = build_vqc_circuit(n_qubits, x, params, layers=layers)
        expz = statevector_expectation(qc, qubits, qubit_idx=0)
        p = 1.0 / (1.0 + np.exp(-3.0 * expz))
        probs.append(p)
    return np.array(probs)

def bce_loss(params:np.ndarray, X:np.ndarray, y:np.ndarray, n_qubits:int, layers:int=2, l2=0.0):
    probs = predict_proba(params, X, n_qubits, layers=layers)
    eps = 1e-9
    probs = np.clip(probs, eps, 1-eps)
    loss = - np.mean(y * np.log(probs) + (1-y) * np.log(1-probs))
    loss += l2 * np.sum(params**2)
    return float(loss)

def train_vqc(X:np.ndarray, y:np.ndarray, n_qubits:int=2, layers:int=2, maxiter:int=100, verbose:bool=True):
    param_len = layers * n_qubits * 2
    init = 0.1 * np.random.randn(param_len)
    def obj(p):
        val = bce_loss(p, X, y, n_qubits, layers=layers)
        if verbose:
            print("Obj:", val)
        return val
    res = minimize(obj, init, method='COBYLA', options={'maxiter':maxiter, 'disp':True})
    print("Optimization success:", res.success, res.message)
    return res.x

# small synthetic dataset
def synthetic_2d_dataset(n=80):
    rng = np.random.default_rng(123)
    X0 = rng.normal([-1,-1], 0.35, size=(n//2, 2))
    X1 = rng.normal([1,1], 0.35, size=(n//2, 2))
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n//2), np.ones(n//2)])
    X = np.tanh(X) * np.pi
    return X, y

if __name__ == "__main__":
    print("Training VQC (Cirq) small demo")
    X, y = synthetic_2d_dataset(80)
    n_qubits = 2
    params = train_vqc(X, y, n_qubits=n_qubits, layers=2, maxiter=40)
    probs = predict_proba(params, X, n_qubits=n_qubits, layers=2)
    preds = (probs >= 0.5).astype(int)
    acc = np.mean(preds == y)
    print("Train accuracy:", acc)
