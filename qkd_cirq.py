# qkd_cirq.py
"""
BB84-like QKD simulation using Cirq simulator (local only).
- Alice prepares qubits in Z or X basis.
- Optional Eve (intercept-resend) measuring in random basis and resending.
- Channel noise modeled as random bit-flips on measurement outcomes.
- Returns sifted keys and metadata including QBER.
"""

import cirq
import numpy as np
from typing import Tuple, Dict
import random

def single_bb84_circuit(alice_bit:int, alice_basis:int, bob_basis:int, eve_basis:int=None, eve_meas:bool=False):
    """
    Returns (cirq.Circuit, role) where role indicates:
     - 'direct' : no Eve (Alice -> Bob)
     - 'eve'    : Eve measures and then must reprepare (we'll handle reprepare separately)
    """
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit()
    # prepare |1> if alice_bit == 1
    if alice_bit == 1:
        circuit.append(cirq.X(q))
    # prepare X-basis if alice_basis==1
    if alice_basis == 1:
        circuit.append(cirq.H(q))
    # If Eve measures, she will measure in eve_basis; we append measurement op
    if eve_meas and eve_basis is not None:
        if eve_basis == 1:
            circuit.append(cirq.H(q))
        circuit.append(cirq.measure(q, key='m'))  # Eve measures
        return circuit, 'eve'
    # Otherwise Bob measures
    if bob_basis == 1:
        circuit.append(cirq.H(q))
    circuit.append(cirq.measure(q, key='m'))
    return circuit, 'direct'

def reprepare_and_measure_circuit(meas_outcome:int, eve_basis:int, bob_basis:int):
    """Return circuit where Eve has re-prepared measured outcome and Bob measures."""
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit()
    # reprepare measured state
    if eve_basis == 0:
        if meas_outcome == 1:
            circuit.append(cirq.X(q))
    else:
        if meas_outcome == 1:
            circuit.append(cirq.X(q))
        circuit.append(cirq.H(q))
    # Bob measurement
    if bob_basis == 1:
        circuit.append(cirq.H(q))
    circuit.append(cirq.measure(q, key='m'))
    return circuit

def bb84_cirq(n_bits:int=512, noise_prob:float=0.01, eve_prob:float=0.0, seed:int=42, sampler=None) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Run BB84 simulation using Cirq.
    Returns: sifted_alice_bits, sifted_bob_bits, metadata
    """
    rng = np.random.default_rng(seed)
    alice_bits = rng.integers(0,2,n_bits)
    alice_bases = rng.integers(0,2,n_bits)
    bob_bases = rng.integers(0,2,n_bits)

    sampler = sampler or cirq.Simulator()

    bob_measurements = np.full(n_bits, -1, dtype=int)
    intercepted_mask = np.zeros(n_bits, dtype=bool)

    for i in range(n_bits):
        a_bit = int(alice_bits[i]); a_basis = int(alice_bases[i]); b_basis = int(bob_bases[i])
        # Eve intercept?
        if rng.random() < eve_prob:
            intercepted_mask[i] = True
            eve_basis = int(rng.integers(0,2))
            qc, role = single_bb84_circuit(a_bit, a_basis, b_basis, eve_basis=eve_basis, eve_meas=True)
            res = sampler.run(qc, repetitions=1)
            meas = int(res.measurements['m'][0][0])
            # Eve reprepare and Bob measures:
            qc2 = reprepare_and_measure_circuit(meas, eve_basis, b_basis)
            res2 = sampler.run(qc2, repetitions=1)
            bob_meas = int(res2.measurements['m'][0][0])
            # channel flip noise
            if rng.random() < noise_prob:
                bob_meas ^= 1
            bob_measurements[i] = bob_meas
            continue
        # No Eve:
        qc, _ = single_bb84_circuit(a_bit, a_basis, b_basis, eve_meas=False)
        res = sampler.run(qc, repetitions=1)
        bob_meas = int(res.measurements['m'][0][0])
        if rng.random() < noise_prob:
            bob_meas ^= 1
        bob_measurements[i] = bob_meas

    # sifting
    valid = (bob_measurements != -1) & (alice_bases == bob_bases)
    sifted_alice = alice_bits[valid]
    sifted_bob = bob_measurements[valid]
    qber = float(np.mean(sifted_alice != sifted_bob)) if len(sifted_alice)>0 else 1.0

    meta = {"n_sent": n_bits, "sifted_len": int(len(sifted_alice)), "qber": qber, "intercepted_frac": float(intercepted_mask.mean())}
    return sifted_alice, sifted_bob, meta

if __name__ == "__main__":
    print("Running Cirq BB84 (small demo)...")
    sa, sb, meta = bb84_cirq(n_bits=256, noise_prob=0.02, eve_prob=0.05, seed=123)
    print("Sifted length:", meta["sifted_len"], "QBER:", meta["qber"], "Intercepted:", meta["intercepted_frac"])
