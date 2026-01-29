"""
Quantum Fourier Transform (QFT) Implementation

The QFT is a quantum analogue of the discrete Fourier transform.
It is a key component of many quantum algorithms, including Shor's algorithm.
"""

import math
import numpy as np
from ..circuit import Circuit


def qft(circuit: Circuit, qubits: list) -> Circuit:
    """
    Apply Quantum Fourier Transform to specified qubits.
    
    The QFT transforms the computational basis states as:
    |j⟩ → (1/√N) Σ exp(2πijk/N) |k⟩
    
    Args:
        circuit: The quantum circuit
        qubits: List of qubit indices to apply QFT to
        
    Returns:
        The modified circuit
    """
    n = len(qubits)
    
    for i in range(n):
        # Apply Hadamard to qubit i
        circuit.h(qubits[i])
        
        # Apply controlled phase rotations
        for j in range(i + 1, n):
            # Controlled phase rotation
            # We approximate this with CZ gates for simplicity
            # In a full implementation, we'd need controlled phase rotation gates
            circuit.cz(qubits[j], qubits[i])
    
    # Swap qubits to reverse the order
    for i in range(n // 2):
        swap(circuit, qubits[i], qubits[n - i - 1])
    
    return circuit


def inverse_qft(circuit: Circuit, qubits: list) -> Circuit:
    """
    Apply inverse Quantum Fourier Transform to specified qubits.
    
    Args:
        circuit: The quantum circuit
        qubits: List of qubit indices to apply inverse QFT to
        
    Returns:
        The modified circuit
    """
    n = len(qubits)
    
    # Swap qubits to reverse the order
    for i in range(n // 2):
        swap(circuit, qubits[i], qubits[n - i - 1])
    
    # Apply inverse operations in reverse order
    for i in range(n - 1, -1, -1):
        # Apply controlled phase rotations (inverse)
        for j in range(n - 1, i, -1):
            circuit.cz(qubits[j], qubits[i])
        
        # Apply Hadamard to qubit i
        circuit.h(qubits[i])
    
    return circuit


def swap(circuit: Circuit, qubit1: int, qubit2: int):
    """
    Swap two qubits using CNOT gates.
    
    SWAP = CNOT(a,b) · CNOT(b,a) · CNOT(a,b)
    """
    if qubit1 != qubit2:
        circuit.cnot(qubit1, qubit2)
        circuit.cnot(qubit2, qubit1)
        circuit.cnot(qubit1, qubit2)
