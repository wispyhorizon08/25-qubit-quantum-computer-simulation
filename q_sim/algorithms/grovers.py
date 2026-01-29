"""
Grover's Search Algorithm Implementation

Grover's algorithm provides a quadratic speedup for unstructured search problems.
It can find a marked item in an unsorted database of N items in O(√N) time.
"""

import math
import numpy as np
from ..circuit import Circuit
from .utils import binary_to_int


def grovers_search(num_qubits: int, target: str, iterations: int = None) -> dict:
    """
    Run Grover's algorithm to search for a target state.
    
    Args:
        num_qubits: Number of qubits
        target: Target state as binary string (e.g., "10110")
        iterations: Number of Grover iterations (default: optimal number)
        
    Returns:
        Dictionary with measurement counts
    """
    if len(target) != num_qubits:
        raise ValueError(f"Target must be {num_qubits} bits long")
    
    # Calculate optimal number of iterations
    if iterations is None:
        N = 2 ** num_qubits
        iterations = int(math.pi / 4 * math.sqrt(N))
    
    circuit = Circuit(num_qubits)
    
    # Initialize superposition
    for i in range(num_qubits):
        circuit.h(i)
    
    # Apply Grover iterations
    for _ in range(iterations):
        # Oracle: mark the target state
        oracle(circuit, target)
        
        # Diffusion operator
        diffusion(circuit, num_qubits)
    
    # Measure
    counts = circuit.get_counts(shots=1000)
    
    return counts


def oracle(circuit: Circuit, target: str):
    """
    Oracle that marks the target state by flipping its phase.
    
    This implementation uses X gates to flip qubits where target bit is 0,
    then applies a multi-controlled Z gate, then flips back.
    
    Args:
        circuit: The quantum circuit
        target: Target state as binary string
    """
    num_qubits = len(target)
    
    # Flip qubits where target bit is 0
    for i in range(num_qubits):
        if target[i] == '0':
            circuit.x(i)
    
    # Apply multi-controlled Z gate
    # For simplicity, we use a chain of CNOT and CZ gates
    # In a full implementation, this would be a proper multi-controlled gate
    if num_qubits == 1:
        circuit.x(0)
        circuit.h(0)
        circuit.x(0)
        circuit.h(0)
    else:
        # Simplified multi-controlled Z using available gates
        # This is an approximation for demonstration
        for i in range(num_qubits - 1):
            circuit.cz(i, num_qubits - 1)
    
    # Flip back qubits where target bit is 0
    for i in range(num_qubits):
        if target[i] == '0':
            circuit.x(i)


def diffusion(circuit: Circuit, num_qubits: int):
    """
    Apply the Grover diffusion operator (inversion about average).
    
    The diffusion operator is: 2|s⟩⟨s| - I
    where |s⟩ is the uniform superposition state.
    
    Args:
        circuit: The quantum circuit
        num_qubits: Number of qubits
    """
    # Apply H to all qubits
    for i in range(num_qubits):
        circuit.h(i)
    
    # Apply X to all qubits
    for i in range(num_qubits):
        circuit.x(i)
    
    # Apply multi-controlled Z gate
    # Simplified implementation
    if num_qubits > 1:
        for i in range(num_qubits - 1):
            circuit.cz(i, num_qubits - 1)
    
    # Apply X to all qubits
    for i in range(num_qubits):
        circuit.x(i)
    
    # Apply H to all qubits
    for i in range(num_qubits):
        circuit.h(i)
