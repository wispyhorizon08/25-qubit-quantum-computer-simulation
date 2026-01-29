"""
Circuit class for building and executing quantum circuits.
"""

import numpy as np
from typing import List, Tuple
from . import _quantum_sim_core


class Circuit:
    """
    A quantum circuit that can be built gate-by-gate and executed on GPU.
    """
    
    def __init__(self, num_qubits: int):
        """
        Initialize a quantum circuit.
        
        Args:
            num_qubits: Number of qubits in the circuit (max 25)
        """
        if num_qubits > 25:
            raise ValueError("Maximum 25 qubits supported")
        
        self.num_qubits = num_qubits
        self.engine = _quantum_sim_core.QuantumEngine(num_qubits)
        self.gate_queue = []
    
    def h(self, qubit: int) -> 'Circuit':
        """Apply Hadamard gate to a qubit."""
        self._validate_qubit(qubit)
        self.gate_queue.append(("H", qubit, None))
        self.engine.apply_gate("H", qubit)
        return self
    
    def x(self, qubit: int) -> 'Circuit':
        """Apply Pauli-X (NOT) gate to a qubit."""
        self._validate_qubit(qubit)
        self.gate_queue.append(("X", qubit, None))
        self.engine.apply_gate("X", qubit)
        return self
    
    def cnot(self, control: int, target: int) -> 'Circuit':
        """Apply CNOT gate."""
        self._validate_qubit(control)
        self._validate_qubit(target)
        if control == target:
            raise ValueError("Control and target qubits must be different")
        
        self.gate_queue.append(("CNOT", target, control))
        self.engine.apply_gate("CNOT", target, control)
        return self
    
    def cz(self, control: int, target: int) -> 'Circuit':
        """Apply Controlled-Z gate."""
        self._validate_qubit(control)
        self._validate_qubit(target)
        if control == target:
            raise ValueError("Control and target qubits must be different")
        
        self.gate_queue.append(("CZ", target, control))
        self.engine.apply_gate("CZ", target, control)
        return self
    
    def get_statevector(self) -> np.ndarray:
        """
        Get the current state vector.
        
        Returns:
            Complex numpy array of shape (2^num_qubits,)
        """
        return np.array(self.engine.get_statevector(), dtype=np.complex128)
    
    def measure_all(self, shots: int = 1) -> List[str]:
        """
        Measure all qubits.
        
        Args:
            shots: Number of measurements to perform
            
        Returns:
            List of measurement outcomes as binary strings
        """
        state_vector = self.get_statevector()
        probabilities = np.abs(state_vector) ** 2
        
        # Sample from the probability distribution
        indices = np.random.choice(len(state_vector), size=shots, p=probabilities)
        
        # Convert indices to binary strings
        results = []
        for idx in indices:
            binary_str = format(idx, f'0{self.num_qubits}b')
            results.append(binary_str)
        
        return results
    
    def get_counts(self, shots: int = 1000) -> dict:
        """
        Measure all qubits multiple times and return counts.
        
        Args:
            shots: Number of measurements to perform
            
        Returns:
            Dictionary mapping measurement outcomes to counts
        """
        measurements = self.measure_all(shots)
        counts = {}
        for measurement in measurements:
            counts[measurement] = counts.get(measurement, 0) + 1
        return counts
    
    def _validate_qubit(self, qubit: int):
        """Validate that a qubit index is valid."""
        if qubit < 0 or qubit >= self.num_qubits:
            raise ValueError(f"Qubit index {qubit} out of range [0, {self.num_qubits-1}]")
    
    def reset(self):
        """Reset the circuit to the initial |0...0> state."""
        self.engine = _quantum_sim_core.QuantumEngine(self.num_qubits)
        self.gate_queue = []
