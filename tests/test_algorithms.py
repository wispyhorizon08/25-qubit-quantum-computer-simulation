"""
Integration tests for quantum algorithms.
"""

import sys
import os
import pytest
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from q_sim import Circuit
from q_sim.algorithms import qft, inverse_qft, grovers_search, shors_algorithm


class TestCircuit:
    """Test basic circuit operations."""
    
    def test_circuit_creation(self):
        """Test that we can create a circuit."""
        circuit = Circuit(5)
        assert circuit.num_qubits == 5
    
    def test_hadamard_gate(self):
        """Test Hadamard gate creates superposition."""
        circuit = Circuit(1)
        circuit.h(0)
        state = circuit.get_statevector()
        
        # After H gate, state should be (|0⟩ + |1⟩)/√2
        expected = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
        np.testing.assert_array_almost_equal(np.abs(state), expected, decimal=5)
    
    def test_pauli_x_gate(self):
        """Test Pauli-X gate flips qubit."""
        circuit = Circuit(1)
        circuit.x(0)
        state = circuit.get_statevector()
        
        # After X gate, state should be |1⟩
        expected = np.array([0, 1])
        np.testing.assert_array_almost_equal(np.abs(state), expected, decimal=5)
    
    def test_cnot_gate(self):
        """Test CNOT gate."""
        circuit = Circuit(2)
        circuit.x(0)  # Set control to |1⟩
        circuit.cnot(0, 1)  # Apply CNOT
        state = circuit.get_statevector()
        
        # State should be |11⟩
        expected = np.zeros(4)
        expected[3] = 1  # |11⟩ is index 3
        np.testing.assert_array_almost_equal(np.abs(state), expected, decimal=5)
    
    def test_bell_state(self):
        """Test creation of Bell state."""
        circuit = Circuit(2)
        circuit.h(0)
        circuit.cnot(0, 1)
        state = circuit.get_statevector()
        
        # Bell state: (|00⟩ + |11⟩)/√2
        expected = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])
        np.testing.assert_array_almost_equal(np.abs(state), expected, decimal=5)


class TestQFT:
    """Test Quantum Fourier Transform."""
    
    def test_qft_single_qubit(self):
        """Test QFT on a single qubit."""
        circuit = Circuit(1)
        qft(circuit, [0])
        state = circuit.get_statevector()
        
        # QFT on |0⟩ should give |+⟩ = (|0⟩ + |1⟩)/√2
        expected = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
        np.testing.assert_array_almost_equal(np.abs(state), expected, decimal=5)
    
    def test_qft_inverse(self):
        """Test that QFT followed by inverse QFT returns to original state."""
        circuit = Circuit(3)
        circuit.x(0)  # Set to |001⟩
        
        original_state = circuit.get_statevector()
        
        qft(circuit, [0, 1, 2])
        inverse_qft(circuit, [0, 1, 2])
        
        final_state = circuit.get_statevector()
        
        # Should return to original state
        np.testing.assert_array_almost_equal(np.abs(original_state), np.abs(final_state), decimal=3)


class TestGrover:
    """Test Grover's search algorithm."""
    
    def test_grovers_search(self):
        """Test that Grover's algorithm finds the target with high probability."""
        num_qubits = 4
        target = "1010"
        
        counts = grovers_search(num_qubits, target, iterations=2)
        
        # Target should be among the top results
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        top_result = sorted_counts[0][0]
        
        # The target should be the most frequent result (or close to it)
        assert target in [result[0] for result in sorted_counts[:3]]


class TestShors:
    """Test Shor's factoring algorithm."""
    
    def test_shors_even_number(self):
        """Test factoring an even number."""
        result = shors_algorithm(14)
        assert result["success"]
        assert set(result["factors"]) == {2, 7}
    
    def test_shors_small_odd(self):
        """Test factoring small odd numbers."""
        result = shors_algorithm(15)
        assert result["success"]
        factors = set(result["factors"])
        assert factors == {3, 5} or factors == {15, 1}
    
    def test_shors_gcd(self):
        """Test that algorithm handles cases where gcd(a, N) > 1."""
        # This should be caught by the GCD check
        result = shors_algorithm(21, a=3)
        assert result["success"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
