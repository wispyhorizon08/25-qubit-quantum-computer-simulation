from q_sim import Circuit

# Create a 2-qubit circuit
circuit = Circuit(2)

# Create a Bell state
circuit.h(0)        # Apply Hadamard to qubit 0
circuit.cnot(0, 1)  # Apply CNOT with control=0, target=1

# Get the state vector
state = circuit.get_statevector()
print("State vector:", state)

# Measure the qubits
counts = circuit.get_counts(shots=1000)
print("Measurement results:", counts)