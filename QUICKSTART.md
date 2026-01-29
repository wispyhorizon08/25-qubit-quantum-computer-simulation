# Quick Start Guide

This guide will help you get started with the quantum computer simulator in just a few minutes.

## Installation

### Step 1: Install Prerequisites

Before installing the simulator, ensure you have:

*   **NVIDIA GPU** with CUDA support
*   **CUDA Toolkit** (download from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads))
*   **Python 3.7+** (download from [python.org](https://www.python.org/downloads/))
*   **Visual Studio 2019+** with C++ tools (download from [Microsoft](https://visualstudio.microsoft.com/))
*   **CMake 3.18+** (download from [cmake.org](https://cmake.org/download/))

### Step 2: Install the Simulator

```bash
# Clone the repository
git clone https://github.com/wispyhorizon08/25-qubit-quantum-computer-simulation.git
cd quantum_sim

# Install Python dependencies
pip install -r requirements.txt

# Build and install
pip install -e .
```

## Your First Quantum Circuit

Create a file called `my_first_circuit.py`:

```python
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
```

Run the script:

```bash
python my_first_circuit.py
```

You should see output showing the Bell state: approximately 50% |00⟩ and 50% |11⟩.

## Running Examples

The `examples/` directory contains ready-to-run scripts:

### Grover's Search

Search for a specific state in a quantum superposition:

```bash
python examples/run_grover.py
```

This will search for the state `|10110⟩` in a 5-qubit system and demonstrate the quantum speedup.

### Shor's Algorithm

Factor integers using quantum computing:

```bash
python examples/run_shor.py
```

This will factor numbers like 15 and 21 using Shor's algorithm.

## Basic Operations

### Creating a Circuit

```python
from q_sim import Circuit

# Create a circuit with n qubits
circuit = Circuit(n)
```

### Applying Gates

```python
# Single-qubit gates
circuit.h(0)    # Hadamard gate
circuit.x(1)    # Pauli-X (NOT) gate

# Two-qubit gates
circuit.cnot(0, 1)  # CNOT gate (control=0, target=1)
circuit.cz(0, 1)    # Controlled-Z gate
```

### Measuring

```python
# Get the full state vector
state = circuit.get_statevector()

# Measure all qubits (returns list of binary strings)
results = circuit.measure_all(shots=1)

# Get measurement counts
counts = circuit.get_counts(shots=1000)
```

## Common Quantum States

### Superposition

```python
circuit = Circuit(1)
circuit.h(0)  # Creates (|0⟩ + |1⟩)/√2
```

### Entanglement (Bell State)

```python
circuit = Circuit(2)
circuit.h(0)
circuit.cnot(0, 1)  # Creates (|00⟩ + |11⟩)/√2
```

### GHZ State

```python
circuit = Circuit(3)
circuit.h(0)
circuit.cnot(0, 1)
circuit.cnot(0, 2)  # Creates (|000⟩ + |111⟩)/√2
```

