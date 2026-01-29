# Quantum Computer Simulation

This project is a high-performance 25-qubit quantum computer simulator built from scratch using CUDA, C++, and Python. It provides a powerful and flexible environment for developing and testing quantum algorithms without relying on external quantum computing libraries like Qiskit or Cirq.

## Features

*   **25-Qubit Simulation:** Simulate quantum systems up to 25 qubits, with a state vector size of over 500 MB.
*   **CUDA Acceleration:** All quantum gate operations are implemented as highly optimized CUDA kernels, leveraging the parallel processing power of NVIDIA GPUs for maximum performance.
*   **Python Interface:** A user-friendly Python interface, built with Pybind11, allows for easy construction and execution of quantum circuits.
*   **Quantum Algorithms:** Includes implementations of key quantum algorithms:
    *   Quantum Fourier Transform (QFT)
    *   Grover's Search Algorithm
    *   Shor's Factoring Algorithm (simplified for demonstration)
*   **Cross-Platform Build:** Uses CMake to provide a cross-platform build system that works on both Windows and Linux.

## Design Choices

### Core Engine: C++/CUDA

The simulation's core is built in C++ and CUDA to achieve the high performance required for quantum state vector simulation. The state of the 25-qubit system is represented by a vector of 2^25 complex numbers. All quantum gate operations are implemented as CUDA kernels that manipulate this state vector on the GPU.

**Alternatives Considered:**

*   **CPU-based simulation:** A pure CPU implementation would be significantly slower and impractical for a 25-qubit system due to the large state vector size (2^25 * 16 bytes = 512 MB) and the computational complexity of gate operations.
*   **OpenCL:** OpenCL is a more portable alternative to CUDA, but CUDA offers a more mature ecosystem and better performance on NVIDIA GPUs, which are widely used in high-performance computing.

### Python Wrapper: Pybind11

A Python wrapper is provided to offer a user-friendly interface for building and running quantum circuits. Pybind11 was chosen for its simplicity and efficiency in creating bindings between C++ and Python.

**Alternatives Considered:**

*   **SWIG:** SWIG is a powerful tool for creating bindings for multiple languages, but it can be more complex to set up and use than Pybind11.
*   **ctypes:** ctypes is a built-in Python library for calling C functions, but it is less convenient for wrapping C++ classes and can lead to more verbose code.

### Build System: CMake

CMake is used to manage the build process. It is a cross-platform build system that can generate build files for various environments, including Windows (with Visual Studio) and Linux.

**Alternatives Considered:**

*   **Make:** Make is a classic build tool, but it is less portable and can be more difficult to configure for complex projects.
*   **SCons:** SCons is a Python-based build system that is easy to use, but it can be slower than CMake for large projects.

## Getting Started

### Prerequisites

*   **Windows 10/11**
*   **NVIDIA GPU** with CUDA support (Compute Capability 5.0 or higher)
*   **NVIDIA CUDA Toolkit** (version 11.0 or later)
*   **Visual Studio 2019 or later** (with C++ development tools)
*   **CMake** (version 3.18 or later)
*   **Python** (version 3.7 or later)

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/wispyhorizon08/25-qubit-quantum-computer-simulation.git
    cd quantum_sim
    ```

2.  **Install Python dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Build and install the simulator:**

    ```bash
    pip install -e .
    ```

    This will invoke CMake to build the C++/CUDA backend and install the Python package in editable mode.

## Usage

### Running Examples

The `examples/` directory contains scripts that demonstrate how to use the simulator.

*   **Grover's Search:**

    ```bash
    python examples/run_grover.py
    ```

*   **Shor's Algorithm:**

    ```bash
    python examples/run_shor.py
    ```

### Running Tests

To run the test suite, use `pytest`:

```bash
pytest
```

## File Structure

```
quantum_sim/
├── CMakeLists.txt                 # Configures NVCC, OpenMP, and Pybind11
├── requirements.txt               # python: pybind11, numpy, pytest
├── setup.py                       # For 'pip install -e .'
│
├── include/                       # C++/CUDA Headers
│   ├── quantum_engine.cuh         # Class definition (GPU pointers, state management)
│   ├── gate_kernels.cuh           # CUDA __global__ kernel declarations
│   ├── cuda_utils.hpp             # Error handling macros (CUDA_CHECK)
│   └── constants.hpp              # Complex math constants (PI, I)
│
├── src/                           # Implementations
│   ├── quantum_engine.cu          # Memory allocation (cudaMalloc) & Orchestration
│   ├── gate_kernels.cu            # The raw CUDA math (H, X, CNOT, CP kernels)
│   └── bindings.cpp               # Pybind11 wrapper (C++ file that calls CUDA)
│
├── q_sim/                        # Python Package (User Interface)
│   ├── __init__.py                # Exposes the 'Circuit' class
│   ├── circuit.py                 # Logic to build gate queues and call the GPU
│   └── algorithms/                # High-level logic (Math heavy)
│       ├── __init__.py
│       ├── qft.py                 # Quantum Fourier Transform implementation
│       ├── grovers.py             # Oracle & Diffusion operator logic
│       ├── shors.py               # Modular exponentiation & Classical GCD/LCD
│       └── utils.py               # Bit-manipulation helpers for Python
│
├── tests/                         # Verification
│   ├── test_gpu_kernels.cu        # Unit tests for CUDA kernels
│   └── test_algorithms.py         # Integration tests for Shor/Grover
│
└── examples/                      # Quick-start scripts
    ├── run_grover.py              # Search for '10110'
    └── run_shor.py                # Factor 15 or 21
```

## Future Work

*   **Additional Gates:** Implement a wider range of quantum gates, including parameterized gates (e.g., `U3`, `Rz`).
*   **Noise Simulation:** Add support for simulating noise and decoherence to model real-world quantum hardware.
*   **Multi-GPU Support:** Extend the simulator to run on multiple GPUs for even larger qubit systems.
*   **Quantum Phase Estimation:** Implement a full quantum phase estimation circuit for a more accurate version of Shor's algorithm.
