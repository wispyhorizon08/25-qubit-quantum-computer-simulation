# Technical Documentation

## Architecture Overview

This quantum computer simulator is designed as a three-layer architecture, separating concerns between low-level computation, middle-level orchestration, and high-level user interface.

### Layer 1: CUDA Kernels (GPU Computation)

The lowest layer consists of highly optimized CUDA kernels that perform the actual quantum gate operations on the GPU. These kernels operate directly on the quantum state vector, which is stored in GPU memory as an array of complex numbers.

**Key Design Decisions:**

The state vector is represented as an array of `cuDoubleComplex` values, where each element represents the complex amplitude of a basis state. For a 25-qubit system, this results in 2^25 = 33,554,432 complex numbers, requiring 512 MB of GPU memory.

Each gate kernel is designed to operate on the entire state vector in parallel. The parallelization strategy differs based on the gate type:

*   **Single-qubit gates (H, X):** These gates affect pairs of amplitudes. Each thread processes one pair, resulting in 2^(n-1) threads for an n-qubit system.
*   **Two-qubit gates (CNOT, CZ):** These gates conditionally modify amplitudes based on the state of control and target qubits. Each thread checks the relevant qubit states and performs the appropriate operation.

**Bit Manipulation Technique:**

The kernels use efficient bit manipulation to determine which amplitudes to modify. For a single-qubit gate on qubit `k`, the state vector is divided into pairs where the indices differ only in the k-th bit. This is computed using:

```
bit_mask = 1ULL << qubit_index
lower_bits = idx & (bit_mask - 1)
upper_bits = (idx >> qubit_index) << (qubit_index + 1)
idx0 = upper_bits | lower_bits
idx1 = idx0 | bit_mask
```

This approach avoids conditional branching and enables efficient parallel execution.

### Layer 2: Quantum Engine (C++ Orchestration)

The middle layer is implemented in C++ and provides a high-level interface for managing the quantum state and applying gates. The `QuantumEngine` class handles:

*   Memory allocation and deallocation on the GPU
*   State vector initialization
*   Gate application (dispatching to appropriate CUDA kernels)
*   State vector retrieval for measurement

**Memory Management:**

The engine uses CUDA's unified memory management functions (`cudaMalloc`, `cudaFree`, `cudaMemcpy`) to allocate and transfer data between host and device. All memory operations are wrapped with error checking macros to ensure robustness.

**Gate Dispatch:**

The `apply_gate` method acts as a dispatcher, selecting the appropriate CUDA kernel based on the gate name. It also calculates the optimal grid and block dimensions for kernel execution:

```cpp
int threads_per_block = 256;
int num_blocks = (state_vector_size / 2 + threads_per_block - 1) / threads_per_block;
```

This ensures that the GPU's computational resources are fully utilized.

### Layer 3: Python Interface (User API)

The top layer provides a Python interface using Pybind11. The `Circuit` class allows users to build quantum circuits using a fluent API:

```python
circuit = Circuit(5)
circuit.h(0).cnot(0, 1).x(2)
```

**State Vector Access:**

The `get_statevector()` method retrieves the current quantum state from the GPU and converts it to a NumPy array. This allows users to inspect the state at any point during circuit execution.

**Measurement:**

The `measure_all()` and `get_counts()` methods simulate quantum measurement by sampling from the probability distribution defined by the state vector. The probability of measuring basis state `|i⟩` is `|α_i|^2`, where `α_i` is the complex amplitude of that state.

## Algorithm Implementations

### Quantum Fourier Transform (QFT)

The Quantum Fourier Transform is a quantum analogue of the discrete Fourier transform. It transforms the computational basis states as:

```
|j⟩ → (1/√N) Σ exp(2πijk/N) |k⟩
```

**Implementation Strategy:**

The QFT is implemented using a sequence of Hadamard gates and controlled phase rotations. For an n-qubit system, the circuit consists of:

1.  Apply Hadamard to qubit 0
2.  Apply controlled phase rotations from qubits 1 to n-1 to qubit 0
3.  Repeat for qubits 1 to n-1
4.  Reverse the qubit order using SWAP gates

**Simplification:**

In this implementation, controlled phase rotations are approximated using CZ gates for simplicity. A full implementation would require parameterized controlled phase gates with arbitrary angles.

### Grover's Search Algorithm

Grover's algorithm provides a quadratic speedup for unstructured search problems. It can find a marked item in an unsorted database of N items in O(√N) time, compared to O(N) for classical algorithms.

**Algorithm Structure:**

1.  **Initialization:** Create a uniform superposition of all basis states using Hadamard gates.
2.  **Grover Iteration:** Repeat the following steps approximately π/4 * √N times:
    *   **Oracle:** Mark the target state by flipping its phase.
    *   **Diffusion:** Apply the inversion-about-average operator.
3.  **Measurement:** Measure all qubits to obtain the result.

**Oracle Implementation:**

The oracle marks the target state by applying a multi-controlled Z gate that flips the phase only when all qubits match the target pattern. This is implemented by:

1.  Flip qubits where the target bit is 0 (using X gates)
2.  Apply a multi-controlled Z gate
3.  Flip back the qubits (using X gates again)

**Diffusion Operator:**

The diffusion operator, also known as inversion-about-average, is implemented as:

```
H^⊗n · (2|0⟩⟨0| - I) · H^⊗n
```

This is equivalent to:

1.  Apply H to all qubits
2.  Apply X to all qubits
3.  Apply multi-controlled Z gate
4.  Apply X to all qubits
5.  Apply H to all qubits

### Shor's Factoring Algorithm

Shor's algorithm factors integers in polynomial time using quantum computing, providing an exponential speedup over the best known classical algorithms.

**Algorithm Overview:**

1.  **Classical Preprocessing:** Check for trivial factors (even numbers, prime powers, GCD).
2.  **Quantum Period Finding:** Use quantum phase estimation to find the period `r` of the function `f(x) = a^x mod N`.
3.  **Classical Postprocessing:** Use the period to compute factors via `gcd(a^(r/2) ± 1, N)`.

**Simplification:**

This implementation uses a classical period-finding subroutine for demonstration purposes. A full quantum implementation would require:

*   Quantum arithmetic circuits for modular exponentiation
*   Quantum phase estimation using the QFT
*   Additional gate types (controlled rotations, multi-controlled gates)

These components are beyond the scope of this basic simulator but could be added in future versions.

## Performance Considerations

### Memory Requirements

The memory required for the state vector grows exponentially with the number of qubits:

| Qubits | State Vector Size | Memory (double precision) |
|--------|-------------------|---------------------------|
| 10     | 1,024             | 16 KB                     |
| 15     | 32,768            | 512 KB                    |
| 20     | 1,048,576         | 16 MB                     |
| 25     | 33,554,432        | 512 MB                    |
| 30     | 1,073,741,824     | 16 GB                     |

This exponential growth is a fundamental limitation of state vector simulation and is why 25 qubits is a practical upper limit for consumer GPUs.

### Computational Complexity

The computational complexity of gate operations also grows exponentially:

*   **Single-qubit gate:** O(2^n) operations (must update all 2^n amplitudes)
*   **Two-qubit gate:** O(2^n) operations
*   **n-qubit gate:** O(2^n) operations

However, the GPU's parallel processing capabilities allow these operations to be performed in O(2^n / p) time, where p is the number of parallel threads (typically thousands).

### GPU Utilization

To maximize GPU utilization, the simulator uses:

*   **Thread blocks:** 256 threads per block (a common choice for NVIDIA GPUs)
*   **Grid size:** Calculated to ensure all amplitudes are processed
*   **Memory coalescing:** Threads access contiguous memory locations when possible


