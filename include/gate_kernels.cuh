#ifndef GATE_KERNELS_CUH
#define GATE_KERNELS_CUH

#include "cuda_runtime.h"
#include <cuComplex.h>

// Single-qubit gates
__global__ void hadamard_kernel(cuDoubleComplex* state_vector, unsigned int qubit_index, unsigned long long state_vector_size);
__global__ void pauli_x_kernel(cuDoubleComplex* state_vector, unsigned int qubit_index, unsigned long long state_vector_size);

// Two-qubit gates
__global__ void cnot_kernel(cuDoubleComplex* state_vector, unsigned int control_qubit, unsigned int target_qubit, unsigned long long state_vector_size);
__global__ void controlled_phase_kernel(cuDoubleComplex* state_vector, unsigned int control_qubit, unsigned int target_qubit, unsigned long long state_vector_size);

#endif // GATE_KERNELS_CUH
