#include "gate_kernels.cuh"
#include <cuComplex.h>

// Hadamard gate kernel
__global__ void hadamard_kernel(cuDoubleComplex* state_vector, unsigned int qubit_index, unsigned long long state_vector_size) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= state_vector_size / 2) return;
    
    unsigned long long bit_mask = 1ULL << qubit_index;
    unsigned long long lower_bits = idx & (bit_mask - 1);
    unsigned long long upper_bits = (idx >> qubit_index) << (qubit_index + 1);
    unsigned long long idx0 = upper_bits | lower_bits;
    unsigned long long idx1 = idx0 | bit_mask;
    
    cuDoubleComplex state0 = state_vector[idx0];
    cuDoubleComplex state1 = state_vector[idx1];
    
    double inv_sqrt2 = 1.0 / sqrt(2.0);
    
    state_vector[idx0] = make_cuDoubleComplex(
        inv_sqrt2 * (state0.x + state1.x),
        inv_sqrt2 * (state0.y + state1.y)
    );
    
    state_vector[idx1] = make_cuDoubleComplex(
        inv_sqrt2 * (state0.x - state1.x),
        inv_sqrt2 * (state0.y - state1.y)
    );
}

// Pauli-X (NOT) gate kernel
__global__ void pauli_x_kernel(cuDoubleComplex* state_vector, unsigned int qubit_index, unsigned long long state_vector_size) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= state_vector_size / 2) return;
    
    unsigned long long bit_mask = 1ULL << qubit_index;
    unsigned long long lower_bits = idx & (bit_mask - 1);
    unsigned long long upper_bits = (idx >> qubit_index) << (qubit_index + 1);
    unsigned long long idx0 = upper_bits | lower_bits;
    unsigned long long idx1 = idx0 | bit_mask;
    
    cuDoubleComplex temp = state_vector[idx0];
    state_vector[idx0] = state_vector[idx1];
    state_vector[idx1] = temp;
}

// CNOT gate kernel
__global__ void cnot_kernel(cuDoubleComplex* state_vector, unsigned int control_qubit, unsigned int target_qubit, unsigned long long state_vector_size) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= state_vector_size) return;
    
    unsigned long long control_mask = 1ULL << control_qubit;
    unsigned long long target_mask = 1ULL << target_qubit;
    
    // Only apply X to target if control qubit is 1
    if ((idx & control_mask) != 0) {
        unsigned long long partner_idx = idx ^ target_mask;
        
        // Only swap if we're the lower index to avoid double-swapping
        if (idx < partner_idx) {
            cuDoubleComplex temp = state_vector[idx];
            state_vector[idx] = state_vector[partner_idx];
            state_vector[partner_idx] = temp;
        }
    }
}

// Controlled Phase gate kernel
__global__ void controlled_phase_kernel(cuDoubleComplex* state_vector, unsigned int control_qubit, unsigned int target_qubit, unsigned long long state_vector_size) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= state_vector_size) return;
    
    unsigned long long control_mask = 1ULL << control_qubit;
    unsigned long long target_mask = 1ULL << target_qubit;
    
    // Apply phase only if both control and target qubits are 1
    if (((idx & control_mask) != 0) && ((idx & target_mask) != 0)) {
        state_vector[idx] = make_cuDoubleComplex(-state_vector[idx].x, -state_vector[idx].y);
    }
}
