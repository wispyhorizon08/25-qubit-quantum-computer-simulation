#include "quantum_engine.cuh"
#include "gate_kernels.cuh"
#include "cuda_utils.hpp"
#include <iostream>
#include <cstring>

QuantumEngine::QuantumEngine(unsigned int num_qubits) 
    : num_qubits_(num_qubits), state_vector_size_(1ULL << num_qubits) {
    
    // Allocate device memory for state vector
    CUDA_CHECK(cudaMalloc(&device_state_vector_, state_vector_size_ * sizeof(cuDoubleComplex)));
    
    // Initialize state to |00...0>
    std::vector<cuDoubleComplex> initial_state(state_vector_size_);
    initial_state[0] = make_cuDoubleComplex(1.0, 0.0);
    for (size_t i = 1; i < state_vector_size_; ++i) {
        initial_state[i] = make_cuDoubleComplex(0.0, 0.0);
    }
    
    CUDA_CHECK(cudaMemcpy(device_state_vector_, initial_state.data(), 
                         state_vector_size_ * sizeof(cuDoubleComplex), 
                         cudaMemcpyHostToDevice));
}

QuantumEngine::~QuantumEngine() {
    CUDA_CHECK(cudaFree(device_state_vector_));
}

void QuantumEngine::apply_gate(const std::string& gate_name, unsigned int target_qubit, unsigned int control_qubit) {
    int threads_per_block = 256;
    int num_blocks;
    
    if (gate_name == "H") {
        num_blocks = (state_vector_size_ / 2 + threads_per_block - 1) / threads_per_block;
        hadamard_kernel<<<num_blocks, threads_per_block>>>(device_state_vector_, target_qubit, state_vector_size_);
    } else if (gate_name == "X") {
        num_blocks = (state_vector_size_ / 2 + threads_per_block - 1) / threads_per_block;
        pauli_x_kernel<<<num_blocks, threads_per_block>>>(device_state_vector_, target_qubit, state_vector_size_);
    } else if (gate_name == "CNOT") {
        num_blocks = (state_vector_size_ + threads_per_block - 1) / threads_per_block;
        cnot_kernel<<<num_blocks, threads_per_block>>>(device_state_vector_, control_qubit, target_qubit, state_vector_size_);
    } else if (gate_name == "CZ") {
        num_blocks = (state_vector_size_ + threads_per_block - 1) / threads_per_block;
        controlled_phase_kernel<<<num_blocks, threads_per_block>>>(device_state_vector_, control_qubit, target_qubit, state_vector_size_);
    } else {
        std::cerr << "Unknown gate: " << gate_name << std::endl;
        return;
    }
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

std::vector<std::complex<double>> QuantumEngine::get_statevector() {
    std::vector<cuDoubleComplex> device_state(state_vector_size_);
    CUDA_CHECK(cudaMemcpy(device_state.data(), device_state_vector_, 
                         state_vector_size_ * sizeof(cuDoubleComplex), 
                         cudaMemcpyDeviceToHost));
    
    std::vector<std::complex<double>> result(state_vector_size_);
    for (size_t i = 0; i < state_vector_size_; ++i) {
        result[i] = std::complex<double>(device_state[i].x, device_state[i].y);
    }
    
    return result;
}
