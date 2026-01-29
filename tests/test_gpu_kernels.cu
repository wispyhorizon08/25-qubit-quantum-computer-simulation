/**
 * Unit tests for CUDA kernels
 * 
 * This file contains tests for the low-level CUDA kernels.
 * Compile with: nvcc -I../include test_gpu_kernels.cu ../src/gate_kernels.cu -o test_kernels
 */

#include <iostream>
#include <cmath>
#include <vector>
#include "gate_kernels.cuh"
#include "cuda_utils.hpp"

const double EPSILON = 1e-6;

bool approx_equal(double a, double b) {
    return std::abs(a - b) < EPSILON;
}

bool test_hadamard_kernel() {
    std::cout << "Testing Hadamard kernel..." << std::endl;
    
    // Create initial state |0⟩
    const int num_qubits = 1;
    const unsigned long long state_size = 1ULL << num_qubits;
    
    std::vector<cuDoubleComplex> host_state(state_size);
    host_state[0] = make_cuDoubleComplex(1.0, 0.0);
    host_state[1] = make_cuDoubleComplex(0.0, 0.0);
    
    // Allocate device memory
    cuDoubleComplex* device_state;
    CUDA_CHECK(cudaMalloc(&device_state, state_size * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMemcpy(device_state, host_state.data(), 
                         state_size * sizeof(cuDoubleComplex), 
                         cudaMemcpyHostToDevice));
    
    // Apply Hadamard gate
    int threads = 256;
    int blocks = (state_size / 2 + threads - 1) / threads;
    hadamard_kernel<<<blocks, threads>>>(device_state, 0, state_size);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(host_state.data(), device_state, 
                         state_size * sizeof(cuDoubleComplex), 
                         cudaMemcpyDeviceToHost));
    
    // Check result: should be (|0⟩ + |1⟩)/√2
    double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    bool passed = approx_equal(host_state[0].x, inv_sqrt2) &&
                  approx_equal(host_state[0].y, 0.0) &&
                  approx_equal(host_state[1].x, inv_sqrt2) &&
                  approx_equal(host_state[1].y, 0.0);
    
    CUDA_CHECK(cudaFree(device_state));
    
    if (passed) {
        std::cout << "  ✓ Hadamard kernel test passed" << std::endl;
    } else {
        std::cout << "  ✗ Hadamard kernel test failed" << std::endl;
        std::cout << "    Expected: (" << inv_sqrt2 << ", 0), (" << inv_sqrt2 << ", 0)" << std::endl;
        std::cout << "    Got: (" << host_state[0].x << ", " << host_state[0].y << "), "
                  << "(" << host_state[1].x << ", " << host_state[1].y << ")" << std::endl;
    }
    
    return passed;
}

bool test_pauli_x_kernel() {
    std::cout << "Testing Pauli-X kernel..." << std::endl;
    
    // Create initial state |0⟩
    const int num_qubits = 1;
    const unsigned long long state_size = 1ULL << num_qubits;
    
    std::vector<cuDoubleComplex> host_state(state_size);
    host_state[0] = make_cuDoubleComplex(1.0, 0.0);
    host_state[1] = make_cuDoubleComplex(0.0, 0.0);
    
    // Allocate device memory
    cuDoubleComplex* device_state;
    CUDA_CHECK(cudaMalloc(&device_state, state_size * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMemcpy(device_state, host_state.data(), 
                         state_size * sizeof(cuDoubleComplex), 
                         cudaMemcpyHostToDevice));
    
    // Apply Pauli-X gate
    int threads = 256;
    int blocks = (state_size / 2 + threads - 1) / threads;
    pauli_x_kernel<<<blocks, threads>>>(device_state, 0, state_size);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(host_state.data(), device_state, 
                         state_size * sizeof(cuDoubleComplex), 
                         cudaMemcpyDeviceToHost));
    
    // Check result: should be |1⟩
    bool passed = approx_equal(host_state[0].x, 0.0) &&
                  approx_equal(host_state[0].y, 0.0) &&
                  approx_equal(host_state[1].x, 1.0) &&
                  approx_equal(host_state[1].y, 0.0);
    
    CUDA_CHECK(cudaFree(device_state));
    
    if (passed) {
        std::cout << "  ✓ Pauli-X kernel test passed" << std::endl;
    } else {
        std::cout << "  ✗ Pauli-X kernel test failed" << std::endl;
    }
    
    return passed;
}

bool test_cnot_kernel() {
    std::cout << "Testing CNOT kernel..." << std::endl;
    
    // Create initial state |10⟩
    const int num_qubits = 2;
    const unsigned long long state_size = 1ULL << num_qubits;
    
    std::vector<cuDoubleComplex> host_state(state_size);
    for (int i = 0; i < state_size; i++) {
        host_state[i] = make_cuDoubleComplex(0.0, 0.0);
    }
    host_state[2] = make_cuDoubleComplex(1.0, 0.0); // |10⟩ is index 2
    
    // Allocate device memory
    cuDoubleComplex* device_state;
    CUDA_CHECK(cudaMalloc(&device_state, state_size * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMemcpy(device_state, host_state.data(), 
                         state_size * sizeof(cuDoubleComplex), 
                         cudaMemcpyHostToDevice));
    
    // Apply CNOT gate (control=0, target=1)
    int threads = 256;
    int blocks = (state_size + threads - 1) / threads;
    cnot_kernel<<<blocks, threads>>>(device_state, 0, 1, state_size);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(host_state.data(), device_state, 
                         state_size * sizeof(cuDoubleComplex), 
                         cudaMemcpyDeviceToHost));
    
    // Check result: should be |11⟩ (index 3)
    bool passed = approx_equal(host_state[3].x, 1.0) &&
                  approx_equal(host_state[3].y, 0.0);
    
    CUDA_CHECK(cudaFree(device_state));
    
    if (passed) {
        std::cout << "  ✓ CNOT kernel test passed" << std::endl;
    } else {
        std::cout << "  ✗ CNOT kernel test failed" << std::endl;
    }
    
    return passed;
}

int main() {
    std::cout << "Running CUDA kernel tests..." << std::endl;
    std::cout << "=====================================" << std::endl;
    
    int passed = 0;
    int total = 0;
    
    total++; if (test_hadamard_kernel()) passed++;
    total++; if (test_pauli_x_kernel()) passed++;
    total++; if (test_cnot_kernel()) passed++;
    
    std::cout << "=====================================" << std::endl;
    std::cout << "Results: " << passed << "/" << total << " tests passed" << std::endl;
    
    return (passed == total) ? 0 : 1;
}
