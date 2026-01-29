#ifndef QUANTUM_ENGINE_CUH
#define QUANTUM_ENGINE_CUH

#include <vector>
#include <complex>
#include "cuda_runtime.h"
#include <cuComplex.h>

class QuantumEngine {
public:
    QuantumEngine(unsigned int num_qubits);
    ~QuantumEngine();

    void apply_gate(const std::string& gate_name, unsigned int target_qubit, unsigned int control_qubit = 0);
    std::vector<std::complex<double>> get_statevector();

private:
    unsigned int num_qubits_;
    unsigned long long state_vector_size_;
    cuDoubleComplex* device_state_vector_;
};

#endif // QUANTUM_ENGINE_CUH
