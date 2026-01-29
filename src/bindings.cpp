#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include "quantum_engine.cuh"

namespace py = pybind11;

PYBIND11_MODULE(_quantum_sim_core, m) {
    m.doc() = "CUDA-accelerated quantum computer simulator";

    py::class_<QuantumEngine>(m, "QuantumEngine")
        .def(py::init<unsigned int>(), py::arg("num_qubits"))
        .def("apply_gate", &QuantumEngine::apply_gate, 
             py::arg("gate_name"), 
             py::arg("target_qubit"), 
             py::arg("control_qubit") = 0)
        .def("get_statevector", &QuantumEngine::get_statevector);
}
