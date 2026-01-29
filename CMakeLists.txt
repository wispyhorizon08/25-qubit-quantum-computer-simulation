cmake_minimum_required(VERSION 3.18)
project(quantum_sim LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Find Python and pybind11
find_package(Python COMPONENTS Interpreter Development REQUIRED)
# Try to find pybind11 via CMake config first
find_package(pybind11 CONFIG QUIET)

# If not found, try using Python to locate it
if(NOT pybind11_FOUND)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE pybind11_RESULT
    )
    if(pybind11_RESULT EQUAL 0)
        find_package(pybind11 CONFIG REQUIRED)
    else()
        message(FATAL_ERROR "pybind11 not found. Please install it with: pip install pybind11")
    endif()
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# CUDA source files
set(CUDA_SOURCES
    src/quantum_engine.cu
    src/gate_kernels.cu
)

# Create CUDA library
# Create CUDA library
add_library(quantum_engine_cuda STATIC ${CUDA_SOURCES})
target_include_directories(quantum_engine_cuda PUBLIC ${CMAKE_SOURCE_DIR}/include)
set_target_properties(quantum_engine_cuda PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Set CUDA architecture (adjust based on your GPU)
# Common options: 50, 60, 70, 75, 80, 86, 89, 90
set_target_properties(quantum_engine_cuda PROPERTIES CUDA_ARCHITECTURES "75;86")

# Pybind11 module
pybind11_add_module(_quantum_sim_core src/bindings.cpp)
target_link_libraries(_quantum_sim_core PRIVATE quantum_engine_cuda)
target_link_libraries(_quantum_sim_core PRIVATE CUDA::cudart)

# Set output directory
set_target_properties(_quantum_sim_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# Windows-specific settings
if(WIN32)
    # Ensure proper linking on Windows
    target_compile_options(quantum_engine_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/MD>)
    set_target_properties(_quantum_sim_core PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/q_sim
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/q_sim
    )
endif()

# Installation
install(TARGETS _quantum_sim_core LIBRARY DESTINATION q_sim)
