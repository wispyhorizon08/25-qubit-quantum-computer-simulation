#ifndef CUDA_UTILS_HPP
#define CUDA_UTILS_HPP

#include <iostream>
#include <cuda_runtime.h>

#define CUDA_CHECK(err) do { \
    cudaError_t err_ = (err); \
    if (err_ != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err_) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#endif // CUDA_UTILS_HPP