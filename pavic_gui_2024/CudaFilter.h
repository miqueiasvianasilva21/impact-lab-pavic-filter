#pragma once

#include <iostream>
#include "cuda_runtime.h"

// Macro para verificação de erros (sem alterações)
#define CUDA_CHECK(err) { \
    cudaError_t err_ = (err); \
    if (err_ != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err_) << " in " << __FILE__ << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// Declaração da função para o filtro Sépia
extern "C" void run_sepia_filter_cuda(unsigned char* d_out, const unsigned char* d_in, int width, int height, int stride);

// NOVO: Declaração da função para o filtro de Inversão de Cores
extern "C" void run_inversion_filter_cuda(unsigned char* d_out, const unsigned char* d_in, int width, int height, int stride);
