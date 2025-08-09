#include "CudaFilter.h"
#include "device_launch_parameters.h"
#include <stdio.h>

//====================================================================
// FILTRO SÉPIA (código existente, sem alterações)
//====================================================================
__device__ unsigned char clamp(float value) {
    if (value > 255.0f) return 255;
    return (unsigned char)value;
}

__global__ void sepia_kernel(unsigned char* out, const unsigned char* in, int width, int height, int stride)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height)
    {
        int pixel_offset = row * stride + col * 3;
        if (pixel_offset + 2 < stride * height) {
            float b = in[pixel_offset];
            float g = in[pixel_offset + 1];
            float r = in[pixel_offset + 2];
            float new_r = r * 0.393f + g * 0.769f + b * 0.189f;
            float new_g = r * 0.349f + g * 0.686f + b * 0.168f;
            float new_b = r * 0.272f + g * 0.534f + b * 0.131f;
            out[pixel_offset] = clamp(new_b);
            out[pixel_offset + 1] = clamp(new_g);
            out[pixel_offset + 2] = clamp(new_r);
        }
    }
}

extern "C" void run_sepia_filter_cuda(unsigned char* h_output_image, const unsigned char* h_input_image, int width, int height, int stride)
{
    unsigned char* d_in, * d_out;
    int image_size_bytes = stride * height;
    CUDA_CHECK(cudaMalloc((void**)&d_in, image_size_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_out, image_size_bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_input_image, image_size_bytes, cudaMemcpyHostToDevice));
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    sepia_kernel<<<numBlocks, threadsPerBlock>>>(d_out, d_in, width, height, stride);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_output_image, d_out, image_size_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
}

//====================================================================
// NOVO: FILTRO DE INVERSÃO DE CORES
//====================================================================

// Kernel CUDA que inverte a cor de cada pixel
__global__ void inversion_kernel(unsigned char* out, const unsigned char* in, int width, int height, int stride)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height)
    {
        int pixel_offset = row * stride + col * 3;
        if (pixel_offset + 2 < stride * height) {
            // A lógica é simples: 255 - valor_original para cada canal
            out[pixel_offset] = 255 - in[pixel_offset];     // Blue
            out[pixel_offset + 1] = 255 - in[pixel_offset + 1]; // Green
            out[pixel_offset + 2] = 255 - in[pixel_offset + 2]; // Red
        }
    }
}

// Função Wrapper para o filtro de inversão
extern "C" void run_inversion_filter_cuda(unsigned char* h_output_image, const unsigned char* h_input_image, int width, int height, int stride)
{
    unsigned char* d_in, * d_out;
    int image_size_bytes = stride * height;
    CUDA_CHECK(cudaMalloc((void**)&d_in, image_size_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_out, image_size_bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_input_image, image_size_bytes, cudaMemcpyHostToDevice));
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Lança o novo kernel de inversão
    inversion_kernel<<<numBlocks, threadsPerBlock>>>(d_out, d_in, width, height, stride);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_output_image, d_out, image_size_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
}
