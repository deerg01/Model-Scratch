#include <cuda_runtime.h>
#include <cstdio>
#include "../include/layers.hpp"

__global__
void conv2d_kernel(const float* __restrict__ input,
                   const float* __restrict__ weight,
                   const float* __restrict__ bias,
                   float* __restrict__ output,
                   int N, int C_in, int H, int W,
                   int C_out, int K, int outH, int outW)
{
    int n = blockIdx.z;
    int co = blockIdx.y;

    int index = blockIdx.x;
    int oh = index / outW;
    int ow = index % outW;

    if (n >= N || co >= C_out || oh >= outH || ow >= outW) return;

    float sum = bias[co];

    for (int ci = 0; ci < C_in; ++ci) {
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                int ih = oh + kh;
                int iw = ow + kw;

                int in_idx = ((n * C_in + ci) * H + ih) * W + iw;
                int w_idx  = ((co * C_in + ci) * K + kh) * K + kw;

                sum += input[in_idx] * weight[w_idx];
            }
        }
    }

    int out_idx = ((n * C_out + co) * outH + oh) * outW + ow;
    output[out_idx] = sum;
}

__global__
void relu_kernel(float* data, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) data[idx] = max(data[idx], 0.0f);
}

__global__
void maxpool2x2_kernel(const float* input, float* output,
                       int N, int C, int H, int W,
                       int outH, int outW)
{
    int n = blockIdx.z;
    int c = blockIdx.y;

    int idx = blockIdx.x;
    int oh = idx / outW;
    int ow = idx % outW;

    if (n >= N || c >= C || oh >= outH || ow >= outW) return;

    int ih0 = oh * 2;
    int iw0 = ow * 2;

    float m = -1e30f;
    for (int kh = 0; kh < 2; ++kh)
        for (int kw = 0; kw < 2; ++kw) {
            int ih = ih0 + kh;
            int iw = iw0 + kw;
            int in_idx = ((n * C + c) * H + ih) * W + iw;
            m = max(m, input[in_idx]);
        }

    int out_idx = ((n * C + c) * outH + oh) * outW + ow;
    output[out_idx] = m;
}

void conv2d_fo(const float* input, const float* weight,
                         const float* bias, float* output,
                         int N, int C_in, int H, int W,
                         int C_out, int K, int outH, int outW)
{
    size_t in_bytes  = (size_t)N*C_in*H*W*sizeof(float);
    size_t w_bytes   = (size_t)C_out*C_in*K*K*sizeof(float);
    size_t b_bytes   = (size_t)C_out*sizeof(float);
    size_t out_bytes = (size_t)N*C_out*outH*outW*sizeof(float);

    float *d_in, *d_w, *d_b, *d_out;
    cudaMalloc(&d_in, in_bytes);
    cudaMalloc(&d_w, w_bytes);
    cudaMalloc(&d_b, b_bytes);
    cudaMalloc(&d_out, out_bytes);

    cudaMemcpy(d_in, input,  in_bytes,  cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, weight,  w_bytes,   cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, bias,    b_bytes,   cudaMemcpyHostToDevice);

    dim3 grid(outH*outW, C_out, N);
    conv2d_kernel<<<grid, 1>>>(d_in, d_w, d_b, d_out,
                               N, C_in, H, W,
                               C_out, K, outH, outW);
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_out, out_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_w);
    cudaFree(d_b);
    cudaFree(d_out);
}

void relu_fo(float* data, int total) {
    float* d_data;
    size_t bytes = (size_t)total * sizeof(float);
    cudaMalloc(&d_data, bytes);
    cudaMemcpy(d_data, data, bytes, cudaMemcpyHostToDevice);

    int block = 256;
    int grid = (total + block - 1) / block;
    relu_kernel<<<grid, block>>>(d_data, total);
    cudaDeviceSynchronize();

    cudaMemcpy(data, d_data, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_data);
}

void maxpool2x2_fo(const float* input, float* output,
                              int N, int C, int H, int W,
                              int outH, int outW)
{
    size_t in_bytes  = (size_t)N*C*H*W*sizeof(float);
    size_t out_bytes = (size_t)N*C*outH*outW*sizeof(float);

    float *d_in, *d_out;
    cudaMalloc(&d_in, in_bytes);
    cudaMalloc(&d_out, out_bytes);

    cudaMemcpy(d_in, input, in_bytes, cudaMemcpyHostToDevice);

    dim3 grid(outH*outW, C, N);
    maxpool2x2_kernel<<<grid, 1>>>(d_in, d_out, N, C, H, W, outH, outW);
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_out, out_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
}
