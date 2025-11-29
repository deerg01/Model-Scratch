#pragma once
#include "tensor.hpp"
#include <vector>
#include <random>
#include <iostream>

void conv2d_forward_cuda(const float* input, const float* weight,
                         const float* bias, float* output,
                         int N, int C_in, int H, int W,
                         int C_out, int K, int outH, int outW);

void relu_forward_cuda(float* data, int total);

void maxpool2x2_forward_cuda(const float* input, float* output,
                              int N, int C, int H, int W,
                              int outH, int outW);

struct Conv2d {
    int in_channels;
    int out_channels;
    int kernel_size;

    Tensor weight; // (out_channels, in_channels, K, K)
    Tensor bias;   // (1, out_channels, 1, 1)

    Conv2d(int in_c, int out_c, int k)
        : in_channels(in_c),
          out_channels(out_c),
          kernel_size(k),
          weight(out_c, in_c, k, k),
          bias(1, out_c, 1, 1)
    {
        weight.random_normal(0.0f, 0.1f);
        bias.random_normal(0.0f, 0.01f);
    }

    Tensor forward(const Tensor& x) const {
        int N = x.n;
        int C_in = x.c;
        int H = x.h;
        int W = x.w;

        int K = kernel_size;
        int C_out = out_channels;
        int outH = H - K + 1;
        int outW = W - K + 1;

        Tensor y(N, C_out, outH, outW);

        conv2d_forward_cuda(
            x.data.data(), weight.data.data(), bias.data.data(),
            y.data.data(),
            N, C_in, H, W, C_out, K, outH, outW
        );

        return y;
    }
};

struct ReLU {
    Tensor forward(Tensor x) const {
        relu_forward_cuda(x.data.data(), (int)x.size());
        return x;
    }
};

struct MaxPool2x2 {
    Tensor forward(const Tensor& x) const {
        int N = x.n;
        int C = x.c;
        int H = x.h;
        int W = x.w;

        int outH = H / 2;
        int outW = W / 2;

        Tensor y(N, C, outH, outW);

        maxpool2x2_forward_cuda(
            x.data.data(), y.data.data(),
            N, C, H, W,
            outH, outW
        );

        return y;
    }
};


struct Flatten {
    Tensor forward(const Tensor& x) const {
        int N = x.n;
        int C = x.c;
        int H = x.h;
        int W = x.w;

        int feat = C * H * W;
        Tensor y(N, 1, 1, feat);

        y.data = x.data;
        return y;
    }
};

struct LinearCPU {
    int in_dim;
    int out_dim;
    std::vector<float> W; // row-major: [out_dim][in_dim]
    std::vector<float> b; // [out_dim]

    LinearCPU(int in_d, int out_d)
        : in_dim(in_d),
          out_dim(out_d),
          W(out_d * in_d),
          b(out_d)
    {
        std::mt19937 rng(123);
        std::normal_distribution<float> wdist(0.0f, 0.1f);
        std::normal_distribution<float> bdist(0.0f, 0.01f);

        for (auto& w : W) w = wdist(rng);
        for (auto& bb : b) bb = bdist(rng);
    }

    std::vector<float> forward(const Tensor& x) const {
        if (x.n != 1 || x.c != 1 || x.h != 1 || x.w != in_dim) {
            std::cerr << "[LinearCPU] Unexpected input shape: ("
                      << x.n << "," << x.c << "," << x.h << "," << x.w << ")\n";
        }

        const std::vector<float>& in = x.data;
        std::vector<float> out(out_dim, 0.0f);

        for (int o = 0; o < out_dim; ++o) {
            float sum = b[o];
            const float* w_row = &W[o * in_dim];
            for (int i = 0; i < in_dim; ++i) {
                sum += w_row[i] * in[i];
            }
            out[o] = sum;
        }
        return out;
    }
};