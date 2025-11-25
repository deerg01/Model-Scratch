#pragma once
#include "tensor.hpp"

// _fo : forward
void conv2d_fo(const float* input, const float* weight,
                         const float* bias, float* output,
                         int N, int C_in, int H, int W,
                         int C_out, int K, int outH, int outW);

void relu_fo(float* data, int total);

void maxpool2x2_fo(const float* input, float* output,
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

        conv2d_fo(
            x.data.data(), weight.data.data(), bias.data.data(),
            y.data.data(),
            N, C_in, H, W, C_out, K, outH, outW
        );

        return y;
    }
};

struct ReLU {
    Tensor forward(Tensor x) const {
        relu_fo(x.data.data(), (int)x.size());
        return x;
    }
};

struct MaxPool2x2 {
    Tensor forward(const Tensor& x) const {
        int N = x.n;
        int C = x.c;
        int H = x.h;
        int W = x.w;

        Tensor y(N, C, H/2, W/2);

        maxpool2x2_fo(
            x.data.data(), y.data.data(),
            N, C, H, W,
            H/2, W/2
        );

        return y;
    }
};
