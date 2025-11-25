#pragma once
#include <vector>
#include <iostream>
#include <random>

struct Tensor {
    int n, c, h, w;
    std::vector<float> data;

    Tensor() : n(0), c(0), h(0), w(0) {}

    Tensor(int n, int c, int h, int w)
        : n(n), c(c), h(h), w(w), data(n * c * h * w, 0.0f) {}

    inline float& operator()(int ni, int ci, int hi, int wi) {
        return data[((ni * c + ci) * h + hi) * w + wi];
    }

    inline const float& operator()(int ni, int ci, int hi, int wi) const {
        return data[((ni * c + ci) * h + hi) * w + wi];
    }

    size_t size() const { return data.size(); }

    void random_normal(float mean = 0.0f, float stddev = 1.0f) {
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(mean, stddev);
        for (auto& v : data) v = dist(rng);
    }

    void print_shape(const std::string& name = "Tensor") const {
        std::cout << name << " shape: (" << n << ", " << c
                  << ", " << h << ", " << w << ")\n";
    }
};
