#pragma once
#include <vector>
#include <random>
#include <iostream>

class Perceptron {
public:
    std::vector<double> w;
    double b;
    double lr;

    Perceptron(size_t dim, double lr = 0.1)
        : w(dim, 0.0), b(0.0), lr(lr) {}

    double predict_raw(const std::vector<double>& x) const {
        double s = b;
        for (size_t i = 0; i < w.size(); ++i) s += w[i] * x[i];
        return s;
    }

    int predict(const std::vector<double>& x) const {
        return predict_raw(x) >= 0.0 ? 1 : -1;
    }

    void fit(const std::vector<std::vector<double>>& X,
             const std::vector<int>& y,
             size_t epochs = 20) {
        std::mt19937 rng(42);
        std::uniform_int_distribution<size_t> dist(0, X.size() - 1);

        for (size_t e = 0; e < epochs; ++e) {
            size_t idx = dist(rng);
            int y_pred = predict(X[idx]);
            if (y_pred != y[idx]) {
                int error = y[idx];
                for (size_t j = 0; j < w.size(); ++j) {
                    w[j] += lr * error * X[idx][j];
                }
                b += lr * error;
            }
        }
    }

    void summary() const {
        std::cout << "[Perceptron] w = [";
        for (size_t i = 0; i < w.size(); ++i) {
            std::cout << w[i] << (i + 1 < w.size() ? ", " : "");
        }
        std::cout << "], b = " << b << "\n";
    }
};
