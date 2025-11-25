#pragma once
#include <vector>
#include <random>
#include <iostream>
#include <cmath>

/* 

Linear SVM for binary classification, y ∈ {−1, +1}.

Decision function: f(x) = w·x + b, prediction: sign(f(x)).
Objective: minimize  λ/2 ‖w‖² + Σ max(0, 1 − y_i f(x_i))  (hinge loss + L2).

Margin y_i f(x_i) ≥ 1 → no hinge loss, only regularization on w.

SGD update:
  if y_i f(x_i) < 1:  
    w ← w − η(λw − y_i x_i),   b ← b + η y_i
  else:               
    w ← w − ηλw,               b unchanged

*ramdom sampling loop

*/

class LinearSVM {
public:
    std::vector<double> w;
    double b;
    double lr;
    double lambda; // regularization

    LinearSVM(size_t dim, double lr = 0.01, double lambda = 0.01)
        : w(dim, 0.0), b(0.0), lr(lr), lambda(lambda) {}

    double decision_function(const std::vector<double>& x) const {
        double s = b;
        for (size_t i = 0; i < w.size(); ++i) s += w[i] * x[i];
        return s;
    }

    int predict(const std::vector<double>& x) const {
        return decision_function(x) >= 0.0 ? 1 : -1;
    }

    void fit(const std::vector<std::vector<double>>& X,
             const std::vector<int>& y,
             size_t epochs = 50) {

        std::mt19937 rng(123);
        std::uniform_int_distribution<size_t> dist(0, X.size() - 1);

        for (size_t e = 0; e < epochs; ++e) {
            size_t idx = dist(rng);
            const auto& x = X[idx];
            int yi = y[idx];

            double margin = yi * decision_function(x);
            if (margin < 1.0) {
                for (size_t j = 0; j < w.size(); ++j) {
                    w[j] -= lr * (lambda * w[j] - yi * x[j]);
                }
                b += lr * yi;
            } else {
                // regularization only
                for (size_t j = 0; j < w.size(); ++j) {
                    w[j] -= lr * (lambda * w[j]);
                }
            }
        }
    }

    void summary() const {
        std::cout << "[LinearSVM] w = [";
        for (size_t i = 0; i < w.size(); ++i) {
            std::cout << w[i] << (i + 1 < w.size() ? ", " : "");
        }
        std::cout << "], b = " << b << "\n";
    }
};
