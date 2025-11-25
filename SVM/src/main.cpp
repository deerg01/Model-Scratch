#include <iostream>
#include <vector>
#include <random>

#include "perceptron.hpp"
#include "svm.hpp"

struct Sample {
    std::vector<double> x;
    int y;
};

int main() {
    std::cout << "Project 1 - Perceptron & Linear SVM\n";

    // 2D Gaussian Cluster
    std::mt19937 rng(0);
    std::normal_distribution<double> dist_pos_x(2.0, 1.0);
    std::normal_distribution<double> dist_pos_y(2.0, 1.0);
    std::normal_distribution<double> dist_neg_x(-2.0, 1.0);
    std::normal_distribution<double> dist_neg_y(-2.0, 1.0);

    std::vector<std::vector<double>> X;
    std::vector<int> y;

    const int n_per_class = 10000;

    for (int i = 0; i < n_per_class; ++i) {
        X.push_back({dist_pos_x(rng), dist_pos_y(rng)});
        y.push_back(1);
    }
    for (int i = 0; i < n_per_class; ++i) {
        X.push_back({dist_neg_x(rng), dist_neg_y(rng)});
        y.push_back(-1);
    }

    Perceptron p(2, 0.1);
    p.fit(X, y, 2000);
    p.summary();

    LinearSVM svm(2, 0.01, 0.01);
    svm.fit(X, y, 5000);
    svm.summary();

    auto accuracy = [&](auto& model) {
        int correct = 0;
        for (size_t i = 0; i < X.size(); ++i) {
            if (model.predict(X[i]) == y[i]) correct++;
        }
        return (double)correct / (double)X.size();
    };

    std::cout << "Perceptron train acc: " << accuracy(p) << "\n";
    std::cout << "SVM train acc: " << accuracy(svm) << "\n";

    return 0;
}
