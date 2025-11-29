#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include "kmeans.hpp"

int main() {
    std::cout << "Project 3: CPU vs GPU perfomence (K-Means)\n";

    const int n_points  = 100000;  // total points
    const int k         = 4;       // clusters
    const int max_iters = 10;

    std::vector<Point2D> points(n_points);
    std::mt19937 rng(0);
    std::normal_distribution<float> d1x(-5.f, 1.0f);
    std::normal_distribution<float> d1y( 0.f, 1.0f);
    std::normal_distribution<float> d2x( 5.f, 1.0f);
    std::normal_distribution<float> d2y( 0.f, 1.0f);

    for (int i = 0; i < n_points; ++i) {
        if (i < n_points / 2) {
            points[i] = { d1x(rng), d1y(rng) };
        } else {
            points[i] = { d2x(rng), d2y(rng) };
        }
    }

    std::vector<Point2D> centroids_cpu;
    std::vector<int>     labels_cpu;

    KMeansCPU km_cpu(k, max_iters);

    auto t0 = std::chrono::high_resolution_clock::now();
    km_cpu.fit(points, centroids_cpu, labels_cpu);
    auto t1 = std::chrono::high_resolution_clock::now();

    auto cpu_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    std::cout << "\n[CPU K-Means]\n";
    std::cout << "Time: " << cpu_ms << " ms\n";
    std::cout << "Centroids:\n";
    for (int c = 0; c < k; ++c) {
        std::cout << "  c" << c << ": (" << centroids_cpu[c].x
                  << ", " << centroids_cpu[c].y << ")\n";
    }

    std::vector<Point2D> centroids_gpu = centroids_cpu; 
    std::vector<int>     labels_gpu(n_points, 0);

    t0 = std::chrono::high_resolution_clock::now();

    for (int it = 0; it < max_iters; ++it) {
        kmeans_assign_gpu(points.data(), n_points,
                          centroids_gpu.data(), k,
                          labels_gpu.data());

        std::vector<float> sum_x(k, 0.0f);
        std::vector<float> sum_y(k, 0.0f);
        std::vector<int>   count(k, 0);

        for (int i = 0; i < n_points; ++i) {
            int c = labels_gpu[i];
            sum_x[c] += points[i].x;
            sum_y[c] += points[i].y;
            count[c] += 1;
        }

        for (int c = 0; c < k; ++c) {
            if (count[c] > 0) {
                centroids_gpu[c].x = sum_x[c] / count[c];
                centroids_gpu[c].y = sum_y[c] / count[c];
            }
        }
    }

    t1 = std::chrono::high_resolution_clock::now();
    auto gpu_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    std::cout << "\n[Hybrid K-Means: GPU assignment + CPU update]\n";
    std::cout << "Time: " << gpu_ms << " ms\n";
    std::cout << "Centroids:\n";
    for (int c = 0; c < k; ++c) {
        std::cout << "  c" << c << ": (" << centroids_gpu[c].x
                  << ", " << centroids_gpu[c].y << ")\n";
    }

    return 0;
}