#pragma once
#include <vector>
#include <random>
#include <limits>
#include <cmath>
#include <iostream>

struct Point2D {
    float x, y;
};

inline float squared_distance(const Point2D& a, const Point2D& b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    return dx * dx + dy * dy;
}

struct KMeansCPU {
    int k;
    int max_iters;

    KMeansCPU(int k, int max_iters = 20)
        : k(k), max_iters(max_iters) {}

    void fit(const std::vector<Point2D>& points,
             std::vector<Point2D>& centroids,
             std::vector<int>& labels)
    {
        const int n = static_cast<int>(points.size());
        labels.assign(n, 0);

        centroids.resize(k);
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> dist(0, n - 1);
        for (int i = 0; i < k; ++i) {
            centroids[i] = points[dist(rng)];
        }

        for (int it = 0; it < max_iters; ++it) {
            for (int i = 0; i < n; ++i) {
                float best_dist = std::numeric_limits<float>::max();
                int best_k = 0;
                for (int c = 0; c < k; ++c) {
                    float d = squared_distance(points[i], centroids[c]);
                    if (d < best_dist) {
                        best_dist = d;
                        best_k = c;
                    }
                }
                labels[i] = best_k;
            }

            std::vector<float> sum_x(k, 0.0f);
            std::vector<float> sum_y(k, 0.0f);
            std::vector<int> count(k, 0);

            for (int i = 0; i < n; ++i) {
                int c = labels[i];
                sum_x[c] += points[i].x;
                sum_y[c] += points[i].y;
                count[c] += 1;
            }

            for (int c = 0; c < k; ++c) {
                if (count[c] > 0) {
                    centroids[c].x = sum_x[c] / count[c];
                    centroids[c].y = sum_y[c] / count[c];
                }
            }
        }
    }
};

void kmeans_assign_gpu(const Point2D* h_points, int n,
                       const Point2D* h_centroids, int k,
                       int* h_labels);