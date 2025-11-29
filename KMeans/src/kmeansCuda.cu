#include <cuda_runtime.h>
#include <cstdio>
#include "../include/kmeans.hpp"

__device__
float dev_squared_distance(Point2D a, Point2D b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    return dx * dx + dy * dy;
}

__global__
void kmeans_assign_kernel(const Point2D* points, int n,
                          const Point2D* centroids, int k,
                          int* labels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    Point2D p = points[idx];

    float best_dist = 1e30f;
    int best_k = 0;
    for (int c = 0; c < k; ++c) {
        float d = dev_squared_distance(p, centroids[c]);
        if (d < best_dist) {
            best_dist = d;
            best_k = c;
        }
    }
    labels[idx] = best_k;
}

void kmeans_assign_gpu(const Point2D* h_points, int n,
                       const Point2D* h_centroids, int k,
                       int* h_labels)
{
    Point2D *d_points, *d_centroids;
    int *d_labels;

    size_t points_bytes    = (size_t)n * sizeof(Point2D);
    size_t centroids_bytes = (size_t)k * sizeof(Point2D);
    size_t labels_bytes    = (size_t)n * sizeof(int);

    cudaMalloc(&d_points, points_bytes);
    cudaMalloc(&d_centroids, centroids_bytes);
    cudaMalloc(&d_labels, labels_bytes);

    cudaMemcpy(d_points,   h_points,    points_bytes,    cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids,h_centroids, centroids_bytes, cudaMemcpyHostToDevice);

    int block = 256;
    int grid  = (n + block - 1) / block;
    kmeans_assign_kernel<<<grid, block>>>(d_points, n, d_centroids, k, d_labels);
    cudaDeviceSynchronize();

    cudaMemcpy(h_labels, d_labels, labels_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_points);
    cudaFree(d_centroids);
    cudaFree(d_labels);
}