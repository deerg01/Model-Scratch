#pragma once
#include <vector>
#include <stdexcept>
#include <iostream>
#include <iomanip>

/* 

Minimal 2D matrix wrapper for ML demo project

Row-major storage: data[r * cols + c]
Supports basic ops: ctor, element access, transpose, matmul

Used to keep linear algebra notation explicit >>> (A(i,j), A^T, A·B)
optim해야되는데 일단 좀 미룰게 ^~^

*/

class Matrix {
public:
    size_t rows, cols;
    std::vector<double> data;

    Matrix() : rows(0), cols(0) {}

    Matrix(size_t r, size_t c, double init = 0.0)
        : rows(r), cols(c), data(r * c, init) {}

    double& operator()(size_t r, size_t c) {
        return data[r * cols + c];
    }

    double operator()(size_t r, size_t c) const {
        return data[r * cols + c];
    }

    static Matrix from_vector(const std::vector<double>& v, bool as_column = true) {
        if (as_column) {
            Matrix m(v.size(), 1);
            for (size_t i = 0; i < v.size(); ++i) m(i, 0) = v[i];
            return m;
        } else {
            Matrix m(1, v.size());
            for (size_t i = 0; i < v.size(); ++i) m(0, i) = v[i];
            return m;
        }
    }

    static Matrix matmul(const Matrix& A, const Matrix& B) {
        if (A.cols != B.rows) {
            throw std::runtime_error("SVM ERROR(MATRIX):: dimension mismatch for matmul");
        }
        Matrix C(A.rows, B.cols, 0.0);
        for (size_t i = 0; i < A.rows; ++i) {
            for (size_t k = 0; k < A.cols; ++k) {
                double a = A(i, k);
                for (size_t j = 0; j < B.cols; ++j) {
                    C(i, j) += a * B(k, j);
                }
            }
        }
        return C;
    }

    static Matrix transpose(const Matrix& A) {
        Matrix T(A.cols, A.rows);
        for (size_t i = 0; i < A.rows; ++i)
            for (size_t j = 0; j < A.cols; ++j)
                T(j, i) = A(i, j);
        return T;
    }

    void print(const std::string& name = "[Matrix]") const {
        std::cout << name << " (" << rows << "x" << cols << "):\n";
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                std::cout << std::setw(10) << operator()(i, j) << " ";
            }
            std::cout << "\n";
        }
    }
};
