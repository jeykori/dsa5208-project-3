#include "pca_serial.hpp"
#include "pca_core.hpp"
#include <iostream>
#include <cmath>
#include <vector>

void standardize_serial(std::vector<double>& data, int rows, int cols, bool scale) {
    // Mean
    std::vector<double> means(cols, 0.0);
    for (int c = 0; c < cols; ++c) {
        double sum = 0.0;
        for (int r = 0; r < rows; ++r) sum += data[r * cols + c];
        means[c] = sum / rows;
    }

    // Default divisor is 1.0
    std::vector<double> std_devs(cols, 1.0);
    // Standard deviation (only if needed)
    if (scale && rows > 1) {
        for (int c = 0; c < cols; ++c) {
            double sum_sq = 0.0;
            for (int r = 0; r < rows; ++r) {
                double diff = data[r * cols + c] - means[c];
                sum_sq += diff * diff;
            }
            std_devs[c] = std::sqrt(sum_sq / (rows - 1));

            // Prevent division by zero if a feature column is completely constant
            if (std_devs[c] == 0.0) std_devs[c] = 1.0;
        }
    }

    for (int c = 0; c < cols; ++c) {
        for (int r = 0; r < rows; ++r) {
            data[r * cols + c] = (data[r * cols + c] - means[c]) / std_devs[c];
        }
    }
}

std::vector<double> compute_covariance_serial(const std::vector<double>& data, int rows, int cols) {
    std::vector<double> cov(cols * cols, 0.0);
    for (int i = 0; i < cols; ++i) {
        for (int j = i; j < cols; ++j) {
            double sum = 0.0;
            for (int r = 0; r < rows; ++r) sum += data[r * cols + i] * data[r * cols + j];
            cov[i * cols + j] = sum / (rows - 1);
            if (i != j) cov[j * cols + i] = cov[i * cols + j];
        }
    }
    return cov;
}

void compute_eigenvectors_serial(std::vector<double>& cov, int rows, int cols, int top_k) {
    double total_var = 0.0;
    for (int i = 0; i < cols; ++i) total_var += cov[i * cols + i];

    for (int k = 0; k < top_k; ++k) {
        std::vector<double> q(cols, 1.0);
        normalize(q);
        double eigenvalue = 0.0;

        for (int iter = 0; iter < 1000; ++iter) {
            std::vector<double> z(cols, 0.0);
            for (int i = 0; i < cols; ++i) {
                for (int j = 0; j < cols; ++j) z[i] += cov[i * cols + j] * q[j];
            }
            eigenvalue = 0.0;
            for (int i = 0; i < cols; ++i) eigenvalue += q[i] * z[i];
            q = z;
            normalize(q);
        }
        for (int i = 0; i < cols; ++i) {
            for (int j = 0; j < cols; ++j) cov[i * cols + j] -= eigenvalue * q[i] * q[j];
        }
        std::cout << "Serial PC" << k + 1 << ": Eigenvalue = " << eigenvalue << "\n";
    }
}