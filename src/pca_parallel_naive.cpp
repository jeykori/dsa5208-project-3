#include "pca_parallel_naive.hpp"
#include "pca_core.hpp"
#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <cmath>
#include <vector>

void standardize_parallel(std::vector<double>& local_data, int local_rows, int total_rows, int cols, bool scale) {
    // Mean
    std::vector<double> local_sums(cols, 0.0);
    std::vector<double> global_means(cols, 0.0);

    #pragma omp parallel for
    for (int c = 0; c < cols; ++c) {
        double sum = 0.0;
        for (int r = 0; r < local_rows; ++r) sum += local_data[r * cols + c];
        local_sums[c] = sum;
    }

    MPI_Allreduce(local_sums.data(), global_means.data(), cols, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    #pragma omp parallel for
    for (int c = 0; c < cols; ++c) global_means[c] /= total_rows;

    // Default divisor is 1.0
    std::vector<double> global_std_devs(cols, 1.0);
    // Standard deviation (only if needed)
    if (scale && total_rows > 1) {
        std::vector<double> local_sq_sums(cols, 0.0);
        std::vector<double> global_sq_sums(cols, 0.0);

        #pragma omp parallel for
        for (int c = 0; c < cols; ++c) {
            double sq_sum = 0.0;
            for (int r = 0; r < local_rows; ++r) {
                double diff = local_data[r * cols + c] - global_means[c];
                sq_sum += diff * diff;
            }
            local_sq_sums[c] = sq_sum;
        }

        MPI_Allreduce(local_sq_sums.data(), global_sq_sums.data(), cols, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        #pragma omp parallel for
        for (int c = 0; c < cols; ++c) {
            global_std_devs[c] = std::sqrt(global_sq_sums[c] / (total_rows - 1));
            if (global_std_devs[c] == 0.0) global_std_devs[c] = 1.0;
        }
    }

    #pragma omp parallel for
    for (int c = 0; c < cols; ++c) {
        for (int r = 0; r < local_rows; ++r) {
            local_data[r * cols + c] = (local_data[r * cols + c] - global_means[c]) / global_std_devs[c];
        }
    }
}

std::vector<double> compute_covariance_parallel(const std::vector<double>& local_data, int local_rows, int total_rows, int cols) {
    std::vector<double> local_cov(cols * cols, 0.0);
    std::vector<double> global_cov(cols * cols, 0.0);

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < cols; ++i) {
        for (int j = i; j < cols; ++j) {
            double sum = 0.0;
            for (int r = 0; r < local_rows; ++r) sum += local_data[r * cols + i] * local_data[r * cols + j];
            local_cov[i * cols + j] = sum;
            if (i != j) local_cov[j * cols + i] = sum;
        }
    }

    MPI_Allreduce(local_cov.data(), global_cov.data(), cols * cols, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // Scale by N-1
    if (total_rows > 1) {
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < cols; ++i) {
            for (int j = 0; j < cols; ++j) global_cov[i * cols + j] /= (total_rows - 1);
        }
    }
    return global_cov;
}

void compute_eigenvectors_parallel(std::vector<double>& global_cov, int total_rows, int cols, int top_k, int world_rank) {
    if (world_rank == 0) {
        double total_variance = 0.0;
        #pragma omp parallel for reduction(+:total_variance)
        for (int i = 0; i < cols; ++i) total_variance += global_cov[i * cols + i];

        for (int k = 0; k < top_k; ++k) {
            std::vector<double> q(cols, 1.0);
            normalize(q);
            double eigenvalue = 0.0;

            for (int iter = 0; iter < 1000; ++iter) {
                std::vector<double> z(cols, 0.0);
                #pragma omp parallel for
                for (int i = 0; i < cols; ++i) {
                    double local_z = 0.0;
                    for (int j = 0; j < cols; ++j) local_z += global_cov[i * cols + j] * q[j];
                    z[i] = local_z;
                }

                eigenvalue = 0.0;
                #pragma omp parallel for reduction(+:eigenvalue)
                for (int i = 0; i < cols; ++i) eigenvalue += q[i] * z[i];
                q = z;
                normalize(q);
            }

            #pragma omp parallel for collapse(2)
            for (int i = 0; i < cols; ++i) {
                for (int j = 0; j < cols; ++j) global_cov[i * cols + j] -= eigenvalue * q[i] * q[j];
            }
            std::cout << "Parallel PC" << k + 1 << ": Eigenvalue = " << eigenvalue << "\n";
        }
    }
}

