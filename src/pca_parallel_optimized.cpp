#include "pca_parallel_optimized.hpp"
#include "pca_core.hpp"
#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <cmath>
#include <vector>
#include <algorithm>

void standardize_parallel(std::vector<double>& local_data, int local_rows, int total_rows, int cols, bool scale) {
    // 1. Mean calculation: iterate rows on outside, cols on inside for unit-stride access
    std::vector<double> local_sums(cols, 0.0);
    std::vector<double> global_means(cols, 0.0);

    #pragma omp parallel
    {
        std::vector<double> thread_sums(cols, 0.0);
        #pragma omp for
        for (int r = 0; r < local_rows; ++r) {
            const double* row_ptr = &local_data[r * cols];
            for (int c = 0; c < cols; ++c) {
                thread_sums[c] += row_ptr[c];
            }
        }
        #pragma omp critical
        {
            for (int c = 0; c < cols; ++c) {
                local_sums[c] += thread_sums[c];
            }
        }
    }

    MPI_Allreduce(local_sums.data(), global_means.data(), cols, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    #pragma omp parallel for
    for (int c = 0; c < cols; ++c) {
        global_means[c] /= total_rows;
    }

    // Default divisor is 1.0
    std::vector<double> global_std_devs(cols, 1.0);

    // 2. Standard deviation calculation (only if needed)
    if (scale && total_rows > 1) {
        std::vector<double> local_sq_sums(cols, 0.0);
        std::vector<double> global_sq_sums(cols, 0.0);

        #pragma omp parallel
        {
            std::vector<double> thread_sq_sums(cols, 0.0);
            #pragma omp for
            for (int r = 0; r < local_rows; ++r) {
                const double* row_ptr = &local_data[r * cols];
                for (int c = 0; c < cols; ++c) {
                    double diff = row_ptr[c] - global_means[c];
                    thread_sq_sums[c] += diff * diff;
                }
            }
            #pragma omp critical
            {
                for (int c = 0; c < cols; ++c) {
                    local_sq_sums[c] += thread_sq_sums[c];
                }
            }
        }

        MPI_Allreduce(local_sq_sums.data(), global_sq_sums.data(), cols, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        #pragma omp parallel for
        for (int c = 0; c < cols; ++c) {
            global_std_devs[c] = std::sqrt(global_sq_sums[c] / (total_rows - 1));
            if (global_std_devs[c] == 0.0) global_std_devs[c] = 1.0;
        }
    }

    // 3. Standardization / scaling with unit-stride access
    #pragma omp parallel for
    for (int r = 0; r < local_rows; ++r) {
        double* row_ptr = &local_data[r * cols];
        for (int c = 0; c < cols; ++c) {
            row_ptr[c] = (row_ptr[c] - global_means[c]) / global_std_devs[c];
        }
    }
}

std::vector<double> compute_covariance_parallel(const std::vector<double>& local_data, int local_rows, int total_rows, int cols) {
    std::vector<double> local_cov(cols * cols, 0.0);
    std::vector<double> global_cov(cols * cols, 0.0);

    const double* d_data = local_data.data();
    double* d_cov = local_cov.data();
    int total_elements = local_rows * cols;
    int cov_elements = cols * cols;

    // Detect GPU accelerator devices; gracefully fall back to CPU host execution if not found
    int num_devices = omp_get_num_devices();
    bool has_gpu = (num_devices > 0);

    // 1. Compute upper triangular portion of local_cov via OpenMP GPU target offloading
    //    Explicitly maps input data matrix to device and resulting covariance matrix tofrom device
    #pragma omp target teams distribute parallel for if(target: has_gpu) \
        map(to: d_data[0:total_elements]) \
        map(tofrom: d_cov[0:cov_elements])
    for (int i = 0; i < cols; ++i) {
        for (int r = 0; r < local_rows; ++r) {
            double val_i = d_data[r * cols + i];
            const double* row_ptr = &d_data[r * cols];
            for (int j = i; j < cols; ++j) {
                d_cov[i * cols + j] += val_i * row_ptr[j];
            }
        }
    }

    // 2. Safely mirror upper triangular portion to lower triangular portion (eliminates false sharing)
    #pragma omp parallel for
    for (int i = 0; i < cols; ++i) {
        for (int j = 0; j < i; ++j) {
            local_cov[i * cols + j] = local_cov[j * cols + i];
        }
    }

    // 3. Matrix Allreduce to sum local_cov into full DxD global_cov
    MPI_Allreduce(local_cov.data(), global_cov.data(), cols * cols, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // 4. Scale by total_rows - 1
    if (total_rows > 1) {
        double inv_n = 1.0 / (total_rows - 1);
        #pragma omp parallel for
        for (int idx = 0; idx < cols * cols; ++idx) {
            global_cov[idx] *= inv_n;
        }
    }

    return global_cov;
}

void compute_eigenvectors_parallel(std::vector<double>& global_cov, int total_rows, int cols, int top_k, int world_rank) {
    if (world_rank == 0) {
        double total_variance = 0.0;
        #pragma omp parallel for reduction(+:total_variance)
        for (int i = 0; i < cols; ++i) total_variance += global_cov[i * cols + i];

        // Hoist allocations outside the power iteration loops
        std::vector<double> q(cols, 1.0);
        std::vector<double> z(cols, 0.0);

        for (int k = 0; k < top_k; ++k) {
            std::fill(q.begin(), q.end(), 1.0);
            normalize(q);
            double eigenvalue = 0.0;

            for (int iter = 0; iter < 1000; ++iter) {
                // Local OpenMP matrix-vector multiplication with contiguous memory access
                #pragma omp parallel for
                for (int i = 0; i < cols; ++i) {
                    double local_z = 0.0;
                    const double* row_ptr = &global_cov[i * cols];
                    for (int j = 0; j < cols; ++j) {
                        local_z += row_ptr[j] * q[j];
                    }
                    z[i] = local_z;
                }

                // Compute Rayleigh quotient eigenvalue
                eigenvalue = 0.0;
                #pragma omp parallel for reduction(+:eigenvalue)
                for (int i = 0; i < cols; ++i) {
                    eigenvalue += q[i] * z[i];
                }

                // Swap pointers without heap allocation, then normalize
                std::swap(q, z);
                normalize(q);
            }

            // Deflation: subtract eigenvalue * q * q^T from global_cov
            #pragma omp parallel for collapse(2)
            for (int i = 0; i < cols; ++i) {
                for (int j = 0; j < cols; ++j) {
                    global_cov[i * cols + j] -= eigenvalue * q[i] * q[j];
                }
            }

            std::cout << "Parallel PC" << k + 1 << ": Eigenvalue = " << eigenvalue << "\n";
        }
    }
}

