#ifndef PCA_PARALLEL_OPTIMIZED_HPP
#define PCA_PARALLEL_OPTIMIZED_HPP
#include <vector>

void standardize_parallel(std::vector<double>& local_data, int local_rows, int total_rows, int cols, bool scale);
std::vector<double> compute_covariance_parallel(const std::vector<double>& local_data, int local_rows, int total_rows, int cols);
void compute_eigenvectors_parallel(std::vector<double>& global_cov, int total_rows, int cols, int top_k, int world_rank);

#endif

