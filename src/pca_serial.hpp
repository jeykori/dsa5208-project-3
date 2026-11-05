#ifndef PCA_SERIAL_HPP
#define PCA_SERIAL_HPP
#include <vector>

void standardize_serial(std::vector<double>& data, int rows, int cols, bool scale);
std::vector<double> compute_covariance_serial(const std::vector<double>& data, int rows, int cols);
void compute_eigenvectors_serial(std::vector<double>& cov, int rows, int cols, int top_k);

#endif