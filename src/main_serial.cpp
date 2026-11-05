#include "pca_core.hpp"
#include "pca_serial.hpp"
#include <iostream>
#include <vector>
#include <chrono>

int main(int argc, char** argv) {
    CmdArgs args;
    if (!parse_args(argc, argv, args)) {
        print_usage(argv[0]);
        return 1;
    }

    int rows, cols;
    if (!get_dataset_dimensions(args.data_path, rows, cols)) {
        std::cerr << "Failed to parse dimensions from " << args.data_path << "\n";
        return 1;
    }

    auto t_start = std::chrono::high_resolution_clock::now();

    std::vector<double> data(rows * cols, 0.0);
    load_data_chunk(args.data_path, data, 0, rows, cols);
    auto t_io = std::chrono::high_resolution_clock::now();

    standardize_serial(data, rows, cols, args.scale);
    auto t_mean = std::chrono::high_resolution_clock::now();

    std::vector<double> cov = compute_covariance_serial(data, rows, cols);
    auto t_cov = std::chrono::high_resolution_clock::now();

    compute_eigenvectors_serial(cov, rows, cols, args.top_k);
    auto t_eigen = std::chrono::high_resolution_clock::now();

    // Lambda to convert chrono time_points to double timestamps (seconds)
    auto to_sec = [](std::chrono::high_resolution_clock::time_point t) {
        return std::chrono::duration<double>(t.time_since_epoch()).count();
    };

    print_and_save_timings("serial", args.name, 1, 1, rows, cols,
                to_sec(t_start), to_sec(t_io), to_sec(t_mean), to_sec(t_cov), to_sec(t_eigen));

    return 0;
}