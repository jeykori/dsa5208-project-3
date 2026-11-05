#include "pca_core.hpp"
#include <iostream>
#include <sys/stat.h>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>

void print_usage(const std::string& program_name) {
    std::cerr << "Usage: " << program_name << " <data_path> <top_k> <scale>\n"
              << "  scale=0 : Mean centering only (e.g., for UCI HAR)\n"
              << "  scale=1 : Full Z-score standardization (e.g., for YearPredictionMSD)\n";
}

bool parse_args(int argc, char** argv, CmdArgs& args, bool print /*= true*/) {
    if (argc < 4) return false;

    try {
        args.data_path = argv[1];
        args.top_k = std::stoi(argv[2]);
        args.scale = (std::stoi(argv[3]) != 0);
        args.name = argv[4];
        if (print) {
            std::cout << "Run Name: " << args.name
                      << " | Top K: " << args.top_k
                      << " | Scale: " << args.scale << "\n";
        }
        return true;
    } catch (...) {
        return false;
    }
}

void normalize(std::vector<double>& vec) {
    double norm = 0.0;
    for (double val : vec) norm += val * val;
    norm = std::sqrt(norm);
    for (double& val : vec) val /= norm;
}

bool get_dataset_dimensions(const std::string& filepath, int& rows, int& cols) {
    std::ifstream file(filepath);
    if (!file.is_open()) return false;

    std::string line;
    rows = 0; cols = 0;

    if (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string tmp;
        while (ss >> tmp) cols++;
        rows++;
    }

    while (std::getline(file, line)) {
        if (!line.empty()) rows++;
    }
    file.close();

    std::cout << "Dataset: " << rows << " rows, " << cols << " cols.\n\n";
    return true;
}

bool load_data_chunk(const std::string& filepath, std::vector<double>& local_data, int skip_rows, int read_rows, int cols) {
    std::ifstream file(filepath);
    if (!file.is_open()) return false;

    std::string dummy;
    for (int i = 0; i < skip_rows * cols; ++i) file >> dummy;
    for (int i = 0; i < read_rows * cols; ++i) file >> local_data[i];

    file.close();
    return true;
}

void print_and_save_timings(const std::string& name, const std::string& run_name,
                            int num_ranks, int num_threads, int N, int D,
                            double t_start, double t_io, double t_mean,
                            double t_cov, double t_eigen) {

    // Calculate durations
    double io_time = t_io - t_start;
    double preproc_time = t_mean - t_io;
    double cov_time = t_cov - t_mean;
    double eigen_time = t_eigen - t_cov;
    double total_time = t_eigen - t_start;

    // 1. Print to Console
    std::cout << "\n[" << name << " Timings]\n";
    std::cout << "I/O Time:       " << io_time << " s\n";
    std::cout << "Pre-processing: " << preproc_time << " s\n";
    std::cout << "Covariance:     " << cov_time << " s\n";
    std::cout << "Eigendecomp:    " << eigen_time << " s\n";
    std::cout << "Total Time:     " << total_time << " s\n\n";

    // 2. Save to CSV
    mkdir("output", 0777);
    std::string out_dir = "output/" + name;
    mkdir(out_dir.c_str(), 0777);
    std::string csv_filename = out_dir + "/results_" + run_name + ".csv";

    // Check if file exists to determine if we need to write the header
    std::ifstream file_check(csv_filename);
    bool file_exists = file_check.good();
    file_check.close();

    std::ofstream out(csv_filename, std::ios::app);
    if (!out.is_open()) {
        std::cerr << "Error: Could not open " << csv_filename << " for writing.\n";
        return;
    }

    if (!file_exists) {
        out << "RunName,Ranks,Threads,Rows,Cols,IO_s,Preprocess_s,Covariance_s,Eigen_s,Total_s\n";
    }

    out << run_name << ","
        << num_ranks << ","
        << num_threads << ","
        << N << ","
        << D << ","
        << io_time << ","
        << preproc_time << ","
        << cov_time << ","
        << eigen_time << ","
        << total_time << "\n";

    out.close();
}