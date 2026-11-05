#ifndef PCA_CORE_HPP
#define PCA_CORE_HPP

#include <vector>
#include <string>

struct CmdArgs {
    std::string data_path;
    int top_k;
    bool scale;
    std::string name;
};

void print_usage(const std::string& program_name);
bool parse_args(int argc, char** argv, CmdArgs& args, bool print = true);
void normalize(std::vector<double>& vec);
bool get_dataset_dimensions(const std::string& filepath, int& rows, int& cols);
bool load_data_chunk(const std::string& filepath, std::vector<double>& local_data, int skip_rows, int read_rows, int cols);
void print_and_save_timings(const std::string& name, const std::string& run_name,
                            int num_ranks, int num_threads, int N, int D,
                            double t_start, double t_io, double t_mean,
                            double t_cov, double t_eigen);

#endif