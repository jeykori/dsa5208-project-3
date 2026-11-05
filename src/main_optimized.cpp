#include "pca_core.hpp"
#include "pca_parallel_optimized.hpp"
#include <iostream>
#include <vector>
#include <mpi.h>
#include <omp.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int num_threads = omp_get_max_threads();
    int num_devices = omp_get_num_devices();

    if (world_rank == 0) {
        if (num_devices > 0) {
            std::cout << "[INFO] OpenMP Target Offload: " << num_devices << " GPU device(s) detected.\n";
        } else {
            std::cout << "[INFO] OpenMP Target Offload: No GPU detected; executing on CPU host threads.\n";
        }
    }

    CmdArgs args;
    if (!parse_args(argc, argv, args, world_rank == 0)) {
        if (world_rank == 0) print_usage("mpiexec ./pca_optimized");
        MPI_Finalize();
        return 1;
    }

    int rows = 0, cols = 0;

    // Rank 0 calculates dimensions, broadcasts to all
    if (world_rank == 0) {
        if(!get_dataset_dimensions(args.data_path, rows, cols)) {
            std::cerr << "Failed to parse dimensions from " << args.data_path << "\n";
        }
    }
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rows == 0 || cols == 0) {
        MPI_Finalize();
        return 1;
    }

    double t_start, t_io, t_mean, t_cov, t_eigen;
    MPI_Barrier(MPI_COMM_WORLD);
    t_start = MPI_Wtime();

    int base_rows = rows / world_size;
    int local_rows = base_rows + (world_rank == world_size - 1 ? rows % world_size : 0);
    int skip_rows = world_rank * base_rows;

    std::vector<double> local_data(local_rows * cols, 0.0);
    load_data_chunk(args.data_path, local_data, skip_rows, local_rows, cols);

    MPI_Barrier(MPI_COMM_WORLD);
    t_io = MPI_Wtime();

    standardize_parallel(local_data, local_rows, rows, cols, args.scale);
    MPI_Barrier(MPI_COMM_WORLD);
    t_mean = MPI_Wtime();

    std::vector<double> global_cov = compute_covariance_parallel(local_data, local_rows, rows, cols);
    MPI_Barrier(MPI_COMM_WORLD);
    t_cov = MPI_Wtime();

    compute_eigenvectors_parallel(global_cov, rows, cols, args.top_k, world_rank);
    MPI_Barrier(MPI_COMM_WORLD);
    t_eigen = MPI_Wtime();

    if (world_rank == 0) {
        print_and_save_timings("optimized", args.name, world_size, num_threads, rows, cols,
                       t_start, t_io, t_mean, t_cov, t_eigen);
    }

    MPI_Finalize();
    return 0;
}

