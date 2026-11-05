cmake_minimum_required(VERSION 3.10)
project(PCA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/executables)
include_directories(include)

# 1. Serial Baseline
add_executable(pca_serial src/main_serial.cpp src/pca_core.cpp src/pca_serial.cpp)
target_compile_options(pca_serial PRIVATE -O3)

# --- APPLE SILICON OPENMP FIX ---
if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
    set(OpenMP_CXX_LIB_NAMES omp)
    set(OpenMP_omp_LIBRARY /opt/homebrew/opt/libomp/lib/libomp.dylib)
endif()
# --------------------------------

# 2. Parallel Implementations (Naive Baseline vs. Optimized Hybrid)
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# 2.1 Naive Parallel Implementation
add_executable(pca_naive src/main_naive.cpp src/pca_core.cpp src/pca_parallel_naive.cpp)
target_link_libraries(pca_naive PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)
target_compile_options(pca_naive PRIVATE -O3)

# 2.2 Optimized Hybrid Parallel Implementation
add_executable(pca_optimized src/main_optimized.cpp src/pca_core.cpp src/pca_parallel_optimized.cpp)
target_link_libraries(pca_optimized PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)
target_compile_options(pca_optimized PRIVATE -O3)

# 2.3 GPU-Targeted Parallel Implementation
add_executable(pca_gpu src/main_optimized.cpp src/pca_core.cpp src/pca_parallel_optimized.cpp)
target_link_libraries(pca_gpu PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)
target_compile_options(pca_gpu PRIVATE -O3)

# 3. OpenMP Target Offload Flags
option(ENABLE_GPU_OFFLOAD "Enable OpenMP GPU Target Offloading" ON)

if(ENABLE_GPU_OFFLOAD AND NOT APPLE)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC" OR CMAKE_CXX_COMPILER_ID STREQUAL "PGI")
        # Flags for NVIDIA HPC SDK (nvc++) targeting A100 architecture
        target_compile_options(pca_optimized PRIVATE -mp)
        target_link_options(pca_optimized PRIVATE -mp)

        target_compile_options(pca_gpu PRIVATE -mp=gpu -gpu=cc80)
        target_link_options(pca_gpu PRIVATE -mp=gpu -gpu=cc80)
    else()
        # Fallback for standard GCC (requires GCC to be built with offload support)
        target_compile_options(pca_gpu PRIVATE -fopenmp -foffload=nvptx-none)
        target_link_options(pca_gpu PRIVATE -fopenmp -foffload=nvptx-none)
    endif()
endif()