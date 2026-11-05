# DSA5208 Project 3

**Author:** Jian Sheng

---

## Description & Project Structure

This project implements a hybrid MPI + OpenMP Principal Component Analysis (PCA) algorithm in C++. Benchmarks and scaling evaluations were conducted on the [NSCC ASPIRE 2A](https://www.nscc.sg/aspire-2a/) cluster.

The codebase provides three implementations for comparison:
- `pca_serial`: Single-threaded serial baseline
- `pca_naive`: Parallel baseline without memory access and synchronization optimizations
- `pca_optimized`: Parallel implementation with unit-stride memory access, false sharing elimination, and hoisted heap allocations

### Directory Layout

```text
dsa5208-project-3/
├── src/                # C++ source code (serial, naive, and optimized implementations)
├── python/             # Python scripts
│   ├── 01_load_data.py         # Download and prepare raw datasets
│   ├── 02_baseline_pca.py      # Scikit-learn reference eigenvalues & cluster plots
│   └── 03_visualize.py         # Empirical scaling and efficiency visualizations
├── jobs/               # PBS job submission scripts for cluster benchmarks
│   ├── 01_serial_baseline/     # Big-O complexity validation
│   ├── 02_naive_sweep/         # Naive parallel 64-core single-node topology sweep
│   ├── 03_optimized_sweep/     # Optimized hybrid 64-core single-node topology sweep
│   ├── 04_optimized_scaling/   # Strong and weak multi-node scaling jobs (up to 8 nodes / 512 cores)
│   ├── 05_gpu/                 # OpenMP GPU target offload benchmark
├── data/               # Raw and preprocessed datasets (YearPredictionMSD, DSADS)
├── executables/        # Compiled C++ binaries (pca_serial, pca_naive, pca_optimized, pca_gpu)
├── output/             # Local run outputs and visualization charts
├── hpc-output/         # Benchmark timing CSVs synced back from the cluster
├── hpc-logs/           # Raw PBS job output logs (.o<jobid>) from cluster runs
├── pyproject.toml      # Project configuration and uv dependencies
├── requirements.txt    # Pip dependencies for non-uv environments
├── CMakeLists.txt      # CMake build configuration
└── README.md           # Project documentation and reproduction guide
```

> **Notes:**
> - `data/`: Not included in the submission archive. Run `uv run python/01_load_data.py` to download and populate the datasets.
> - `executables/`: Pre-compiled binaries were built on an Apple Silicon M4 MacBook Air (`macOS / arm64`). Recompile using CMake if executing in another environment or architecture.

---

## Setup & Execution Workflow

### 1. Python Environment Setup

Install project dependencies using either `uv` or Python `venv`:

* **Option A (Recommended): Using `uv`**
  ```bash
  uv sync
  ```

* **Option B: Using Python `venv` & `requirements.txt`**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```

---

### 2. Load Data Locally & Run Baseline PCA

Download and preprocess the raw datasets (YearPredictionMSD and Daily and Sports Activities - DSADS):

```bash
uv run python/01_load_data.py
```

*(Optional)* Run the Python scikit-learn reference implementation to generate baseline eigenvalues and explained variance ratios for numerical correctness validation:

```bash
uv run python/02_baseline_pca.py
```

---

### 3. (Optional) Compile and Run Locally

#### Compile Locally

```bash
rm -rf build executables && mkdir build && cd build && cmake .. && cmake --build . && cd ..
```

#### Run Locally (Examples)

Command-line parameters:
- `<data_path>`: Path to the input feature text file (space-separated, rows × cols)
- `<top_k>`: Number of principal components to extract (e.g., `5`)
- `<scale>`: `0` for mean centering only; `1` for full z-score standardization
- `<run_name>`: Identifier string used for timing output file naming
- `<num_ranks>`: Total number of MPI ranks to spawn
- `<ranks_per_node>`: Number of MPI ranks per node (equal to `<num_ranks>` for single-node / local execution)
- `<threads_per_rank>`: Number of OpenMP threads per MPI rank (set via `PE=<threads_per_rank>`)
- `--map-by ppr:<ranks_per_node>:node:PE=<threads_per_rank>`: Maps MPI processes per node and allocates dedicated processing elements (cores) per rank
- `--bind-to core`: Binds each rank and its OpenMP threads to physical cores to avoid thread migration and core starvation

* **Serial Execution:**
  ```bash
  # Syntax: ./executables/pca_serial <data_path> <top_k> <scale> <run_name>
  ./executables/pca_serial data/year-prediction-msd/msd_full.txt 5 1 msd_serial
  ```

* **Parallel Execution (Hybrid MPI + OpenMP):**
  ```bash
  # Syntax: export OMP_NUM_THREADS=<threads_per_rank> && mpiexec -n <num_ranks> --map-by ppr:<ranks_per_node>:node:PE=<threads_per_rank> --bind-to core ./executables/pca_optimized <data_path> <top_k> <scale> <run_name>
  export OMP_NUM_THREADS=4 && mpiexec -n 4 --map-by ppr:4:node:PE=4 --bind-to core ./executables/pca_optimized data/year-prediction-msd/msd_full.txt 5 1 msd_parallel
  ```

---

### 4. Sync to NSCC

Sync source code, job scripts, and data to the NSCC cluster scratch directory. Replace `<REMOTE_HOST>` with your SSH configuration alias or login target (e.g., `<username>@aspire2a.nus.edu.sg` or an SSH config entry like `nscc`):

```bash
rsync -avzPR \
  --exclude="*.zip" \
  --exclude="data/dsads/data" \
  src jobs CMakeLists.txt data/dsads data/year-prediction-msd \
  <REMOTE_HOST>:~/scratch/dsa5208-project-3/
```

---

### 5. Compile on NSCC

SSH into the cluster, load compiler and MPI modules, and build the executables:

```bash
ssh <REMOTE_HOST>
cd ~/scratch/dsa5208-project-3
module load nvhpc/23.7 openmpi/4.1.2-hpe
rm -rf build executables && mkdir build && cd build && cmake .. && cmake --build . && cd ..
```

---

### 6. Submit PBS Jobs (Cluster Benchmarks)

Submit benchmark suites to the PBS queue using `qsub`:

#### 1. Serial Baseline (O(N) & O(1) Complexity Validation)

```bash
qsub jobs/01_serial_baseline/01_dsads_full_serial.pbs
qsub jobs/01_serial_baseline/02_dsads_serial.pbs
qsub jobs/01_serial_baseline/03_msd_serial.pbs
```

#### 2. Naive Parallel Single-Node Sweep (False Sharing & Strided Baseline)

```bash
qsub jobs/02_naive_sweep/01_dsads_full_1n.pbs
qsub jobs/02_naive_sweep/02_msd_1n.pbs
```

#### 3. Optimized Parallel Single-Node Sweep (Unit-Stride & False Sharing Removed)

```bash
qsub jobs/03_optimized_sweep/01_dsads_full_1n.pbs
qsub jobs/03_optimized_sweep/02_msd_1n.pbs
```

#### 4. Optimized Multi-Node Scaling (Strong & Weak Scaling up to 8 Nodes / 512 Cores)

* **DSADS Strong Scaling (Optimal: 1 Rank x 64 Threads per node):**
  ```bash
  qsub jobs/04_optimized_scaling/01_dsads_strong_1n.pbs
  qsub jobs/04_optimized_scaling/02_dsads_strong_2n.pbs
  qsub jobs/04_optimized_scaling/03_dsads_strong_4n.pbs
  qsub jobs/04_optimized_scaling/04_dsads_strong_8n.pbs
  ```

* **MSD Strong Scaling (Optimal: 32 Ranks x 2 Threads per node):**
  ```bash
  qsub jobs/04_optimized_scaling/05_msd_strong_1n.pbs
  qsub jobs/04_optimized_scaling/06_msd_strong_2n.pbs
  qsub jobs/04_optimized_scaling/07_msd_strong_4n.pbs
  qsub jobs/04_optimized_scaling/08_msd_strong_8n.pbs
  ```

* **DSADS Weak Scaling (Optimal: 1 Rank x 64 Threads per node; 1k, 2k, 4k, 8k rows):**
  ```bash
  qsub jobs/04_optimized_scaling/09_dsads_weak_1n.pbs
  qsub jobs/04_optimized_scaling/10_dsads_weak_2n.pbs
  qsub jobs/04_optimized_scaling/11_dsads_weak_4n.pbs
  qsub jobs/04_optimized_scaling/12_dsads_weak_8n.pbs
  ```

#### 5. OpenMP GPU Target Offloading (NVIDIA A100 Accelerator Queue)

```bash
qsub jobs/05_gpu/01_gpu_run.pbs
```

---

### 7. Sync Output Back to Local & Visualize

Sync benchmark timing CSVs and raw job output logs back to your local machine:

```bash
# Sync timing results
rsync -avzP <REMOTE_HOST>:~/scratch/dsa5208-project-3/output/ ./hpc-output/

# Sync raw PBS job execution logs
rsync -avzP '<REMOTE_HOST>:~/scratch/dsa5208-project-3/*.o*' ./hpc-logs/
```

Generate visualization plots comparing naive vs. optimized performance, Amdahl's Law strong scaling, Gustafson's Law weak scaling, parallel kernel efficiency, and hardware accelerator comparisons (Serial vs. CPU vs. GPU):

```bash
uv run python/03_visualize.py
```
Generated plots will be saved in `output/visualization/`.
