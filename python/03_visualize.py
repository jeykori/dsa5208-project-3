import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Create output visualization directory
OUTPUT_DIR = "output/visualization"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set global plotting style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

print("=== Starting PCA Benchmark Visualization (Naive vs Optimized) ===")

# ==========================================
# 1. Algorithmic Complexity (Serial Baseline)
# ==========================================
print("Generating Plot 1: Algorithmic Complexity...")
serial_files = glob.glob("hpc-output/serial/results_dsads_serial_*.csv") + glob.glob("hpc-output/serial/results_dsads_full_serial.csv")

if serial_files:
    df_serial = pd.concat([pd.read_csv(f) for f in serial_files], ignore_index=True)
    numeric_cols = ['Rows', 'Cols', 'Covariance_s', 'Eigen_s', 'Total_s']
    for col in numeric_cols:
        if col in df_serial.columns:
            df_serial[col] = pd.to_numeric(df_serial[col], errors='coerce')

    df_serial_agg = df_serial.groupby(['Rows', 'Cols'], as_index=False).median(numeric_only=True).sort_values('Rows')

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(df_serial_agg['Rows'], df_serial_agg['Covariance_s'], marker='o', linewidth=2, label='Covariance Step O(N)')
    ax.plot(df_serial_agg['Rows'], df_serial_agg['Eigen_s'], marker='s', linewidth=2, label='Eigendecomposition O(1)')
    ax.plot(df_serial_agg['Rows'], df_serial_agg['Total_s'], marker='^', linewidth=2, linestyle='--', label='Total Execution Time')

    ax.set_title('DSADS Serial Algorithmic Complexity (Big-O Validation)', fontsize=13, pad=12, fontweight='bold')
    ax.set_xlabel('Dataset Size (Rows, N)', fontsize=11)
    ax.set_ylabel('Execution Time (Seconds)', fontsize=11)
    ax.legend(frameon=True, facecolor='white')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '01_serial_complexity.png'), dpi=300)
    plt.close()
else:
    print("Warning: No serial CSV files found in hpc-output/serial/. Skipping Plot 1.")


# ==========================================
# 2. The Memory Hierarchy Fix (Naive vs Optimized Topology)
# ==========================================
def plot_topology_comparison(dataset_prefix, dataset_name):
    naive_path = f"hpc-output/naive/final_{dataset_prefix}_1node_sweep.csv"
    opt_path = f"hpc-output/optimized/final_{dataset_prefix}_1node_sweep.csv"

    # Fallback to general names if specific ones aren't found
    if not os.path.exists(naive_path): naive_path = "hpc-output/naive/final_1node_sweep.csv"
    if not os.path.exists(opt_path): opt_path = "hpc-output/optimized/final_1node_sweep.csv"

    if not (os.path.exists(naive_path) and os.path.exists(opt_path)):
        print(f"Warning: Missing naive or optimized topology CSVs for {dataset_name}. Skipping Plot 2.")
        return

    df_naive = pd.read_csv(naive_path)
    df_opt = pd.read_csv(opt_path)

    df_naive['Config'] = df_naive['Ranks'].astype(str) + "R x " + df_naive['Threads'].astype(str) + "T"
    df_opt['Config'] = df_opt['Ranks'].astype(str) + "R x " + df_opt['Threads'].astype(str) + "T"

    df_naive = df_naive.sort_values(by='Ranks')
    df_opt = df_opt.sort_values(by='Ranks')

    configs = df_naive['Config'].tolist()
    x = np.arange(len(configs))
    width = 0.35

    # Plot 2 Subplots: Covariance comparison and Total Time comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Subplot 1: Covariance Compute (The Memory Fix)
    ax1.bar(x - width/2, df_naive['Covariance_s'], width, label='Naive (Strided + False Sharing)', color='#e74c3c', edgecolor='black')
    ax1.bar(x + width/2, df_opt['Covariance_s'], width, label='Optimized (Unit-Stride)', color='#27ae60', edgecolor='black')
    ax1.set_title(f'Covariance Compute Time ({dataset_name})', fontweight='bold')
    ax1.set_ylabel('Execution Time (Seconds)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, rotation=15)
    ax1.legend(frameon=True)

    # Subplot 2: Total End-to-End Time
    ax2.bar(x - width/2, df_naive['Total_s'], width, label='Naive Total', color='#95a5a6', edgecolor='black')
    ax2.bar(x + width/2, df_opt['Total_s'], width, label='Optimized Total', color='#2c3e50', edgecolor='black')
    ax2.set_title(f'Total End-to-End Time ({dataset_name})', fontweight='bold')
    ax2.set_ylabel('Execution Time (Seconds)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, rotation=15)
    ax2.legend(frameon=True)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'02_{dataset_prefix}_topology_comparison.png'), dpi=300)
    plt.close()

print("Generating Plot 2: Topology Comparisons...")
plot_topology_comparison("dsads", "DSADS Full")
plot_topology_comparison("msd", "MSD Full")


# ==========================================
# 3. Decoupled Strong Scaling & Amdahl's Law
# ==========================================
def plot_decoupled_strong_scaling(dataset_prefix, dataset_name):
    files = {n: f"hpc-output/optimized/results_{dataset_prefix}_strong_{n}node.csv" for n in [1, 2, 4, 8]}

    if not all(os.path.exists(p) for p in files.values()):
        print(f"Warning: Missing optimized strong scaling CSVs for {dataset_name}. Skipping Plot 3.")
        return

    nodes = [1, 2, 4, 8]
    cov_times, total_times = [], []
    for n in nodes:
        df = pd.read_csv(files[n])
        cov_times.append(df['Covariance_s'].median())
        total_times.append(df['Total_s'].median())

    cov_speedups = [cov_times[0] / t for t in cov_times]
    tot_speedups = [total_times[0] / t for t in total_times]
    cov_eff = [s / n for s, n in zip(cov_speedups, nodes)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Speedup Comparison (Log-Log Scale to visualize Amdahl's Law)
    ax1.plot(nodes, cov_speedups, marker='o', linewidth=2.5, markersize=8, color='#2980b9', label='Covariance Kernel (Compute)')
    ax1.plot(nodes, tot_speedups, marker='^', linewidth=2.5, markersize=8, color='#e74c3c', linestyle='--', label='End-to-End (Total with I/O & Eigen)')
    ax1.plot(nodes, nodes, linestyle=':', color='#7f8c8d', linewidth=1.5, label='Ideal Linear ($y = x$)')

    # Logarithmic scaling for Amdahl's Law
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log', base=2)
    ax1.set_xticks(nodes)
    ax1.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax1.set_yticks([0.5, 1, 2, 4, 8])
    ax1.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
    ax1.minorticks_off()
    ax1.set_ylim(bottom=0.35, top=12)
    ax1.set_xlim(left=0.8, right=10)

    ax1.set_title(f'Strong Scaling Speedup ({dataset_name} - Log2 Scale)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Number of Nodes ($N$, Log2 Scale)', fontsize=11)
    ax1.set_ylabel('Speedup Factor ($T_1 / T_N$, Log2 Scale)', fontsize=11)
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend(frameon=True, facecolor='white', loc='upper left')

    # Covariance Efficiency
    ax2.plot(nodes, cov_eff, marker='s', linewidth=2.5, markersize=8, color='#27ae60', label='Covariance Parallel Efficiency')
    ax2.axhline(1.0, linestyle='--', color='#7f8c8d', linewidth=1.5, label='Ideal Efficiency (100%)')

    ax2.set_xscale('log', base=2)
    ax2.set_xticks(nodes)
    ax2.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax2.minorticks_off()

    ax2.set_title(f'Kernel Efficiency ({dataset_name})', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Number of Nodes ($N$, Log2 Scale)', fontsize=11)
    ax2.set_ylabel(r'Efficiency ($\eta = S_N / N$)', fontsize=11)
    ax2.set_ylim(0, 1.25)
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    ax2.legend(frameon=True, facecolor='white', loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'03_{dataset_prefix}_decoupled_strong_scaling.png'), dpi=300)
    plt.close()

print("Generating Plot 3: Decoupled Strong Scaling...")
plot_decoupled_strong_scaling("dsads", "DSADS Full")
plot_decoupled_strong_scaling("msd", "MSD Full")


# ==========================================
# 4. Weak Scaling (Gustafson's Law on Log Scale)
# ==========================================
def plot_weak_scaling():
    nodes = [n for n in [1, 2, 4, 8] if os.path.exists(f"hpc-output/optimized/results_dsads_weak_{n}node.csv")]

    if len(nodes) < 2:
        print("Warning: Insufficient optimized weak scaling CSVs for DSADS. Skipping Plot 4.")
        return

    io_times, prep_times, cov_times, eigen_times, total_times = [], [], [], [], []

    for n in nodes:
        df = pd.read_csv(f"hpc-output/optimized/results_dsads_weak_{n}node.csv")
        io_times.append(df['IO_s'].median())
        prep_times.append(df['Preprocess_s'].median())
        cov_times.append(df['Covariance_s'].median())
        eigen_times.append(df['Eigen_s'].median())
        total_times.append(df['Total_s'].median())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Subplot 1: Component Execution Times on Log Scale
    ax1.plot(nodes, total_times, marker='o', linewidth=2.5, markersize=8, color='#2c3e50', label='Total Time (End-to-End)')
    ax1.plot(nodes, cov_times, marker='s', linewidth=2.5, markersize=8, color='#e67e22', label=r'Covariance Step $O(N)$ (Parallel Kernel)')
    ax1.plot(nodes, eigen_times, marker='^', linewidth=2.5, markersize=8, color='#e74c3c', label='Eigendecomposition (Rank 0)')
    ax1.plot(nodes, io_times, marker='d', linewidth=2.5, markersize=8, color='#95a5a6', label='I/O Ingestion')
    ax1.axhline(cov_times[0], linestyle=':', color='#7f8c8d', linewidth=1.5, label=f'Ideal Gustafson Invariance ({cov_times[0]:.2f}s)')

    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log', base=2)
    ax1.set_xticks(nodes)
    ax1.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax1.set_yticks([0.5, 1, 2, 4, 8])
    ax1.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
    ax1.minorticks_off()
    ax1.set_ylim(bottom=0.35, top=10.0)

    ax1.set_title("Weak Scaling Execution Times (Gustafson's Law - DSADS)", fontsize=12, fontweight='bold')
    ax1.set_xlabel('Number of Nodes (Workload Scaled at 1k Rows/Node, Log2 Scale)', fontsize=11)
    ax1.set_ylabel('Execution Time (Seconds, Log Scale)', fontsize=11)
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend(frameon=True, facecolor='white', loc='best')

    # Subplot 2: Weak Scaling Efficiency
    cov_weak_eff = [cov_times[0] / t for t in cov_times]
    tot_weak_eff = [total_times[0] / t for t in total_times]

    ax2.plot(nodes, cov_weak_eff, marker='s', linewidth=2.5, markersize=8, color='#e67e22', label='Covariance Parallel Efficiency')
    ax2.plot(nodes, tot_weak_eff, marker='o', linewidth=2.5, markersize=8, color='#2c3e50', linestyle='--', label='Total End-to-End Efficiency')
    ax2.axhline(1.0, linestyle='--', color='#7f8c8d', linewidth=1.5, label='Ideal Scaled Efficiency (100%)')

    ax2.set_xscale('log', base=2)
    ax2.set_xticks(nodes)
    ax2.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax2.minorticks_off()

    ax2.set_title("Weak Scaling Parallel Efficiency", fontsize=12, fontweight='bold')
    ax2.set_xlabel('Number of Nodes (Log2 Scale)', fontsize=11)
    ax2.set_ylabel(r'Weak Efficiency ($\eta = T_1 / T_N$)', fontsize=11)
    ax2.set_ylim(0, 1.25)
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    ax2.legend(frameon=True, facecolor='white', loc='lower left')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '04_dsads_weak_scaling_optimized.png'), dpi=300)
    plt.close()

print("Generating Plot 4: Weak Scaling...")
plot_weak_scaling()


# ==========================================
# 5. Hardware Architecture Comparison (Serial vs CPU vs GPU)
# ==========================================
def plot_hardware_comparison():
    print("Generating Plot 5: Hardware Accelerator Comparison...")

    ds_serial_path = "hpc-output/serial/results_dsads_full_serial.csv"
    ds_cpu_path = "hpc-output/optimized/results_dsads_full_1n_1r_64t.csv"
    ds_gpu_path = "hpc-output/optimized/results_dsads_gpu_1n_1r_1t.csv"

    msd_serial_path = "hpc-output/serial/results_msd_serial.csv"
    msd_cpu_path = "hpc-output/optimized/results_msd_full_1n_32r_2t.csv"
    msd_gpu_path = "hpc-output/optimized/results_msd_gpu_1n_1r_1t.csv"

    required_files = [
        ds_serial_path, ds_cpu_path, ds_gpu_path,
        msd_serial_path, msd_cpu_path, msd_gpu_path
    ]
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        print(f"Warning: Missing required benchmark CSVs for hardware comparison: {missing}. Skipping Plot 5.")
        return

    ds_serial = pd.read_csv(ds_serial_path)['Total_s'].median()
    ds_cpu = pd.read_csv(ds_cpu_path)['Total_s'].median()
    ds_gpu = pd.read_csv(ds_gpu_path)['Total_s'].median()

    msd_serial = pd.read_csv(msd_serial_path)['Total_s'].median()
    msd_cpu = pd.read_csv(msd_cpu_path)['Total_s'].median()
    msd_gpu = pd.read_csv(msd_gpu_path)['Total_s'].median()

    # Create independent subplots side-by-side to accommodate vast dynamic range
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    arch_labels = [
        'Serial Baseline\n(1R x 1T)',
        'Optimal CPU\n(1R x 64T)',
        'OpenMP GPU Offload\n(1R x 1T + A100)'
    ]
    colors = ['#7f8c8d', '#2980b9', '#27ae60']

    # Subplot 1: DSADS (D = 5,625, N = 9,120)
    ds_vals = [ds_serial, ds_cpu, ds_gpu]
    ds_bars = ax1.bar(arch_labels, ds_vals, color=colors, edgecolor='black', width=0.55)
    ax1.set_title('DSADS Full Benchmark (5,625 Features, 9,120 Samples)\nCompute/Memory-Bound', fontsize=12, pad=10, fontweight='bold')
    ax1.set_ylabel('Execution Time (Seconds)', fontsize=11)
    ax1.set_ylim(0, max(ds_vals) * 1.18)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.5)

    for bar, val in zip(ds_bars, ds_vals):
        height = bar.get_height()
        speedup = ds_serial / val
        label_text = f"{val:.1f} s\n({speedup:.1f}x)" if speedup != 1.0 else f"{val:.1f} s\n(1.0x Baseline)"
        ax1.annotate(label_text,
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 4), textcoords="offset points",
                     ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Subplot 2: MSD (D = 90, N = 515,345)
    msd_labels = [
        'Serial Baseline\n(1R x 1T)',
        'Optimal CPU\n(32R x 2T)',
        'OpenMP GPU Offload\n(1R x 1T + A100)'
    ]
    msd_vals = [msd_serial, msd_cpu, msd_gpu]
    msd_bars = ax2.bar(msd_labels, msd_vals, color=colors, edgecolor='black', width=0.55)
    ax2.set_title('YearPredictionMSD Benchmark (90 Features, 515,345 Samples)\nThroughput/Sample-Bound', fontsize=12, pad=10, fontweight='bold')
    ax2.set_ylabel('Execution Time (Seconds)', fontsize=11)
    ax2.set_ylim(0, max(msd_vals) * 1.18)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.5)

    for bar, val in zip(msd_bars, msd_vals):
        height = bar.get_height()
        speedup = msd_serial / val
        label_text = f"{val:.2f} s\n({speedup:.1f}x)" if speedup != 1.0 else f"{val:.2f} s\n(1.0x Baseline)"
        ax2.annotate(label_text,
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 4), textcoords="offset points",
                     ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.suptitle("Hardware Architecture Performance Comparison (Total Execution Time)", fontsize=14, y=1.02, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '05_hardware_architecture_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

plot_hardware_comparison()

print(f"\n=== Success! All visualizations saved to '{OUTPUT_DIR}/' ===")