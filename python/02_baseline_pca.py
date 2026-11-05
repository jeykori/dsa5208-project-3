import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VALIDATION_DIR = os.path.join(BASE_DIR, "output", "validation")
VISUALIZATION_DIR = os.path.join(BASE_DIR, "output", "visualization")

def generate_baseline(dataset_name, features_file, labels_file, scale_std, label_mapper=None):
    print(f"\n{'='*40}\nGenerating Baseline for {dataset_name.upper()}\n{'='*40}")

    # Map dataset path, supporting both 'msd' and 'year-prediction-msd' directory names
    if dataset_name == "msd" and not os.path.exists(os.path.join(BASE_DIR, "data", "msd")):
        data_dir = os.path.join(BASE_DIR, "data", "year-prediction-msd")
    else:
        data_dir = os.path.join(BASE_DIR, "data", dataset_name)

    os.makedirs(VALIDATION_DIR, exist_ok=True)
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)

    print(f"Loading features and labels from {data_dir}...")
    X = pd.read_csv(os.path.join(data_dir, features_file), sep=r"\s+", header=None)
    y = pd.read_csv(os.path.join(data_dir, labels_file), header=None)

    if label_mapper:
        y_labels = y[0].map(label_mapper)
    else:
        y_labels = y[0]

    print(f"Preprocessing data (Scale Variance: {scale_std})...")
    scaler = StandardScaler(with_std=scale_std)
    X_scaled = scaler.fit_transform(X)

    print("Computing PCA via Scikit-Learn...")
    pca = PCA()
    X_projected = pca.fit_transform(X_scaled)

    # 1. Export Validation Targets
    report_lines = [
        f"--- Scikit-Learn Validation Targets ({dataset_name.upper()}) ---",
        f"Scaling applied: {'Z-Score Standardization (with std)' if scale_std else 'Mean Centering Only'}",
        "\nTop 5 Eigenvalues (Variance):"
    ]
    for i, val in enumerate(pca.explained_variance_[:5]):
        report_lines.append(f"PC{i+1}: {val:.6f}")

    report_lines.append("\nTop 5 Explained Variance Ratios:")
    for i, val in enumerate(pca.explained_variance_ratio_[:5]):
        report_lines.append(f"PC{i+1}: {val:.6%}")

    report_lines.append(f"\nTotal Variance Explained by Top 5: {pca.explained_variance_ratio_[:5].sum():.6%}")

    report_text = "\n".join(report_lines)
    print(f"\n{report_text}\n")

    targets_path = os.path.join(VALIDATION_DIR, f"{dataset_name}_targets.txt")
    with open(targets_path, "w") as f:
        f.write(report_text)
    print(f"Validation targets saved to: {targets_path}")

    # 2. Export Cluster Plot
    print("Generating plot for Top 2 Components...")
    df = pd.DataFrame(data=X_projected[:, :2], columns=['PC1', 'PC2'])
    df['Label'] = y_labels

    # Downsample for plotting if dataset is massive to prevent crashing Matplotlib
    if len(df) > 50000:
        print("Dataset is large, sampling 50,000 points for the scatter plot...")
        df = df.sample(50000, random_state=42)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='PC1', y='PC2', hue='Label', palette='tab10', data=df, s=15, alpha=0.5)

    plt.title(f'PCA of {dataset_name.upper()} (Top 2 Components)')
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} Variance)')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} Variance)')

    plt.legend(title='True Label', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    plot_path = os.path.join(VISUALIZATION_DIR, f"{dataset_name}_pca_clusters.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Cluster plot saved to: {plot_path}")

if __name__ == "__main__":
    # 1. YearPredictionMSD (Full Standardization)
    generate_baseline(
        dataset_name="msd",
        features_file="msd_full.txt",
        labels_file="msd_labels.txt",
        scale_std=True,
        label_mapper=lambda y: f"{(int(y) // 10) * 10}s"
    )

    # 2. DSADS (Full Standardization) - 19 Activities
    generate_baseline(
        dataset_name="dsads",
        features_file="dsads_full.txt",
        labels_file="dsads_labels.txt",
        scale_std=True,
        label_mapper=lambda y: f"Activity {int(y)}"
    )

    print("\n" + "="*70)
    print("VERIFICATION REMINDER:")
    print("Please manually cross-check these 'explained_variance_' values against the")
    print("C++ stdout eigenvalues (from pca_serial or pca_optimized) to verify")
    print("numerical correctness for your final report.")
    print(f"Validation targets saved in: {VALIDATION_DIR}/")
    print(f"Cluster plots saved in:     {VISUALIZATION_DIR}/")
    print("="*70 + "\n")
