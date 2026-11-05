import os
import urllib.request
import zipfile
import glob
import re
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(BASE_DIR, "data")

def download_and_extract(url, target_dir, zip_name):
    os.makedirs(target_dir, exist_ok=True)
    zip_path = os.path.join(target_dir, zip_name)

    if not os.path.exists(zip_path):
        print(f"Downloading {zip_name}...")
        urllib.request.urlretrieve(url, zip_path)

    print(f"Extracting {zip_name}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)

def process_msd():
    print("\n--- Processing YearPredictionMSD Dataset ---")
    msd_dir = os.path.join(data_dir, "year-prediction-msd")
    download_and_extract(
        "https://archive.ics.uci.edu/static/public/203/yearpredictionmsd.zip",
        msd_dir, "msd.zip"
    )

    txt_path = os.path.join(msd_dir, "YearPredictionMSD.txt")
    print("Loading MSD CSV (515k rows x 91 columns)...")
    df = pd.read_csv(txt_path, header=None)

    y_full = df.iloc[:, 0]
    X_full = df.iloc[:, 1:]

    print("Exporting features and labels...")
    X_full.to_csv(os.path.join(msd_dir, "msd_full.txt"), sep=" ", index=False, header=False)
    y_full.to_csv(os.path.join(msd_dir, "msd_labels.txt"), sep=" ", index=False, header=False)
    print(f"MSD Ready! Shape: {X_full.shape}")

def process_dsads():
    print("\n--- Processing Daily and Sports Activities Dataset ---")
    dsads_dir = os.path.join(data_dir, "dsads")
    download_and_extract(
        "https://archive.ics.uci.edu/static/public/256/daily+and+sports+activities.zip",
        dsads_dir, "dsads.zip"
    )

    # Extract nested activity/subject zips (skipping top-level dsads.zip)
    for root, _, files in os.walk(dsads_dir):
        for file in files:
            if file.endswith('.zip') and file != "dsads.zip":
                inner_path = os.path.join(root, file)
                with zipfile.ZipFile(inner_path, 'r') as inner_ref:
                    inner_ref.extractall(root)

    txt_files = sorted(glob.glob(os.path.join(dsads_dir, "**", "a*", "p*", "*.txt"), recursive=True))
    if not txt_files:
        raise FileNotFoundError("Could not locate DSADS sensor text files.")

    print(f"Found {len(txt_files)} segment files. Flattening 125x45 windows into 5,625 features...")

    X_list, y_list = [], []
    for file_path in txt_files:
        match = re.search(r'[/\\]a(\d+)[/\\]', file_path)
        if not match:
            continue
        activity_id = int(match.group(1))

        # Standard CSV parsing without regex engine for performance
        df_segment = pd.read_csv(file_path, sep=",", header=None)
        X_list.append(df_segment.values.flatten())
        y_list.append(activity_id)

    X_full = pd.DataFrame(X_list)
    y_full = pd.Series(y_list)

    # 1. Generate Row-Scaled Subsets for Algorithmic Complexity Benchmark
    subsets = [1000, 2000, 4000, 8000]
    for n in subsets:
        if len(X_full) >= n:
            print(f"Exporting {n}-row subset...")
            X_full.iloc[:n].to_csv(os.path.join(dsads_dir, f"dsads_{n}.txt"), sep=" ", index=False, header=False)
            y_full.iloc[:n].to_csv(os.path.join(dsads_dir, f"dsads_{n}_labels.txt"), sep=" ", index=False, header=False)

    # 2. Generate Full Dataset for Strong Scaling Benchmark
    print("Exporting full dataset...")
    X_full.to_csv(os.path.join(dsads_dir, "dsads_full.txt"), sep=" ", index=False, header=False)
    y_full.to_csv(os.path.join(dsads_dir, "dsads_labels.txt"), sep=" ", index=False, header=False)
    print(f"DSADS Ready! Final shape: {X_full.shape}")

if __name__ == "__main__":
    process_msd()
    process_dsads()
