import os
import shutil
import sys

import kagglehub


def ingest_kaggle_data():
    print("Downloading historical football data from Kaggle...")
    cache_path = kagglehub.dataset_download(
        "martj42/international-football-results-from-1872-to-2017"
    )
    raw_dir = os.path.join("data", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    target_files = ["results.csv", "shootouts.csv", "former_names.csv"]
    for file in target_files:
        src_file = os.path.join(cache_path, file)
        dst_file = os.path.join(raw_dir, file)
        if os.path.exists(src_file):
            _ = shutil.copy2(src_file, dst_file)
            print(f"Successfully moved {file} to {raw_dir}/")

    print("Raw Kaggle data ingested successfully.")


def verify_data_layer():
    """Validates presence of DataCamp blueprints and manages automated Kaggle recovery."""
    datacamp_files = [
        os.path.join("data", "raw", "group_fixtures.csv"),
        os.path.join("data", "raw", "knockout_slots.csv"),
    ]

    kaggle_files = [
        os.path.join("data", "raw", "results.csv"),
        os.path.join("data", "raw", "shootouts.csv"),
        os.path.join("data", "raw", "former_names.csv"),
    ]

    # Check DataCamp Blueprints (Hard Gate)
    missing_datacamp = [f for f in datacamp_files if not os.path.exists(f)]
    if missing_datacamp:
        print("\n🛑 CRITICAL COMPONENT MISMATCH: COMPETITION BLUEPRINTS MISSING")
        print("=" * 64)
        print(
            "The engine cannot map out the tournament framework without your tournament slots."
        )
        print(
            "Please manually download the following files from your DataCamp workspace:"
        )
        for file in missing_datacamp:
            print(f" ❌ - {os.path.basename(file)}")
        print("\n📝 ACTION MEMO:")
        print(" 1. Log into your DataCamp DataLab World Cup competition environment.")
        print(
            " 2. Open the left sidebar, click the 'Files' folder icon, and expand 'data/'."
        )
        print(
            " 3. Right-click each missing file, select 'Download', and save them locally."
        )
        print(" 4. Drop them inside your native local path: data/raw/")
        print("=" * 64)
        sys.exit(1)

    # Check Kaggle History (Automated Recovery)
    missing_kaggle = [f for f in kaggle_files if not os.path.exists(f)]
    if missing_kaggle:
        print("\n⚠️ Missing local historical files. Initializing automated scraper...")
        ingest_kaggle_data()
    else:
        print("📥 Full raw Kaggle data layer detected locally. Skipping download.")
