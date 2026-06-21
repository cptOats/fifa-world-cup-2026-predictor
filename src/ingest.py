"""
Automated Data Ingestion and Infrastructure Gateway Layer.

Ensures the necessary raw dependency files exist prior to model execution.
Provides a fallback automated Kaggle scraper if core historical results
are missing from the local environment.
"""

import logging
import os
import shutil
import sys

import kagglehub


def _ingest_kaggle_data():
    """Downloads the martj42 international football results dataset via KaggleHub API."""

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


def verify_data_layer():
    """
    Validates presence of tournament blueprints and handles recovery actions.
    Triggers a hard runtime exit if structural tournament maps are missing.
    """

    datacamp_files = [
        os.path.join("data", "raw", "group_fixtures.csv"),
        os.path.join("data", "raw", "knockout_slots.csv"),
    ]

    kaggle_files = [
        os.path.join("data", "raw", "results.csv"),
        os.path.join("data", "raw", "shootouts.csv"),
        os.path.join("data", "raw", "former_names.csv"),
    ]

    # 1. Hard Gate Validation: Structural bracket files cannot be auto-scraped
    missing_datacamp = [f for f in datacamp_files if not os.path.exists(f)]
    if missing_datacamp:
        logging.critical(
            "🛑 CRITICAL COMPONENT MISMATCH: TOURNAMENT BLUEPRINTS MISSING"
        )
        print(
            "The engine cannot map out the tournament framework without group fixtures and knockout slots."
        )
        print("Please manually download the following files from GitHub or DataCamp:")
        for file in missing_datacamp:
            print(f" ❌ - {os.path.basename(file)}")
        print("\n📝 Drop them inside your native local path: data/raw/")
        sys.exit(1)

    # 2. Soft Gate Validation: Kaggle history can be automatically recovered
    missing_kaggle = [f for f in kaggle_files if not os.path.exists(f)]
    if missing_kaggle:
        logging.warning(
            "⚠️  Missing local historical files. Initializing automated scraper..."
        )
        _ingest_kaggle_data()
        logging.info("📥 Automated scraper ingestion completed successfully.")
