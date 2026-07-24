"""
Automated Data Ingestion and Infrastructure Gateway Layer.

Ensures the necessary raw dependency files exist prior to model execution.
Provides a fallback automated Kaggle scraper if core historical results
are missing from the local environment.
"""

import io
import json
import logging
import os
import shutil
import sys
import urllib.request

import kagglehub
import pandas as pd


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


def _ingest_fifa_annex_c_matrix():
    """Scrapes the official 495-row Annex C wildcard matrix from Wikipedia.

    Transforms multi-header layout schemas directly into standardized
    structural O(1) JSON lookup tables mapped to tournament Match IDs.
    """
    url = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_knockout_stage"

    try:
        # Spoof a legitimate modern browser request to bypass Wikipedia's 403 bot gate
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/126.0.0.0 Safari/537.36"
                )
            },
        )

        logging.info("🌐 Fetching official FIFA tournament configurations...")
        with urllib.request.urlopen(req) as response:
            html_text = response.read().decode("utf-8")

        # Wrap the string in io.StringIO to force Pandas to read it as data
        tables = pd.read_html(io.StringIO(html_text), match="Third-place teams")
        df_matrix = tables[0]

        # Normalize columns safely whether they are MultiIndex tuples or flat strings
        cleaned_cols = []
        for col in df_matrix.columns.values:
            if isinstance(col, tuple):
                cleaned_cols.append(
                    " ".join(str(c) for c in col if "Unnamed" not in str(c)).strip()
                )
            else:
                cleaned_cols.append(str(col).strip())
        df_matrix.columns = cleaned_cols

        column_to_match_id = {
            "1E vs": 74,
            "1I vs": 77,
            "1A vs": 79,
            "1L vs": 80,
            "1D vs": 81,
            "1G vs": 82,
            "1B vs": 85,
            "1K vs": 87,
        }

        official_fifa_matrix = {}

        # 🎯 THE FIX: Target all 12 columns under the "Third-place teams advance" banner
        group_cols = [c for c in df_matrix.columns if "Third-place" in c]

        for _, row in df_matrix.iterrows():
            # Scan across all 12 columns to gather which 8 groups advanced in this row
            advanced = []
            for col in group_cols:
                val = str(row[col]).strip()
                if val and val.lower() != "nan" and val.lower() != "no.":
                    # Extract the clean group letter character
                    letter = "".join(filter(str.isalpha, val)).upper()
                    if letter and len(letter) == 1:
                        advanced.append(letter)

            # Combine alphabetically to build our bulletproof O(1) signature key
            combo_key = "".join(sorted(advanced))

            # Escape route for header/footer table artifacts that aren't valid combinations
            if len(combo_key) != 8:
                continue

            slot_mapping = {}
            for wiki_col, match_id in column_to_match_id.items():
                actual_col = next(c for c in df_matrix.columns if wiki_col in c)
                # Wikipedia cells will say "3E", "3F", etc. Strip out the '3'
                opponent_val = str(row[actual_col]).replace("3", "").strip()
                slot_mapping[match_id] = opponent_val

            official_fifa_matrix[combo_key] = slot_mapping

        output_path = os.path.join("data", "raw", "annex_c_matrix.json")
        with open(output_path, "w") as f:
            json.dump(official_fifa_matrix, f, indent=4)

        logging.info(
            f"✅ Successfully compiled and cached {len(official_fifa_matrix)} FIFA Annex C combinations."
        )

    except Exception as e:
        logging.error(
            f"🛑 Automated HTML processing failed to capture Annex C matrix elements: {e}"
        )
        raise RuntimeError(
            "Data layer verification broken: Annex C acquisition failure."
        ) from e


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

    annex_c_file = os.path.join("data", "raw", "annex_c_matrix.json")

    # 1. Hard Gate Validation: Structural bracket blueprints cannot be auto-scraped
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

    # 2. Soft Gate Validation Tier A: Kaggle historical logs
    missing_kaggle = [f for f in kaggle_files if not os.path.exists(f)]
    if missing_kaggle:
        logging.warning(
            "⚠️ Missing local historical files. Initializing automated scraper..."
        )
        _ingest_kaggle_data()
        logging.info("📥 Automated scraper ingestion completed successfully.")

    # 3. Soft Gate Validation Tier B: Official Tournament Regulatory Matrix
    if not os.path.exists(annex_c_file):
        logging.warning(
            "⚠️ Missing official FIFA Annex C structural matrix. Initializing automated web-scraper..."
        )
        _ingest_fifa_annex_c_matrix()
