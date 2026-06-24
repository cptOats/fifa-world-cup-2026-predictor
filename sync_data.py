"""
Independent Data Ingestion and Transformation Utility.

Updates the raw Kaggle logs and compiles the live tournament 'as-played'
overrides without triggering heavy training loops or stochastic simulations.
"""

import logging
import os
import shutil
import sys

from main import DATACAMP_TO_KAGGLE, TRAINING_VARIABLES, verify_data_layer
from src.transform import patch_tournament_structures, prepare_historical_features

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure console logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def run_isolated_data_sync():
    """Executes the data ETL and verification sequence end-to-end."""
    logging.info("🚀 Starting Isolated Data Sync...")

    try:
        # 1. Clean out processed staging environments
        processed_dir = os.path.join("data", "processed")
        if os.path.exists(processed_dir):
            shutil.rmtree(processed_dir)
            logging.info(f"🗑️ Removed processed directory: {processed_dir}")

        # 2. Evict stale Kaggle international football match dataset
        raw_dir = os.path.join("data", "raw")
        target_raw_evictions = ["former_names.csv", "results.csv", "shootouts.csv"]

        for file_name in target_raw_evictions:
            file_path = os.path.join(raw_dir, file_name)
            if os.path.exists(file_path):
                os.remove(file_path)
                logging.info(
                    f"♻️ Evicted raw historical asset to force fresh Kaggle pull: {file_path}"
                )

        # 3. Assert data layer integrity before closing the loop 🛡️
        logging.info("🛡️ Running data layer validation suite...")
        verify_data_layer()

        # 4. Patch structures and auto-ingest real scores from results.csv
        logging.info("⚙️ Parsing results.csv and embedding actual score lines...")
        patch_tournament_structures(TRAINING_VARIABLES)

        # 5. Re-build the Parquet feature engine for the new date window
        logging.info("🏗️ Re-compiling parquet historical learning features...")
        prepare_historical_features(DATACAMP_TO_KAGGLE, TRAINING_VARIABLES)

        logging.info("✅ Data sync and verification complete!")

    except Exception as e:
        logging.critical(
            f"❌ Data Sync or Verification failed: {str(e)}", exc_info=True
        )
        sys.exit(1)


if __name__ == "__main__":
    run_isolated_data_sync()
