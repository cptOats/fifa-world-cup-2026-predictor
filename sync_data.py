"""
Independent Data Ingestion and Transformation Utility.

Updates the raw Kaggle logs and compiles the live tournament 'as-played'
overrides without triggering heavy training loops or stochastic simulations.
"""

import logging
import os
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
    logging.info("🚀 Starting Isolated Mid-Tournament Data Sync...")

    try:
        # 1. Assert data layer integrity before closing the loop 🛡️
        logging.info("🛡️  Running data layer validation suite...")
        verify_data_layer()

        # 2. Patch structures and auto-ingest real scores from results.csv
        logging.info("⚙️  Parsing results.csv and embedding actual score lines...")
        patch_tournament_structures(TRAINING_VARIABLES)

        # 3. Re-build the Parquet feature engine for the new date window
        logging.info("🏗️  Re-compiling parquet historical learning features...")
        prepare_historical_features(DATACAMP_TO_KAGGLE, TRAINING_VARIABLES)

        logging.info("✅ Data sync and verification complete!")

    except Exception as e:
        logging.critical(
            f"❌ Data Sync or Verification failed: {str(e)}", exc_info=True
        )
        sys.exit(1)


if __name__ == "__main__":
    run_isolated_data_sync()
