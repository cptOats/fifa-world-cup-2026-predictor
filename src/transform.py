"""Data Transformation, Entity Resolution, and Feature Engineering Layer.

Data cleaning to execute rigid string validation gates, neutralizing downstream entity mismatch errors.
Handles temporal slicing, and compiles optimized feature matrices for model consumption.
"""

import logging
import os

import numpy as np
import pandas as pd

# Transformation Dictionary
DATACAMP_TO_KAGGLE = {
    "USA": "United States",
    "Côte d'Ivoire": "Ivory Coast",
    "Cabo Verde": "Cape Verde",
    "UEFA Playoff A": "Bosnia and Herzegovina",
    "UEFA Playoff B": "Sweden",
    "UEFA Playoff C": "Turkey",
    "UEFA Playoff D": "Czech Republic",
    "FIFA Playoff 1": "DR Congo",
    "FIFA Playoff 2": "Iraq",
}


def _validate_entity_resolution(translation_dict):
    """Validates naming alignment between upcoming tournament fixtures and historical logs.

    Maps current tournament fixture team names using the provided translation dictionary
    and checks them against known historical team entities in the match ledger. Known regional
    playoff placeholder strings are safely bypassed. If an unmapped country variant is
    discovered, execution is halted with actionable error logging.

    Args:
        translation_dict (dict[str, str]): A dictionary mapping source string entities
            (e.g., DataCamp keys) to standardized target naming equivalents (e.g., Kaggle keys).

    Raises:
        LookupError: If unrecognized country name string variants remain unresolved
            after applying the translation mapping step.
    """
    raw_dir = os.path.join("data", "raw")
    fixtures_df = pd.read_csv(os.path.join(raw_dir, "group_fixtures.csv"))

    # Map country strings to their historical equivalents
    datacamp_teams = set(
        fixtures_df["home_team"].replace(translation_dict).unique()
    ).union(set(fixtures_df["away_team"].replace(translation_dict).unique()))

    kaggle_df = pd.read_csv(os.path.join(raw_dir, "results.csv"))
    kaggle_teams = set(kaggle_df["home_team"].unique()).union(
        set(kaggle_df["away_team"].unique())
    )

    # Track structural string discrepancies remaining after mapping
    mismatches = datacamp_teams - kaggle_teams

    # Filter out known placeholder tournament bracket slot keys
    ignored_placeholders = {
        "FIFA Playoff 1",
        "FIFA Playoff 2",
        "UEFA Playoff A",
        "UEFA Playoff B",
        "UEFA Playoff C",
        "UEFA Playoff D",
    }
    true_mismatches = mismatches - ignored_placeholders

    if true_mismatches:
        logging.critical(
            f"❌ CRITICAL ENTITY DISCREPANCY: Unmapped country string variants detected: {true_mismatches}"
        )
        logging.critical(
            "Please expand the 'DATACAMP_TO_KAGGLE' translation dictionary in src/transform.py to resolve these."
        )
        raise LookupError("Pipeline halted due to unresolved entity names.")


def prepare_historical_features(translation_dict):
    """Ingests raw match data, verifies structural formatting, and builds core feature metrics.

    Triggers a defensive entity verification gate before parsing matrix logic. Once validated,
    it filters records to isolate modern era games (post-2018 World Cup up to the 2026 tournament kickoff),
    applies a penalty-based match importance weight matrix, and evaluates localized true home-field
    advantages for multi-nation co-hosts.

    Args:
        translation_dict (dict[str, str]): A dictionary mapping source string entities
            to master historical naming conventions.

    Returns:
        str: The localized absolute file path where the completed Apache Parquet matrix checkpoint
            was written.
    """
    # Execute defensive string verification gate before handling any heavy matrix logic
    _validate_entity_resolution(translation_dict)

    raw_dir = os.path.join("data", "raw")
    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    results_df = pd.read_csv(os.path.join(raw_dir, "results.csv"))

    # Apply Modern Era Time Slice (Post 2018 World Cup)
    TIME_SLICE_START = "2018-08-01"
    START_OF_TOURNAMENT = "2026-06-11"
    modern_df = results_df[
        (results_df["date"] >= TIME_SLICE_START)
        & (results_df["date"] < START_OF_TOURNAMENT)
    ].copy()

    # Assign Dynamic Match Weights
    modern_df["match_weight"] = np.where(
        modern_df["tournament"] == "Friendly", 0.4, 1.0
    )

    # Core Feature Generation
    co_hosts = {"United States", "Mexico", "Canada"}
    modern_df["is_true_home"] = modern_df.apply(
        lambda row: row["home_team"] in co_hosts and row["country"] == row["home_team"],
        axis=1,
    )
    modern_df["total_goals"] = modern_df["home_score"] + modern_df["away_score"]

    # Export complete matrix to Parquet
    output_path = os.path.join(processed_dir, "clean_historical_matches.parquet")
    modern_df.to_parquet(output_path, index=False)

    return output_path


def get_venue_country(venue_string):
    """Parses stadium text parameters to map the physical hosting nation country.

    Inspects localized string tokens to correctly tag structural home-field environment
    advantages across the 2026 multi-host landscape (USA, Mexico, Canada).

    Args:
        venue_string (str): The descriptive stadium name string from the schedule ledger.

    Returns:
        str: Standardized string name representation of the host country
            ("Mexico", "Canada", or "United States").
    """
    venue_lower = venue_string.lower()
    if (
        "mexico" in venue_lower
        or "guadalajara" in venue_lower
        or "monterrey" in venue_lower
    ):
        return "Mexico"
    elif "toronto" in venue_lower or "vancouver" in venue_lower:
        return "Canada"
    else:
        return "United States"


if __name__ == "__main__":
    # Standalone developer manual cache rebuilding
    print(
        "🔄 Standalone execution triggered. Rebuilding historical match feature matrix..."
    )
    prepare_historical_features(DATACAMP_TO_KAGGLE)
    prepare_historical_features(DATACAMP_TO_KAGGLE)
