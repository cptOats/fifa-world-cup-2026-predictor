"""Data Transformation, Entity Resolution, and Feature Engineering Layer."""

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


def load_historical_name_map(
    former_names_path: str = "former_names.csv",
) -> dict[str, str]:
    """Loads former_names.csv and builds a defensive, accent-normalized mapping dictionary."""

    if not os.path.exists(former_names_path):
        return {}

    df_names = pd.read_csv(former_names_path)
    name_map = {}

    for _, row in df_names.iterrows():
        current_name = str(row["current"]).strip()
        former_name = str(row["former"]).strip()

        # Map the primary name string
        name_map[former_name] = current_name

        # Symmetrical character normalization guardrails (e.g., Zaïre -> Zaire, Éire -> Eire)
        normalized_name = (
            former_name.replace("ï", "i")
            .replace("Zaïre", "Zaire")
            .replace("É", "E")
            .replace("é", "e")
        )
        if normalized_name != former_name:
            name_map[normalized_name] = current_name

    # Additional standard manual overrides for international datasets if necessary
    name_map["Zaire"] = "DR Congo"

    return name_map


def _validate_entity_resolution(translation_dict: dict[str, str]) -> None:
    """Validates naming alignment between upcoming tournament fixtures and historical logs."""

    raw_dir = os.path.join("data", "raw")
    fixtures_df = pd.read_csv(os.path.join(raw_dir, "group_fixtures.csv"))

    # Map country strings to their historical equivalents
    datacamp_teams = set(
        fixtures_df["home_team"].replace(translation_dict).unique()
    ).union(set(fixtures_df["away_team"].replace(translation_dict).unique()))

    kaggle_df = pd.read_csv(os.path.join(raw_dir, "results.csv"))

    name_map = load_historical_name_map("former_names.csv")
    if name_map:
        kaggle_df["home_team"] = kaggle_df["home_team"].map(
            lambda x: name_map.get(str(x).strip(), x)
        )
        kaggle_df["away_team"] = kaggle_df["away_team"].map(
            lambda x: name_map.get(str(x).strip(), x)
        )

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


def prepare_historical_features(
    translation_dict: dict[str, str], training_variables: dict[str, str | float]
) -> str:
    """Ingests raw match data, verifies structural formatting, and builds core feature metrics."""

    # Execute defensive string verification gate before handling any heavy matrix logic
    _validate_entity_resolution(translation_dict)

    raw_dir = os.path.join("data", "raw")
    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    results_df = pd.read_csv(os.path.join(raw_dir, "results.csv"))

    # Harmonize historical names BEFORE slicing or filtering features
    name_map = load_historical_name_map("former_names.csv")
    if name_map:
        results_df["home_team"] = results_df["home_team"].map(
            lambda x: name_map.get(str(x).strip(), x)
        )
        results_df["away_team"] = results_df["away_team"].map(
            lambda x: name_map.get(str(x).strip(), x)
        )
        if "country" in results_df.columns:
            results_df["country"] = results_df["country"].map(
                lambda x: name_map.get(str(x).strip(), x)
            )

    # Apply Modern Era Time Slice
    modern_df = results_df[
        (results_df["date"] >= training_variables["time_slice_start"])
        & (results_df["date"] < training_variables["start_of_tournament"])
    ].copy()

    # Assign Dynamic Match Weights

    modern_df["match_weight"] = np.where(
        modern_df["tournament"] == "Friendly",
        training_variables["friendly_weight"],
        1.0,
    )

    # Core Feature Generation
    co_hosts = {"United States", "Mexico", "Canada"}
    modern_df["is_true_home"] = modern_df.apply(
        lambda row: row["home_team"] in co_hosts and row["country"] == row["home_team"],
        axis=1,
    )

    # Guarantee that the native neutrality flag is explicitly clean and cast to integer types
    if "neutral" in modern_df.columns:
        modern_df["neutral"] = modern_df["neutral"].astype(int)
    else:
        modern_df["neutral"] = np.where(
            (modern_df["home_team"] == modern_df["country"])
            | (modern_df["away_team"] == modern_df["country"]),
            0,
            1,
        )

    modern_df["total_goals"] = modern_df["home_score"] + modern_df["away_score"]

    # Export complete matrix to Parquet
    output_path = os.path.join(processed_dir, "clean_historical_matches.parquet")
    modern_df.to_parquet(output_path, index=False)

    return output_path


def get_venue_country(venue_string: str) -> str:
    """Parses stadium text parameters to map the physical hosting nation country."""

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
