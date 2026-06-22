"""
Data Transformation, Entity Resolution, and Feature Engineering Layer.

Applies robust character normalization and entity mapping to unify separate
data sources (e.g., DataCamp blueprints vs. historical Kaggle logs).
Extracts pure match history logic from chaotic historical string structures.
"""

import logging
import os

import numpy as np
import pandas as pd

# Hard-coded Translation Dictionary spanning modern discrepancies
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
    """
    Constructs an accent-normalized backward compatibility map from Kaggle metadata.
    Handles asymmetrical character mutations historically embedded in CSVs.
    """

    if not os.path.exists(former_names_path):
        return {}

    df_names = pd.read_csv(former_names_path)
    name_map = {}

    for _, row in df_names.iterrows():
        current_name = str(row["current"]).strip()
        former_name = str(row["former"]).strip()
        name_map[former_name] = current_name

        # Enforce symmetrical character normalizations bypassing legacy data entry drift
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
    """
    Hard-fault execution gate. Asserts pure 1:1 entity crossover between
    future tournament blueprint data and legacy Kaggle dataset strings.
    """

    raw_dir = os.path.join("data", "raw")
    fixtures_df = pd.read_csv(os.path.join(raw_dir, "group_fixtures.csv"))

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

    # Calculate absolute unmapped entities
    true_mismatches = (datacamp_teams - kaggle_teams) - {
        "FIFA Playoff 1",
        "FIFA Playoff 2",
        "UEFA Playoff A",
        "UEFA Playoff B",
        "UEFA Playoff C",
        "UEFA Playoff D",
    }

    if true_mismatches:
        logging.critical(
            f"❌ CRITICAL ENTITY DISCREPANCY: Unmapped country string variants detected: {true_mismatches}"
        )
        logging.critical(
            "Please expand the 'DATACAMP_TO_KAGGLE' translation dictionary in src/transform.py to resolve these."
        )
        raise LookupError("Pipeline halted due to unresolved entity names.")


def patch_tournament_structures() -> None:
    """
    Applies programmatic patches to the raw DataCamp competition CSVs.
    Fixes chronological cosmetic misalignments in the group stages and
    resolves a major structural routing bug in the Round of 32 knockout
    bracket to align with official FIFA World Cup 2026 documentation.
    """
    raw_dir = os.path.join("data", "raw")
    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    # --- 1. PATCH GROUP STAGES (Cosmetic & Chronological Fixes) ---
    fixtures_path = os.path.join(raw_dir, "group_fixtures.csv")
    if os.path.exists(fixtures_path):
        df_groups = pd.read_csv(fixtures_path)

        # Fix the chronological ordering bugs by swapping their IDs
        group_order_patch = {
            29: 30,
            30: 29,  # Group D2
            33: 34,
            34: 33,  # Group F2
            17: 20,
            20: 17,  # Group J1
        }
        df_groups["match_id"] = df_groups["match_id"].replace(group_order_patch)
        df_groups = df_groups.sort_values(by="match_id").reset_index(drop=True)

        # Fix the home/away assignment for match 59 (USA vs Turkey)
        idx_59 = df_groups["match_id"] == 59
        if idx_59.any():
            home_temp = df_groups.loc[idx_59, "home_team"].copy()
            df_groups.loc[idx_59, "home_team"] = df_groups.loc[idx_59, "away_team"]
            df_groups.loc[idx_59, "away_team"] = home_temp

        # Save to processed directory to protect the raw file
        df_groups.to_csv(
            os.path.join(processed_dir, "clean_group_fixtures.csv"), index=False
        )

    # --- 2. PATCH KNOCKOUT SLOTS (Major Bracket Routing Fix) ---
    knockout_path = os.path.join(raw_dir, "knockout_slots.csv")
    if os.path.exists(knockout_path):
        df_knockout = pd.read_csv(knockout_path)

        # Dictionary of incorrect match mappings to their true FIFA counterparts
        r32_patch = {
            74: 76,
            75: 74,
            76: 75,
            77: 78,
            78: 77,
            81: 82,
            82: 81,
            86: 88,
            87: 86,
            88: 87,
        }

        # Simultaneous replacement prevents cascading overwrites
        df_knockout["match_id"] = df_knockout["match_id"].replace(r32_patch)
        df_knockout = df_knockout.sort_values(by="match_id").reset_index(drop=True)

        # Save to processed directory
        df_knockout.to_csv(
            os.path.join(processed_dir, "clean_knockout_slots.csv"), index=False
        )


def prepare_historical_features(translation_dict, training_variables) -> str:
    """
    Executes base ETL pipeline, resolving names, applying time slicing,
    and generating the raw un-shifted feature layout structure.
    """

    _validate_entity_resolution(translation_dict)

    raw_dir = os.path.join("data", "raw")
    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    results_df = pd.read_csv(os.path.join(raw_dir, "results.csv"))

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

    # Slice strictly to target learning horizon
    modern_df = results_df[
        (results_df["date"] >= training_variables["time_slice_start"])
        & (results_df["date"] < training_variables["start_of_tournament"])
    ].copy()

    # Friendly degradation weighting
    modern_df["match_weight"] = np.where(
        modern_df["tournament"] == "Friendly",
        training_variables["friendly_weight"],
        1.0,
    )

    # Calculate boolean truth metrics for true venue hosting parameters
    co_hosts = {"United States", "Mexico", "Canada"}
    modern_df["is_true_home"] = modern_df.apply(
        lambda row: row["home_team"] in co_hosts and row["country"] == row["home_team"],
        axis=1,
    )

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
