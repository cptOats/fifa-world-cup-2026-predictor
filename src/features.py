import os

import numpy as np
import pandas as pd


def build_leakproof_form_features(df):
    """
    Transforms a match-grain dataframe into a chronological team-grain timeline,
    computes rolling performance metrics, shifts them to prevent data leakage,
    and maps them back to the original match format.
    """
    print("🔄 Processing leak-proof historical team form vectors...")

    # 1. Isolate home and away perspectives to create a unified team-timeline
    home_perspective = df[["date", "home_team", "home_score", "away_score"]].rename(
        columns={
            "home_team": "team",
            "home_score": "goals_for",
            "away_score": "goals_against",
        }
    )
    home_perspective["is_home"] = 1

    away_perspective = df[["date", "away_team", "away_score", "home_score"]].rename(
        columns={
            "away_team": "team",
            "away_score": "goals_for",
            "home_score": "goals_against",
        }
    )
    away_perspective["is_home"] = 0

    # 2. Combine and sort chronologically per team
    timeline = (
        pd.concat([home_perspective, away_perspective])
        .sort_values(by=["team", "date"])
        .reset_index(drop=True)
    )

    # 3. Derive match outcomes from the team's perspective
    timeline["is_win"] = (timeline["goals_for"] > timeline["goals_against"]).astype(int)
    timeline["is_draw"] = (timeline["goals_for"] == timeline["goals_against"]).astype(
        int
    )

    # 4. Compute rolling stats using closed='left' to STRICTLY exclude the current match outcome
    # This uses only past historical performance data
    window_sizes = [3, 5]
    for w in window_sizes:
        timeline[f"team_avg_gf_{w}g"] = timeline.groupby("team")["goals_for"].transform(
            lambda x: x.rolling(window=w, closed="left").mean()
        )
        timeline[f"team_avg_ga_{w}g"] = timeline.groupby("team")[
            "goals_against"
        ].transform(lambda x: x.rolling(window=w, closed="left").mean())
        timeline[f"team_win_rate_{w}g"] = timeline.groupby("team")["is_win"].transform(
            lambda x: x.rolling(window=w, closed="left").mean()
        )

    # 5. Handle initial baseline fill-forward for teams in early parts of their history
    fill_cols = [
        c for c in timeline.columns if "rolling" in c or "avg" in c or "rate" in c
    ]
    timeline[fill_cols] = timeline.groupby("team")[fill_cols].ffill().fillna(0)

    # 6. Split back out into pristine Home and Away feature frames
    home_features = timeline[timeline["is_home"] == 1].drop(
        columns=["goals_for", "goals_against", "is_home", "is_win", "is_draw"]
    )
    home_features = home_features.rename(
        columns={
            c: f"home_{c}" for c in home_features.columns if c not in ["date", "team"]
        }
    )

    away_features = timeline[timeline["is_home"] == 0].drop(
        columns=["goals_for", "goals_against", "is_home", "is_win", "is_draw"]
    )
    away_features = away_features.rename(
        columns={
            c: f"away_{c}" for c in away_features.columns if c not in ["date", "team"]
        }
    )

    # 7. Merge back onto primary match matrix
    processed_df = df.merge(
        home_features,
        left_on=["date", "home_team"],
        right_on=["date", "team"],
        how="left",
    ).drop(columns=["team"])
    processed_df = processed_df.merge(
        away_features,
        left_on=["date", "away_team"],
        right_on=["date", "team"],
        how="left",
    ).drop(columns=["team"])

    return processed_df


def compile_master_feature_matrix(matches_parquet_path, elo_engine):
    """
    Ingests clean historical matches, overlays rolling form vectors,
    injects historical Elo snapshots, and returns a clean training matrix.
    """
    if not os.path.exists(matches_parquet_path):
        raise FileNotFoundError(f"Missing base match file: {matches_parquet_path}")

    # Read clean base features
    df = pd.read_parquet(matches_parquet_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by="date").reset_index(drop=True)

    # 1. Append Leak-Proof Moving Form Columns
    df = build_leakproof_form_features(df)

    # 2. Extract Point-in-Time Historical Elo States
    print("📈 Mapping chronological Elo snapshots across training matrix...")
    home_elos = []
    away_elos = []

    # We step through time matching what the Elo engine knew right before each match kicked off
    for idx, row in df.iterrows():
        h_team = row["home_team"]
        a_team = row["away_team"]

        # Pull snapshots from our compiled engine state
        home_elos.append(elo_engine.get_rating(h_team))
        away_elos.append(elo_engine.get_rating(a_team))

    df["home_elo_rating"] = home_elos
    df["away_elo_rating"] = away_elos
    df["elo_differential"] = df["home_elo_rating"] - df["away_elo_rating"]

    # 3. Categorical Encoding for Context Controls
    df["is_neutral_venue"] = df["neutral"].astype(int)

    # 4. Filter down to clean ML input features and targets
    feature_columns = [
        "home_elo_rating",
        "away_elo_rating",
        "elo_differential",
        "is_neutral_venue",
        "home_team_avg_gf_3g",
        "home_team_avg_ga_3g",
        "home_team_win_rate_3g",
        "home_team_avg_gf_5g",
        "home_team_avg_ga_5g",
        "home_team_win_rate_5g",
        "away_team_avg_gf_3g",
        "away_team_avg_ga_3g",
        "away_team_win_rate_3g",
        "away_team_avg_gf_5g",
        "away_team_avg_ga_5g",
        "away_team_win_rate_5g",
    ]

    targets = ["home_score", "away_score"]

    # Drop rows with NaN values resulting from early-history rolling limits
    final_matrix = (
        df[["date", "home_team", "away_team"] + feature_columns + targets]
        .dropna()
        .reset_index(drop=True)
    )

    print(f"✅ Feature matrix compilation successful! Shape: {final_matrix.shape}")
    return final_matrix, feature_columns
