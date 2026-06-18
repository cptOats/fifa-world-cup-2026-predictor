"""Historical Feature Engineering and Time-Series Alignment Layer."""

import os

import pandas as pd


def build_leakproof_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transforms match data into an interleaved timeline to compute leak-proof form."""

    # 1. Isolate home and away perspectives to create a unified team timeline
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

    # 4. Compute EWM (Continuous decay) - .shift(1) inside the transform ensures strict lookback protection
    spans = [4, 10]  # Short-term momentum vs long-term baseline stability
    for s in spans:
        timeline[f"team_ewm_gf_{s}s"] = timeline.groupby("team")["goals_for"].transform(
            lambda x: x.ewm(span=s, adjust=False).mean().shift(1)
        )
        timeline[f"team_ewm_ga_{s}s"] = timeline.groupby("team")[
            "goals_against"
        ].transform(lambda x: x.ewm(span=s, adjust=False).mean().shift(1))
        timeline[f"team_ewm_wr_{s}s"] = timeline.groupby("team")["is_win"].transform(
            lambda x: x.ewm(span=s, adjust=False).mean().shift(1)
        )

    # 5. Handle initial baseline fill-forward for early history gaps
    fill_cols = [c for c in timeline.columns if "ewm" in c]
    timeline[fill_cols] = timeline.groupby("team")[fill_cols].ffill().fillna(0)

    # 6. Split back out into pristine Home and Away feature frames
    home_features = timeline[timeline["is_home"] == 1].drop(
        columns=["goals_for", "goals_against", "is_home", "is_win"]
    )
    home_features = home_features.rename(
        columns={
            c: f"home_{c}" for c in home_features.columns if c not in ["date", "team"]
        }
    )

    away_features = timeline[timeline["is_home"] == 0].drop(
        columns=["goals_for", "goals_against", "is_home", "is_win"]
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


def compile_master_feature_matrix(
    matches_parquet_path: str, elo_engine
) -> tuple[pd.DataFrame, list[str]]:
    """Compiles clean matches, momentum vectors, and Elo snapshots into a master matrix."""

    if not os.path.exists(matches_parquet_path):
        raise FileNotFoundError(f"Missing base match file: {matches_parquet_path}")

    # Read clean base features
    df = pd.read_parquet(matches_parquet_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by="date").reset_index(drop=True)

    # 1. Append Leak-Proof Moving Form Columns
    df = build_leakproof_form_features(df)

    # 2. Extract Point-in-Time Historical Elo States
    home_elos = []
    away_elos = []

    # Interacts cleanly with src/elo.py by passing clean string team identifiers
    for idx, row in df.iterrows():
        h_team = row["home_team"]
        a_team = row["away_team"]

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
        "home_team_ewm_gf_4s",
        "home_team_ewm_ga_4s",
        "home_team_ewm_wr_4s",
        "home_team_ewm_gf_10s",
        "home_team_ewm_ga_10s",
        "home_team_ewm_wr_10s",
        "away_team_ewm_gf_4s",
        "away_team_ewm_ga_4s",
        "away_team_ewm_wr_4s",
        "away_team_ewm_gf_10s",
        "away_team_ewm_ga_10s",
        "away_team_ewm_wr_10s",
    ]

    targets = ["home_score", "away_score"]

    # 5. Extract matrix and rename 'date' to 'match_date' for direct XGBoost synergy
    final_matrix = (
        df[
            ["date", "home_team", "away_team", "match_weight"]
            + feature_columns
            + targets
        ]
        .dropna()
        .rename(columns={"date": "match_date"})
        .reset_index(drop=True)
    )

    return final_matrix, feature_columns
