"""
Historical Feature Engineering and Time-Series Alignment Layer.

This module is responsible for computing opponent-adjusted performance metrics
and rigorously managing temporal alignment. It enforces strict Point-in-Time (PiT)
shifting architectures to guarantee zero data leakage between future target labels
and historical feature states.
"""

import logging
import os

import pandas as pd


def build_leakproof_form_features(
    df: pd.DataFrame, alpha: float = 1.5, baseline_elo: float = 1500.0
) -> pd.DataFrame:
    """
    Transforms chronologically ordered match data into context-aware performance vectors.

    Args:
        df (pd.DataFrame): Base match ledger.
        alpha (float): Exponential scaling factor for opponent difficulty adjustments.
        baseline_elo (float): Normalization constant for Elo scaling.

    Returns:
        pd.DataFrame: Match ledger augmented with PiT rolling form features.
    """

    # 1. Isolate home and away perspectives to create a unified linear team timeline
    home_perspective = df[
        [
            "date",
            "home_team",
            "home_score",
            "away_score",
            "home_elo_rating",
            "away_elo_rating",
        ]
    ].rename(
        columns={
            "home_team": "team",
            "home_score": "goals_for",
            "away_score": "goals_against",
            "home_elo_rating": "team_elo",
            "away_elo_rating": "opp_elo",
        }
    )
    home_perspective["is_home"] = 1

    away_perspective = df[
        [
            "date",
            "away_team",
            "away_score",
            "home_score",
            "away_elo_rating",
            "home_elo_rating",
        ]
    ].rename(
        columns={
            "away_team": "team",
            "away_score": "goals_for",
            "home_score": "goals_against",
            "away_elo_rating": "team_elo",
            "home_elo_rating": "opp_elo",
        }
    )
    away_perspective["is_home"] = 0

    # 2. Combine and sort chronologically per team to ensure proper rolling operations
    timeline = (
        pd.concat([home_perspective, away_perspective])
        .sort_values(by=["team", "date"])
        .reset_index(drop=True)
    )

    # 3. Apply Opponent-Adjusted Elo Scaling
    # Offensive: Rewards goals scored against elite defenses exponentially.
    timeline["adj_gf"] = timeline["goals_for"] * (
        (timeline["opp_elo"] / baseline_elo) ** alpha
    )
    # Defensive: Penalizes goals conceded against weak attacks exponentially.
    timeline["adj_ga"] = timeline["goals_against"] * (
        (baseline_elo / timeline["opp_elo"]) ** alpha
    )

    # 4. Compute Dual-Horizon EWMs and Volatility
    spans = [5, 15]
    eps = 1e-5  # Epsilon prevents zero-division

    for s in spans:
        # CRITICAL LEAKAGE GUARD: .shift(1) ensures the model only sees the moving average
        # calculated *prior* to the current match taking place.
        timeline[f"ewm_adj_gf_{s}"] = timeline.groupby("team")["adj_gf"].transform(
            lambda x: x.ewm(span=s, adjust=False, min_periods=1).mean().shift(1)
        )
        timeline[f"ewm_adj_ga_{s}"] = timeline.groupby("team")["adj_ga"].transform(
            lambda x: x.ewm(span=s, adjust=False, min_periods=1).mean().shift(1)
        )

        var_gf = timeline.groupby("team")["adj_gf"].transform(
            lambda x: x.ewm(span=s, adjust=False, min_periods=2).var().shift(1)
        )
        var_ga = timeline.groupby("team")["adj_ga"].transform(
            lambda x: x.ewm(span=s, adjust=False, min_periods=2).var().shift(1)
        )

        # Consistency Index: Coefficient of Variation (Volatility)
        timeline[f"cv_adj_gf_{s}"] = (var_gf**0.5) / (timeline[f"ewm_adj_gf_{s}"] + eps)
        timeline[f"cv_adj_ga_{s}"] = (var_ga**0.5) / (timeline[f"ewm_adj_ga_{s}"] + eps)

    # 5. Elo Momentum Delta
    # Extracts the pre-match Elo rating from exactly 5 matches ago.
    timeline["elo_momentum_5"] = timeline.groupby("team")["team_elo"].transform(
        lambda x: x - x.shift(5)
    )

    # 6. Handle initial baseline fill-forward for early history gaps
    ewm_cols = [c for c in timeline.columns if "ewm" in c]
    cv_mom_cols = [c for c in timeline.columns if "cv" in c or "momentum" in c]

    timeline[ewm_cols] = timeline.groupby("team")[ewm_cols].ffill().fillna(1.2)
    timeline[cv_mom_cols] = timeline.groupby("team")[cv_mom_cols].ffill().fillna(0.0)

    # 7. Split back out into pristine Home and Away feature frames
    home_features = timeline[timeline["is_home"] == 1].drop(
        columns=[
            "goals_for",
            "goals_against",
            "is_home",
            "adj_gf",
            "adj_ga",
            "team_elo",
            "opp_elo",
        ]
    )
    home_features = home_features.rename(
        columns={
            c: f"home_{c}" for c in home_features.columns if c not in ["date", "team"]
        }
    )

    away_features = timeline[timeline["is_home"] == 0].drop(
        columns=[
            "goals_for",
            "goals_against",
            "is_home",
            "adj_gf",
            "adj_ga",
            "team_elo",
            "opp_elo",
        ]
    )
    away_features = away_features.rename(
        columns={
            c: f"away_{c}" for c in away_features.columns if c not in ["date", "team"]
        }
    )

    # 8. Merge isolated perspectives back onto the primary match matrix
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
    """
    Compiles clean match logs, PiT Elo snapshots, and rolling form into a final XGBoost input matrix.

    Returns:
        tuple[pd.DataFrame, list[str]]: The final matrix and the exact column list used for modeling.
    """

    if not os.path.exists(matches_parquet_path):
        raise FileNotFoundError(f"Missing base match file: {matches_parquet_path}")

    df = pd.read_parquet(matches_parquet_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by="date").reset_index(drop=True)

    # 1. Rebuild Point-in-Time Elo History Internally
    home_elos = []
    away_elos = []
    current_elos = {}

    for _, row in df.iterrows():
        h_team = row["home_team"]
        a_team = row["away_team"]
        h_goals = int(row["home_score"])
        a_goals = int(row["away_score"])
        is_neutral = int(row.get("neutral", 0))
        match_weight = float(row.get("match_weight", 1.0))
        outcome = row.get("match_outcome", "Unknown")

        # Capture strictly Pre-Match Elos for feature generation
        r_home = current_elos.get(h_team, elo_engine.default_elo)
        r_away = current_elos.get(a_team, elo_engine.default_elo)
        home_elos.append(r_home)
        away_elos.append(r_away)

        # Process outcome inline to step the dictionary forward for the next iteration
        w_home, w_away = elo_engine._calculate_expected_score(
            r_home, r_away, is_neutral=is_neutral
        )

        if h_goals > a_goals:
            act_h, act_a = 1.0, 0.0
        elif a_goals > h_goals:
            act_h, act_a = 0.0, 1.0
        else:
            # Penalty Shootout Result
            if outcome == "Home_Win":
                act_h, act_a = 0.75, 0.25
            elif outcome == "Away_Win":
                act_h, act_a = 0.25, 0.75
            else:
                act_h, act_a = 0.5, 0.5

        g_factor = elo_engine._get_goal_margin_multiplier(h_goals, a_goals)
        current_k = elo_engine.k_factor * match_weight

        current_elos[h_team] = r_home + current_k * g_factor * (act_h - w_home)
        current_elos[a_team] = r_away + current_k * g_factor * (act_a - w_away)

    df["home_elo_rating"] = home_elos
    df["away_elo_rating"] = away_elos
    df["elo_differential"] = df["home_elo_rating"] - df["away_elo_rating"]

    # 2. Append Leak-Proof Moving Form Columns
    df = build_leakproof_form_features(df)

    # 3. Categorical Encoding
    df["is_neutral_venue"] = df["neutral"].astype(int)

    # 4. Final Feature Selection
    feature_columns = [
        "home_elo_rating",
        "away_elo_rating",
        "elo_differential",
        "is_neutral_venue",
        "home_elo_momentum_5",
        "away_elo_momentum_5",
        "home_ewm_adj_gf_5",
        "home_ewm_adj_ga_5",
        "home_cv_adj_gf_5",
        "home_cv_adj_ga_5",
        "home_ewm_adj_gf_15",
        "home_ewm_adj_ga_15",
        "home_cv_adj_gf_15",
        "home_cv_adj_ga_15",
        "away_ewm_adj_gf_5",
        "away_ewm_adj_ga_5",
        "away_cv_adj_gf_5",
        "away_cv_adj_ga_5",
        "away_ewm_adj_gf_15",
        "away_ewm_adj_ga_15",
        "away_cv_adj_gf_15",
        "away_cv_adj_ga_15",
    ]

    targets = ["home_score", "away_score"]

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


def extract_latest_team_form(feature_matrix, participating_teams):
    """Extracts the final pre-tournament EWM form states to feed the ML models during forecasting."""

    latest_team_form = {}

    for team in participating_teams:
        team_rows = feature_matrix[
            (feature_matrix["home_team"] == team)
            | (feature_matrix["away_team"] == team)
        ]

        if not team_rows.empty:
            latest_row = team_rows.iloc[-1]
            prefix = "home_" if latest_row["home_team"] == team else "away_"
            latest_team_form[team] = {
                "elo_momentum_5": latest_row[f"{prefix}elo_momentum_5"],
                "ewm_adj_gf_5": latest_row[f"{prefix}ewm_adj_gf_5"],
                "ewm_adj_ga_5": latest_row[f"{prefix}ewm_adj_ga_5"],
                "cv_adj_gf_5": latest_row[f"{prefix}cv_adj_gf_5"],
                "cv_adj_ga_5": latest_row[f"{prefix}cv_adj_ga_5"],
                "ewm_adj_gf_15": latest_row[f"{prefix}ewm_adj_gf_15"],
                "ewm_adj_ga_15": latest_row[f"{prefix}ewm_adj_ga_15"],
                "cv_adj_gf_15": latest_row[f"{prefix}cv_adj_gf_15"],
                "cv_adj_ga_15": latest_row[f"{prefix}cv_adj_ga_15"],
            }
        else:
            # Fallback for entities with zero historical footprint
            latest_team_form[team] = {
                "elo_momentum_5": 0.0,
                "ewm_adj_gf_5": 1.2,
                "ewm_adj_ga_5": 1.2,
                "cv_adj_gf_5": 0.0,
                "cv_adj_ga_5": 0.0,
                "ewm_adj_gf_15": 1.2,
                "ewm_adj_ga_15": 1.2,
                "cv_adj_gf_15": 0.0,
                "cv_adj_ga_15": 0.0,
            }

    return latest_team_form


def test_point_in_time_leakage():
    """Unit test asserting strict PiT architecture by verifying mathematical shifting."""

    test_df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=6),
            "home_team": ["A", "B", "A", "C", "A", "D"],
            "away_team": ["B", "A", "C", "A", "D", "A"],
            "home_score": [2, 1, 3, 0, 1, 2],
            "away_score": [1, 2, 0, 1, 1, 0],
            "home_elo_rating": [1500, 1490, 1510, 1495, 1520, 1500],
            "away_elo_rating": [1500, 1510, 1490, 1520, 1495, 1530],
        }
    )

    processed_df = build_leakproof_form_features(test_df)
    team_a_rows = processed_df[
        (processed_df["home_team"] == "A") | (processed_df["away_team"] == "A")
    ]

    # The first match for A shouldn't have EWM > 1.2 (baseline fallback).
    # If it does, future data has leaked backwards into the feature state.
    first_match = team_a_rows.iloc[0]
    assert first_match["home_ewm_adj_gf_5"] == 1.2, (
        "Leakage Alert: Future data influenced the first match vector."
    )

    logging.info("✅ PiT architecture verified. Zero leakage detected.")


if __name__ == "__main__":
    test_point_in_time_leakage()
