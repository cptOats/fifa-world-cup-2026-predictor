"""Dixon-Coles Poisson Model."""

import json
import math
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


def train_poisson_ratings(
    poisson_alpha=0.00047,
) -> tuple[dict[str, dict[str, float]], float, float, float]:
    """Calculates weighted historical attack and defense strengths for all teams."""

    ARTIFACTS_DIR = os.path.join("data", "artifacts")
    POISSON_PATH = os.path.join(ARTIFACTS_DIR, "poisson_artifacts.json")

    if os.path.exists(POISSON_PATH):
        with open(POISSON_PATH, "r") as f:
            artifacts = json.load(f)
        return (
            artifacts["ratings"],
            artifacts["global_home_avg"],
            artifacts["global_away_avg"],
            artifacts["global_neutral_avg"],
        )

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    processed_path = os.path.join(
        "data", "processed", "clean_historical_matches.parquet"
    )
    df = pd.read_parquet(processed_path)
    train_df = df[df["match_weight"] > 0].copy()

    # Enforce chronological datetime calculation
    train_df["date"] = pd.to_datetime(train_df["date"])
    max_date = train_df["date"].max()
    days_elapsed = (max_date - train_df["date"]).dt.days

    # Apply exponential decay to the existing importance weight
    time_decay = np.exp(-poisson_alpha * days_elapsed)
    train_df["match_weight"] = train_df["match_weight"] * time_decay

    # Symmetrical venue splitting across historical dataframes
    non_neutral_df = train_df[train_df["neutral"] == 0].copy()
    neutral_df = train_df[train_df["neutral"] == 1].copy()

    # 1. Compute baseline home/away expectations exclusively on true non-neutral turf
    total_non_neutral_weight = non_neutral_df["match_weight"].sum()
    if total_non_neutral_weight > 0:
        global_home_avg = (
            non_neutral_df["home_score"] * non_neutral_df["match_weight"]
        ).sum() / total_non_neutral_weight
        global_away_avg = (
            non_neutral_df["away_score"] * non_neutral_df["match_weight"]
        ).sum() / total_non_neutral_weight
    else:
        global_home_avg, global_away_avg = 1.35, 1.35

    # 2. Compute true neutral venue expectations by pooling goals uniformly
    total_neutral_weight = neutral_df["match_weight"].sum()
    if total_neutral_weight > 0:
        neutral_goals_sum = (
            neutral_df["home_score"] * neutral_df["match_weight"]
        ).sum() + (neutral_df["away_score"] * neutral_df["match_weight"]).sum()
        global_neutral_avg = neutral_goals_sum / (2 * total_neutral_weight)
    else:
        global_neutral_avg = (global_home_avg + global_away_avg) / 2.0

    # 3. Compile individual team statistics matrices
    train_df["weighted_home_goals_scored"] = (
        train_df["home_score"] * train_df["match_weight"]
    )
    train_df["weighted_home_goals_conceded"] = (
        train_df["away_score"] * train_df["match_weight"]
    )

    home_stats = (
        train_df.groupby("home_team")
        .agg(
            goals_scored=("weighted_home_goals_scored", "sum"),
            goals_conceded=("weighted_home_goals_conceded", "sum"),
            weight_sum=("match_weight", "sum"),
        )
        .reset_index()
    )

    train_df["weighted_away_goals_scored"] = (
        train_df["away_score"] * train_df["match_weight"]
    )
    train_df["weighted_away_goals_conceded"] = (
        train_df["home_score"] * train_df["match_weight"]
    )

    away_stats = (
        train_df.groupby("away_team")
        .agg(
            goals_scored=("weighted_away_goals_scored", "sum"),
            goals_conceded=("weighted_away_goals_conceded", "sum"),
            weight_sum=("match_weight", "sum"),
        )
        .reset_index()
    )

    teams = set(home_stats["home_team"]).union(set(away_stats["away_team"]))
    ratings = {}

    for team in teams:
        h_row = home_stats[home_stats["home_team"] == team]
        a_row = away_stats[away_stats["away_team"] == team]

        h_sc = h_row["goals_scored"].values[0] if not h_row.empty else 0
        h_cn = h_row["goals_conceded"].values[0] if not h_row.empty else 0
        h_wt = h_row["weight_sum"].values[0] if not h_row.empty else 0

        a_sc = a_row["goals_scored"].values[0] if not a_row.empty else 0
        a_cn = a_row["goals_conceded"].values[0] if not a_row.empty else 0
        a_wt = a_row["weight_sum"].values[0] if not a_row.empty else 0

        total_matches_weighted = h_wt + a_wt
        if total_matches_weighted == 0:
            ratings[team] = {"attack": 1.0, "defense": 1.0}
            continue

        avg_scored = (h_sc + a_sc) / total_matches_weighted
        avg_conceded = (h_cn + a_cn) / total_matches_weighted

        # Normalize team ratings against the true expected baseline of a neutral field
        attack_rating = avg_scored / global_neutral_avg
        defense_rating = avg_conceded / global_neutral_avg

        # Apply structural clipping boundaries to handle extreme outliers safely
        if attack_rating < 0.1000:
            attack_rating = 0.1000
        if defense_rating < 0.2500:
            defense_rating = 0.2500

        ratings[team] = {"attack": attack_rating, "defense": defense_rating}

    # Save to disk including the new neutral metric parameter
    artifacts = {
        "global_home_avg": float(global_home_avg),
        "global_away_avg": float(global_away_avg),
        "global_neutral_avg": float(global_neutral_avg),
        "ratings": ratings,
    }

    with open(POISSON_PATH, "w") as f:
        json.dump(artifacts, f, indent=4)

    return ratings, global_home_avg, global_away_avg, global_neutral_avg


def train_poisson_oof_predictions(feature_matrix, poisson_alpha=0.00047):
    """Calculates leak-proof, continuous out-of-fold Poisson predictions via cross-validation."""

    n_matches = len(feature_matrix)
    oof_home_preds = np.zeros(n_matches)
    oof_away_preds = np.zeros(n_matches)

    tscv = TimeSeriesSplit(n_splits=3)

    for fold, (train_idx, test_idx) in enumerate(tscv.split(feature_matrix), 1):
        train_df = feature_matrix.iloc[train_idx].copy()
        test_df = feature_matrix.iloc[test_idx]

        if "match_weight" not in train_df.columns:
            train_df["match_weight"] = 1.0

        # Calculate fold-specific decay using the fold's own timeline frontier
        train_df["match_date"] = pd.to_datetime(train_df["match_date"])
        fold_max_date = train_df["match_date"].max()
        days_elapsed = (fold_max_date - train_df["match_date"]).dt.days

        time_decay = np.exp(-poisson_alpha * days_elapsed)
        train_df["match_weight"] = train_df["match_weight"] * time_decay

        total_weight = train_df["match_weight"].sum()
        g_home = (
            train_df["home_score"] * train_df["match_weight"]
        ).sum() / total_weight
        g_away = (
            train_df["away_score"] * train_df["match_weight"]
        ).sum() / total_weight
        g_neutral = (g_home + g_away) / 2.0

        # Calculate isolated fold statistics
        train_df["w_hd_sc"] = train_df["home_score"] * train_df["match_weight"]
        train_df["w_hd_cn"] = train_df["away_score"] * train_df["match_weight"]
        home_stats = (
            train_df.groupby("home_team")
            .agg(
                sc=("w_hd_sc", "sum"), cn=("w_hd_cn", "sum"), wt=("match_weight", "sum")
            )
            .reset_index()
        )

        train_df["w_aw_sc"] = train_df["away_score"] * train_df["match_weight"]
        train_df["w_aw_cn"] = train_df["home_score"] * train_df["match_weight"]
        away_stats = (
            train_df.groupby("away_team")
            .agg(
                sc=("w_aw_sc", "sum"), cn=("w_aw_cn", "sum"), wt=("match_weight", "sum")
            )
            .reset_index()
        )

        teams = set(home_stats["home_team"]).union(set(away_stats["away_team"]))
        fold_ratings = {}

        for team in teams:
            h = home_stats[home_stats["home_team"] == team]
            a = away_stats[away_stats["away_team"] == team]

            h_sc = h["sc"].values[0] if not h.empty else 0
            h_cn = h["cn"].values[0] if not h.empty else 0
            h_wt = h["wt"].values[0] if not h.empty else 0

            a_sc = a["sc"].values[0] if not a.empty else 0
            a_cn = a["cn"].values[0] if not a.empty else 0
            a_wt = a["wt"].values[0] if not a.empty else 0

            total_matches = h_wt + a_wt
            if total_matches == 0:
                fold_ratings[team] = {"attack": 1.0, "defense": 1.0}
                continue

            attack = ((h_sc + a_sc) / total_matches) / g_neutral
            defense = ((h_cn + a_cn) / total_matches) / g_neutral

            fold_ratings[team] = {
                "attack": max(0.1000, attack),
                "defense": max(0.2500, defense),
            }

        # Predict continuous expected values (lambda) for the unseen test rows
        for idx, row in test_df.iterrows():
            h_team = row["home_team"]
            a_team = row["away_team"]

            h_rat = fold_ratings.get(h_team, {"attack": 1.0, "defense": 1.0})
            a_rat = fold_ratings.get(a_team, {"attack": 1.0, "defense": 1.0})

            oof_home_preds[idx] = h_rat["attack"] * a_rat["defense"] * g_neutral
            oof_away_preds[idx] = a_rat["attack"] * h_rat["defense"] * g_neutral

    return oof_home_preds, oof_away_preds


def predict_poisson_match(
    home, away, venue_country, ratings, g_home, g_away, g_neutral
) -> tuple[float, float, float, float, float]:
    """Generates expected goals, total corners, and card counts using pure Poisson parameters."""

    home_rating = ratings.get(home, {"attack": 1.0, "defense": 1.0})
    away_rating = ratings.get(away, {"attack": 1.0, "defense": 1.0})

    # Directional venue gating resolution
    if home == venue_country:
        # Team A is the true host nation
        lambda_home = home_rating["attack"] * away_rating["defense"] * g_home
        lambda_away = away_rating["attack"] * home_rating["defense"] * g_away
        is_neutral = 0
    elif away == venue_country:
        # Team B is the true host nation (Prevents the asymmetry bug)
        lambda_home = home_rating["attack"] * away_rating["defense"] * g_away
        lambda_away = away_rating["attack"] * home_rating["defense"] * g_home
        is_neutral = 0
    else:
        # True Neutral match turf condition
        lambda_home = home_rating["attack"] * away_rating["defense"] * g_neutral
        lambda_away = away_rating["attack"] * home_rating["defense"] * g_neutral
        is_neutral = 1

    # PROXY METRICS
    raw_corners = (5.5 * home_rating["attack"] * away_rating["defense"]) + (
        5.5 * away_rating["attack"] * home_rating["defense"]
    )
    raw_yellows = (3.0 * home_rating["defense"] * away_rating["attack"]) + (
        3.0 * away_rating["defense"] * home_rating["attack"]
    )
    raw_reds = 0.12 if is_neutral == 0 else 0.10

    return lambda_home, lambda_away, raw_corners, raw_yellows, raw_reds

# Unused function
def get_dixon_coles_score(lambda_home, lambda_away, rho=-0.10):
    """Evaluates a joint probability grid up to a 5-5 scoreline with low-score coupling."""
    best_prob = -1.0
    pred_home, pred_away = 0, 0

    # Evaluate a realistic 6x6 grid of international football outcomes
    for x in range(6):
        for y in range(6):
            # Compute independent Poisson probabilities
            prob_home = _matrix_lambda_calc(lambda_home, x)
            prob_away = _matrix_lambda_calc(lambda_away, y)
            joint_prob = prob_home * prob_away

            # Apply Dixon-Coles tau adjustment layer for low-scoring states
            if x == 0 and y == 0:
                tau = 1.0 - (rho * lambda_home * lambda_away)
            elif x == 1 and y == 0:
                tau = 1.0 + (rho * lambda_away)
            elif x == 0 and y == 1:
                tau = 1.0 + (rho * lambda_home)
            elif x == 1 and y == 1:
                tau = 1.0 - rho
            else:
                tau = 1.0

            adjusted_prob = joint_prob * tau

            # Track the peak of the probability mass function
            if adjusted_prob > best_prob:
                best_prob = adjusted_prob
                pred_home, pred_away = x, y

    return pred_home, pred_away

# Unused function
def _matrix_lambda_calc(lam, k):
    """Evaluates the standard Poisson Probability Mass Function."""
    return (lam**k * math.exp(-lam)) / math.factorial(k)
