import json
import math
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

# The Master Entity Resolution Translation Layer
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

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "poisson_artifacts.json")


def train_poisson_ratings():
    """Calculates weighted attack/defense strengths for final production 2026 runs."""
    if os.path.exists(MODEL_PATH):
        print(f"💾 Loading pre-compiled Poisson model from cache: {MODEL_PATH}")
        with open(MODEL_PATH, "r") as f:
            artifacts = json.load(f)
        return (
            artifacts["ratings"],
            artifacts["global_home_avg"],
            artifacts["global_away_avg"],
        )

    print("🧠 Cache miss. Compiling team ratings from historical data...")
    os.makedirs(MODEL_DIR, exist_ok=True)

    processed_path = os.path.join(
        "data", "processed", "clean_historical_matches.parquet"
    )
    df = pd.read_parquet(processed_path)
    train_df = df[df["match_weight"] > 0].copy()

    total_weight = train_df["match_weight"].sum()
    global_home_avg = (
        train_df["home_score"] * train_df["match_weight"]
    ).sum() / total_weight
    global_away_avg = (
        train_df["away_score"] * train_df["match_weight"]
    ).sum() / total_weight

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

        attack_rating = avg_scored / ((global_home_avg + global_away_avg) / 2)
        defense_rating = avg_conceded / ((global_home_avg + global_away_avg) / 2)

        if attack_rating < 0.1000:
            attack_rating = 0.1000
        if defense_rating < 0.2500:
            defense_rating = 0.2500

        ratings[team] = {"attack": attack_rating, "defense": defense_rating}

    artifacts = {
        "global_home_avg": float(global_home_avg),
        "global_away_avg": float(global_away_avg),
        "ratings": ratings,
    }

    with open(MODEL_PATH, "w") as f:
        json.dump(artifacts, f, indent=4)
    print(f"💾 Model architecture serialized and stored at: {MODEL_PATH}")

    return ratings, global_home_avg, global_away_avg


def train_poisson_oof_predictions(feature_matrix):
    """
    Executes an identical chronological cross-validation loop to calculate
    leak-proof, continuous out-of-fold Poisson predictions for the blender.
    """
    print("📋 Generating Out-of-Fold Poisson Baseline Horizon...")
    n_matches = len(feature_matrix)
    oof_home_preds = np.zeros(n_matches)
    oof_away_preds = np.zeros(n_matches)

    tscv = TimeSeriesSplit(n_splits=3)

    for fold, (train_idx, test_idx) in enumerate(tscv.split(feature_matrix), 1):
        train_df = feature_matrix.iloc[train_idx].copy()
        test_df = feature_matrix.iloc[test_idx]

        # Use 1.0 default if match_weight was dropped during earlier matrix slicing
        if "match_weight" not in train_df.columns:
            train_df["match_weight"] = 1.0

        total_weight = train_df["match_weight"].sum()
        g_home = train_df["home_score"].sum() / len(train_df)
        g_away = train_df["away_score"].sum() / len(train_df)
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


def get_venue_country(venue_string):
    """Parses the stadium venue string to identify the physical host country."""
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


def predict_match_score(home, away, venue, ratings, g_home_avg, g_away_avg):
    """Generates expected goal counts, corners, and cards using structural proxies."""
    home_rating = ratings.get(home, {"attack": 1.0, "defense": 1.0})
    away_rating = ratings.get(away, {"attack": 1.0, "defense": 1.0})

    venue_country = get_venue_country(venue)
    g_neutral = (g_home_avg + g_away_avg) / 2.0

    # 1. GOAL CALCULATIONS (Symmetrical Environmental Baselines)
    if home == venue_country:
        lambda_home = home_rating["attack"] * away_rating["defense"] * g_home_avg
        lambda_away = away_rating["attack"] * home_rating["defense"] * g_away_avg
    elif away == venue_country:
        lambda_home = home_rating["attack"] * away_rating["defense"] * g_away_avg
        lambda_away = away_rating["attack"] * home_rating["defense"] * g_home_avg
    else:
        lambda_home = home_rating["attack"] * away_rating["defense"] * g_neutral
        lambda_away = away_rating["attack"] * home_rating["defense"] * g_neutral

    # Directly integrate Dixon-Coles Poisson probability :
    pred_home_score, pred_away_score = get_dixon_coles_score(lambda_home, lambda_away)

    # 2. CORNERS PROXY MODEL (scaled by matchup threat)
    home_corners = 5.5 * home_rating["attack"] * away_rating["defense"]
    away_corners = 5.5 * away_rating["attack"] * home_rating["defense"]
    total_corners = int(np.clip(np.round(home_corners + away_corners), 5, 16))

    # 3. YELLOW CARDS PROXY MODEL (scaled by defensive pressure)
    home_cards = 3.0 * home_rating["defense"] * away_rating["attack"]
    away_cards = 3.0 * away_rating["defense"] * home_rating["attack"]
    total_yellows = int(np.clip(np.round(home_cards + away_cards), 1, 9))

    # 4. RED CARDS (Most games have no red cards - high-variance, low-frequency anomalies)
    total_reds = 0

    # 5. WINNING TEAM STRING
    if pred_home_score > pred_away_score:
        win_label = "home"
    elif pred_away_score > pred_home_score:
        win_label = "away"
    else:
        win_label = "draw"

    return (
        pred_home_score,
        pred_away_score,
        total_corners,
        total_yellows,
        total_reds,
        win_label,
    )


def print_team_power_rankings(ratings):
    """Parses the ratings dictionary, computes a unified Dominance Ratio, and ranks teams from most dominant to least dominant."""
    records = []
    for team, metrics in ratings.items():
        # Calculate dominance ratio
        dominance_ratio = metrics["attack"] / metrics["defense"]

        records.append(
            {
                "Team": team,
                "Attack Power": metrics["attack"],
                "Defense Power": metrics["defense"],
                "Dominance Ratio": dominance_ratio,
            }
        )

    df_rankings = pd.DataFrame(records)

    # SORT BY DOMINANCE RATIO DESCENDING
    df_rankings = df_rankings.sort_values(
        by="Dominance Ratio", ascending=False
    ).reset_index(drop=True)

    print("\n🔥 Team Metric Power Rankings (Ranked by Dominance Ratio):")
    print("=" * 100)
    print(
        f"{'Rank':<5} | {'Team':<32} | {'Attack Power':<15} | {'Defense Power':<15} | {'Dominance Ratio':<15}"
    )
    print("-" * 100)
    for idx, row in df_rankings.iterrows():
        print(
            f"{idx + 1:<5} | {row['Team']:<32} | {row['Attack Power']:<15.3f} | {row['Defense Power']:<15.3f} | {row['Dominance Ratio']:<15.3f}"
        )
    print("=" * 100)


def get_dixon_coles_score(lambda_home, lambda_away, rho=-0.10):
    """Evaluates a joint probability distribution grid up to a 5-5 scoreline,
    applies the Dixon-Coles low-score coupling adjustment, and returns
    the absolute most probable integer scoreline (the distribution mode).
    """
    best_prob = -1.0
    pred_home, pred_away = 0, 0

    # Evaluate a realistic 6x6 grid of international football outcomes
    for x in range(6):
        for y in range(6):
            # Compute independent Poisson probabilities
            prob_home = matrix_lambda_calc(lambda_home, x)
            prob_away = matrix_lambda_calc(lambda_away, y)
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


def matrix_lambda_calc(lam, k):
    """Helper PMF calculation to keep code fast and dependency-free."""
    return (lam**k * math.exp(-lam)) / math.factorial(k)
