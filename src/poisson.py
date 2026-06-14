"""Poisson Statistical Modeling and Match Prediction Engine.

This module provides the core probabilistic execution layers for the tournament
prediction pipeline. It calculates weighted historical attack and defense strengths
for international football teams, generates leak-proof out-of-fold (OOF) baseline
expectations via chronological cross-validation, and provides a low-scoring
Dixon-Coles coupling adjustment layer to predict integer match scores.
"""

import json
import math
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "poisson_artifacts.json")


def train_poisson_ratings():
    """Calculates weighted historical attack and defense strengths for all teams.

    Checks the local directory for a pre-compiled JSON cache artifact. If an archive
    is missing, it reads the modern processed Parquet match logs, weights outcomes
    by fixture importance context, maps aggregated goals against global baselines,
    and saves the calculated parameters to disk to avoid retraining.

    Returns:
        tuple: A parsed combination containing:
            - ratings (dict): Team names mapping to nested 'attack' and 'defense' coefficients.
            - global_home_avg (float): Global weighted average for home goals scored.
            - global_away_avg (float): Global weighted average for away goals scored.
    """
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "r") as f:
            artifacts = json.load(f)
        return (
            artifacts["ratings"],
            artifacts["global_home_avg"],
            artifacts["global_away_avg"],
        )

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

    return ratings, global_home_avg, global_away_avg


def train_poisson_oof_predictions(feature_matrix):
    r"""Calculates leak-proof, continuous out-of-fold Poisson predictions via cross-validation.

    Uses a scikit-learn `TimeSeriesSplit` to slice the master feature matrix along
    chronological lines. For each fold, it fits isolated training-set Poisson attack/defense
    coefficients and scores the unseen testing horizon, outputting unbiased baseline numbers
    specifically tailored for downstream meta-blender optimization.

    Args:
        feature_matrix (pd.DataFrame): Master feature matrix tracking historical matches,
            containing `home_team`, `away_team`, `home_score`, and `away_score` data.

    Returns:
        tuple: A combination containing:
            - oof_home_preds (np.ndarray): Array of float expectations ($\lambda$) for the
                home-side goals generated when records were out-of-fold.
            - oof_away_preds (np.ndarray): Array of float expectations ($\lambda$) for the
                away-side goals generated when records were out-of-fold.
    """
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


def predict_poisson_match(home, away, venue_country, ratings, g_home_avg, g_away_avg):
    """Generates expected goals, total corners, and card counts using pure Poisson parameters.

    Evaluates relative attack and defense capabilities adjusted for host-nation stadium
    proximity, feeding rate parameters into the Dixon-Coles discrete matrix solver.

    Args:
        home (str): Name string identifying the designated home team entity.
        away (str): Name string identifying the designated away team entity.
        venue_country (str): Cleaned host nation country string ("Mexico", "Canada", "United States").
        ratings (dict): Dictionary map of attack and defense team capability coefficients.
        g_home_avg (float): Global dataset baseline for home goals scored.
        g_away_avg (float): Global dataset baseline for away goals scored.

    Returns:
        tuple: Mode scores, corners, cards, and outcome win classification labels.
    """
    home_rating = ratings.get(home, {"attack": 1.0, "defense": 1.0})
    away_rating = ratings.get(away, {"attack": 1.0, "defense": 1.0})

    g_neutral = (g_home_avg + g_away_avg) / 2.0

    # GOAL CALCULATIONS
    if home == venue_country:
        lambda_home = home_rating["attack"] * away_rating["defense"] * g_home_avg
        lambda_away = away_rating["attack"] * home_rating["defense"] * g_away_avg
    elif away == venue_country:
        lambda_home = home_rating["attack"] * away_rating["defense"] * g_away_avg
        lambda_away = away_rating["attack"] * home_rating["defense"] * g_home_avg
    else:
        lambda_home = home_rating["attack"] * away_rating["defense"] * g_neutral
        lambda_away = away_rating["attack"] * home_rating["defense"] * g_neutral

    pred_home_score, pred_away_score = get_dixon_coles_score(lambda_home, lambda_away)

    # PROXY METRICS
    home_corners = 5.5 * home_rating["attack"] * away_rating["defense"]
    away_corners = 5.5 * away_rating["attack"] * home_rating["defense"]
    total_corners = int(np.clip(np.round(home_corners + away_corners), 5, 16))
    home_cards = 3.0 * home_rating["defense"] * away_rating["attack"]
    away_cards = 3.0 * away_rating["defense"] * home_rating["attack"]
    total_yellows = int(np.clip(np.round(home_cards + away_cards), 1, 9))
    total_reds = 0

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


def get_dixon_coles_score(lambda_home, lambda_away, rho=-0.10):
    r"""Evaluates a joint probability grid up to a 5-5 scoreline with low-score coupling.

    Constructs a 6x6 matrix of independent Poisson goal probabilities, applying a
    Dixon-Coles $\tau$ parameter adjustment layer to fine-tune low-scoring joint states.

    Args:
        lambda_home (float): Continuous goal intensity parameter expectation for the home side.
        lambda_away (float): Continuous goal intensity parameter expectation for the away side.
        rho (float, optional): Dependence factor parameter optimizing low-score inflation.

    Returns:
        tuple: Calculated discrete goal selection coordinate pair counts (pred_home, pred_away).
    """
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


def _matrix_lambda_calc(lam, k):
    r"""Evaluates the standard Poisson Probability Mass Function (PMF).

    Runs a mathematical helper calculation to return discrete probability scores.

    Args:
        lam (float): Continuous distribution mean scale rate parameter ($\lambda$).
        k (int): Discrete feature frequency target count ($k$).

    Returns:
        float: Calculated probability value corresponding to exactly $k$ occurrences.
    """
    return (lam**k * math.exp(-lam)) / math.factorial(k)
