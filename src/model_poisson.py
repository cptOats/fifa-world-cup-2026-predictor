"""Vectorized Dixon-Coles Poisson Joint Maximum Likelihood Estimation Model."""

import json
import logging
import os

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, minimize
from sklearn.model_selection import TimeSeriesSplit


def _pure_poisson_neg_log_likelihood(
    params,
    home_indices,
    away_indices,
    home_scores,
    away_scores,
    weights,
    neutral_flags,
    N,
    global_neutral_avg,
):
    """Vectorized independent Poisson log-likelihood."""
    attacks = params[:N]
    defenses = params[N : 2 * N]
    gamma = params[2 * N]

    atk_h = attacks[home_indices]
    def_a = defenses[away_indices]
    atk_a = attacks[away_indices]
    def_h = defenses[home_indices]

    home_premium = np.where(neutral_flags == 0, gamma, 1.0)

    lam = np.maximum(1e-4, atk_h * def_a * global_neutral_avg * home_premium)
    mu = np.maximum(1e-4, atk_a * def_h * global_neutral_avg * (1.0 / home_premium))

    log_lik_home = -lam + home_scores * np.log(lam)
    log_lik_away = -mu + away_scores * np.log(mu)

    return -np.sum(weights * (log_lik_home + log_lik_away))


def _dixon_coles_neg_log_likelihood(
    params,
    home_indices,
    away_indices,
    home_scores,
    away_scores,
    weights,
    neutral_flags,
    N,
    global_neutral_avg,
):
    """Vectorized negative log-likelihood function with active Dixon-Coles low-score coupling."""
    attacks = params[:N]
    defenses = params[N : 2 * N]
    gamma = params[2 * N]
    rho = params[2 * N + 1]

    atk_h = attacks[home_indices]
    def_a = defenses[away_indices]
    atk_a = attacks[away_indices]
    def_h = defenses[home_indices]

    home_premium = np.where(neutral_flags == 0, gamma, 1.0)

    lam = np.maximum(1e-4, atk_h * def_a * global_neutral_avg * home_premium)
    mu = np.maximum(1e-4, atk_a * def_h * global_neutral_avg * (1.0 / home_premium))

    tau = np.ones_like(home_scores, dtype=float)

    mask_00 = (home_scores == 0) & (away_scores == 0)
    mask_10 = (home_scores == 1) & (away_scores == 0)
    mask_01 = (home_scores == 0) & (away_scores == 1)
    mask_11 = (home_scores == 1) & (away_scores == 1)

    tau[mask_00] = 1.0 - (rho * lam[mask_00] * mu[mask_00])
    tau[mask_10] = 1.0 + (rho * mu[mask_10])
    tau[mask_01] = 1.0 + (rho * lam[mask_01])
    tau[mask_11] = 1.0 - rho

    if np.any(tau <= 0):
        return 1e10

    log_lik_home = -lam + home_scores * np.log(lam)
    log_lik_away = -mu + away_scores * np.log(mu)

    return -np.sum(weights * (np.log(tau) + log_lik_home + log_lik_away))


def train_poisson_ratings(
    poisson_alpha=0.00047, dixon_coles=False
) -> tuple[dict[str, dict[str, float]], float, float, float]:
    """Fits team capabilities using a global Joint MLE framework with structural fast-paths."""
    ARTIFACTS_DIR = os.path.join("data", "artifacts")
    filename = (
        "poisson_artifacts_dixon_coles.json"
        if dixon_coles
        else "poisson_artifacts_pure.json"
    )
    POISSON_PATH = os.path.join(ARTIFACTS_DIR, filename)

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

    train_df["date"] = pd.to_datetime(train_df["date"])
    max_date = train_df["date"].max()
    days_elapsed = (max_date - train_df["date"]).dt.days
    train_df["match_weight"] = train_df["match_weight"] * np.exp(
        -poisson_alpha * days_elapsed
    )

    neutral_df = train_df[train_df["neutral"] == 1].copy()
    total_neutral_weight = neutral_df["match_weight"].sum()
    global_neutral_avg = (
        (
            (neutral_df["home_score"] * neutral_df["match_weight"]).sum()
            + (neutral_df["away_score"] * neutral_df["match_weight"]).sum()
        )
        / (2 * total_neutral_weight)
        if total_neutral_weight > 0
        else 1.35
    )

    teams = sorted(list(set(train_df["home_team"]).union(set(train_df["away_team"]))))
    N = len(teams)
    team_to_idx = {team: idx for idx, team in enumerate(teams)}

    home_indices = train_df["home_team"].map(team_to_idx).to_numpy()
    away_indices = train_df["away_team"].map(team_to_idx).to_numpy()
    home_scores = train_df["home_score"].to_numpy()
    away_scores = train_df["away_score"].to_numpy()
    weights = train_df["match_weight"].to_numpy()
    neutral_flags = train_df["neutral"].to_numpy()

    if dixon_coles:
        initial_guess = np.concatenate([np.ones(N), np.ones(N), [1.20], [0.0]])
        constraint_matrix = np.zeros((1, 2 * N + 2))
        constraint_matrix[0, N : 2 * N] = 1.0 / N
        bounds = Bounds(
            np.concatenate([np.repeat(0.05, N), np.repeat(0.05, N), [0.5], [-0.25]]),
            np.concatenate([np.repeat(15.0, N), np.repeat(15.0, N), [2.5], [0.25]]),
        )
        obj_func = _dixon_coles_neg_log_likelihood
        args = (
            home_indices,
            away_indices,
            home_scores,
            away_scores,
            weights,
            neutral_flags,
            N,
            global_neutral_avg,
        )
    else:
        # Fast-path bypasses parameter matrix: rho = zero
        initial_guess = np.concatenate([np.ones(N), np.ones(N), [1.20]])
        constraint_matrix = np.zeros((1, 2 * N + 1))
        constraint_matrix[0, N : 2 * N] = 1.0 / N
        bounds = Bounds(
            np.concatenate([np.repeat(0.05, N), np.repeat(0.05, N), [0.5]]),
            np.concatenate([np.repeat(15.0, N), np.repeat(15.0, N), [2.5]]),
        )
        obj_func = _pure_poisson_neg_log_likelihood
        args = (
            home_indices,
            away_indices,
            home_scores,
            away_scores,
            weights,
            neutral_flags,
            N,
            global_neutral_avg,
        )

    linear_constraint = LinearConstraint(constraint_matrix, lb=[1.0], ub=[1.0])

    res = minimize(
        obj_func,
        initial_guess,
        args=args,
        method="SLSQP",
        constraints=[linear_constraint],
        bounds=bounds,
        options={"maxiter": 1000, "ftol": 1e-5, "disp": False},
    )

    if not res.success:
        raise RuntimeError(
            f"Joint MLE Poisson Optimization failed to converge: {res.message}"
        )

    fit_attacks = res.x[:N]
    fit_defenses = res.x[N : 2 * N]
    fit_gamma = float(res.x[2 * N])
    fit_rho = float(res.x[2 * N + 1]) if dixon_coles else 0.0

    ratings = {}
    for team, idx in team_to_idx.items():
        ratings[team] = {
            "attack": max(0.1000, float(fit_attacks[idx])),
            "defense": max(0.2500, float(fit_defenses[idx])),
        }

    global_home_avg = global_neutral_avg * fit_gamma
    global_away_avg = global_neutral_avg / fit_gamma

    artifacts = {
        "global_home_avg": float(global_home_avg),
        "global_away_avg": float(global_away_avg),
        "global_neutral_avg": float(global_neutral_avg),
        "dixon_coles_rho": fit_rho,
        "ratings": ratings,
    }

    with open(POISSON_PATH, "w") as f:
        json.dump(artifacts, f, indent=4)

    return ratings, global_home_avg, global_away_avg, global_neutral_avg


def train_poisson_oof_predictions(
    feature_matrix, poisson_alpha=0.00047, dixon_coles=False
):
    """Calculates leak-proof out-of-fold Poisson predictions."""

    ARTIFACTS_DIR = os.path.join("data", "artifacts")
    suffix = "dixon_coles" if dixon_coles else "pure"
    POISSON_OOF_H_PATH = os.path.join(ARTIFACTS_DIR, f"poisson_oof_home_{suffix}.npy")
    POISSON_OOF_A_PATH = os.path.join(ARTIFACTS_DIR, f"poisson_oof_away_{suffix}.npy")

    # 1. Cache Hit Check
    if os.path.exists(POISSON_OOF_H_PATH) and os.path.exists(POISSON_OOF_A_PATH):
        logging.info(
            f"💾 Cached {suffix}-Poisson Out-of-Fold arrays detected. Skipping cross-validation loops..."
        )
        return np.load(POISSON_OOF_H_PATH), np.load(POISSON_OOF_A_PATH)

    # 2. Run Cross Validation
    n_matches = len(feature_matrix)
    oof_home_preds = np.zeros(n_matches)
    oof_away_preds = np.zeros(n_matches)
    tscv = TimeSeriesSplit(n_splits=3)

    for fold, (train_idx, test_idx) in enumerate(tscv.split(feature_matrix), 1):
        logging.info(
            f"🔄 Processing Poisson Out-of-Fold Cross-Validation (Fold {fold}/3)..."
        )

        train_df = feature_matrix.iloc[train_idx].copy()
        test_df = feature_matrix.iloc[test_idx]

        if "match_weight" not in train_df.columns:
            train_df["match_weight"] = 1.0

        train_df["match_date"] = pd.to_datetime(train_df["match_date"])
        fold_max_date = train_df["match_date"].max()
        days_elapsed = (fold_max_date - train_df["match_date"]).dt.days
        train_df["match_weight"] = train_df["match_weight"] * np.exp(
            -poisson_alpha * days_elapsed
        )

        neutral_df = train_df[train_df["is_neutral_venue"] == 1].copy()
        fold_neutral_avg = (
            (
                (neutral_df["home_score"] * neutral_df["match_weight"]).sum()
                + (neutral_df["away_score"] * neutral_df["match_weight"]).sum()
            )
            / (2 * neutral_df["match_weight"].sum())
            if not neutral_df.empty and neutral_df["match_weight"].sum() > 0
            else 1.35
        )

        teams = sorted(
            list(set(train_df["home_team"]).union(set(train_df["away_team"])))
        )
        N = len(teams)
        team_to_idx = {team: idx for idx, team in enumerate(teams)}

        home_indices = train_df["home_team"].map(team_to_idx).to_numpy()
        away_indices = train_df["away_team"].map(team_to_idx).to_numpy()
        home_scores = train_df["home_score"].to_numpy()
        away_scores = train_df["away_score"].to_numpy()
        weights = train_df["match_weight"].to_numpy()
        neutral_flags = train_df["is_neutral_venue"].to_numpy()

        if dixon_coles:
            initial_guess = np.concatenate([np.ones(N), np.ones(N), [1.20], [0.0]])
            constraint_matrix = np.zeros((1, 2 * N + 2))
            constraint_matrix[0, N : 2 * N] = 1.0 / N
            bounds = Bounds(
                np.concatenate(
                    [np.repeat(0.05, N), np.repeat(0.05, N), [0.5], [-0.25]]
                ),
                np.concatenate([np.repeat(15.0, N), np.repeat(15.0, N), [2.5], [0.25]]),
            )
            obj_func = _dixon_coles_neg_log_likelihood
            args = (
                home_indices,
                away_indices,
                home_scores,
                away_scores,
                weights,
                neutral_flags,
                N,
                fold_neutral_avg,
            )
        else:
            initial_guess = np.concatenate([np.ones(N), np.ones(N), [1.20]])
            constraint_matrix = np.zeros((1, 2 * N + 1))
            constraint_matrix[0, N : 2 * N] = 1.0 / N
            bounds = Bounds(
                np.concatenate([np.repeat(0.05, N), np.repeat(0.05, N), [0.5]]),
                np.concatenate([np.repeat(15.0, N), np.repeat(15.0, N), [2.5]]),
            )
            obj_func = _pure_poisson_neg_log_likelihood
            args = (
                home_indices,
                away_indices,
                home_scores,
                away_scores,
                weights,
                neutral_flags,
                N,
                fold_neutral_avg,
            )

        linear_constraint = LinearConstraint(constraint_matrix, lb=[1.0], ub=[1.0])

        res = minimize(
            obj_func,
            initial_guess,
            args=args,
            method="SLSQP",
            constraints=[linear_constraint],
            bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-5, "disp": False},
        )

        fold_attacks = res.x[:N]
        fold_defenses = res.x[N : 2 * N]
        fold_gamma = res.x[2 * N]

        for idx, row in test_df.iterrows():
            h_team = row["home_team"]
            a_team = row["away_team"]
            is_neutral = row.get("is_neutral_venue", 0)

            h_idx = team_to_idx.get(h_team, -1)
            a_idx = team_to_idx.get(a_team, -1)

            atk_h = fold_attacks[h_idx] if h_idx != -1 else 1.0
            def_h = fold_defenses[h_idx] if h_idx != -1 else 1.0
            atk_a = fold_attacks[a_idx] if a_idx != -1 else 1.0
            def_a = fold_defenses[a_idx] if a_idx != -1 else 1.0

            home_premium = fold_gamma if is_neutral == 0 else 1.0

            oof_home_preds[idx] = atk_h * def_a * fold_neutral_avg * home_premium
            oof_away_preds[idx] = (
                atk_a
                * def_h
                * fold_neutral_avg
                * (1.0 / home_premium if is_neutral == 0 else 1.0)
            )

    # 4. Save Artifacts & OOF Arrays
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    np.save(POISSON_OOF_H_PATH, oof_home_preds)
    np.save(POISSON_OOF_A_PATH, oof_away_preds)

    return oof_home_preds, oof_away_preds


def predict_poisson_match(
    home, away, venue_country, ratings, g_home, g_away, g_neutral
) -> tuple[float, float, float, float, float]:
    """Generates continuous baseline match targets matching model api requirements."""
    home_rating = ratings.get(home, {"attack": 1.0, "defense": 1.0})
    away_rating = ratings.get(away, {"attack": 1.0, "defense": 1.0})

    if home == venue_country:
        lambda_home = home_rating["attack"] * away_rating["defense"] * g_home
        lambda_away = away_rating["attack"] * home_rating["defense"] * g_away
        is_neutral = 0
    elif away == venue_country:
        lambda_home = home_rating["attack"] * away_rating["defense"] * g_away
        lambda_away = away_rating["attack"] * home_rating["defense"] * g_home
        is_neutral = 0
    else:
        lambda_home = home_rating["attack"] * away_rating["defense"] * g_neutral
        lambda_away = away_rating["attack"] * home_rating["defense"] * g_neutral
        is_neutral = 1

    raw_corners = (5.5 * home_rating["attack"] * away_rating["defense"]) + (
        5.5 * away_rating["attack"] * home_rating["defense"]
    )
    raw_yellows = (3.0 * home_rating["defense"] * away_rating["attack"]) + (
        3.0 * away_rating["defense"] * home_rating["attack"]
    )
    raw_reds = 0.12 if is_neutral == 0 else 0.10

    return lambda_home, lambda_away, raw_corners, raw_yellows, raw_reds
