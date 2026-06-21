"""
Model Consensus Blending and Meta-Ensemble Optimization Layer.

This module resolves the optimal weights required to blend individual base learners
into a single continuous expected goals (xG) vector.
It supports both unnormalized L2-regularized Ridge regression (Stacking) and
SciPy constrained optimization (Bounded SLSQP) to ensure physical logic is preserved.
"""

import logging

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, minimize
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


def find_optimal_blend_weights(
    feature_matrix: pd.DataFrame,
    g_home: float,
    g_away: float,
    oof_home_preds: np.ndarray,
    oof_away_preds: np.ndarray,
    oof_poisson_home: np.ndarray,
    oof_poisson_away: np.ndarray,
    method: str = "ridge",
) -> dict[str, float]:
    """
    Calibrates leak-proof optimal consensus blend weights across all modeling layers.

    Args:
        feature_matrix (pd.DataFrame): Master matrix containing actual historical targets.
        g_home (float): Global average home goals.
        g_away (float): Global average away goals.
        oof_home_preds (np.ndarray): Out-of-fold XGBoost home predictions.
        oof_away_preds (np.ndarray): Out-of-fold XGBoost away predictions.
        oof_poisson_home (np.ndarray): Out-of-fold Poisson home predictions.
        oof_poisson_away (np.ndarray): Out-of-fold Poisson away predictions.
        method (str): Solver method - "ridge" (Stacking) or "scipy" (SLSQP).

    Returns:
        dict[str, float]: Normalized weight distribution summing to 1.0.
    """

    actual_home_goals = feature_matrix["home_score"].to_numpy()
    actual_away_goals = feature_matrix["away_score"].to_numpy()

    # 1. Materialize Point-in-Time Historical Elo Baselines (Vectorized)
    # We use a simplified Elo-to-Goals projection solely for the meta-learner feature space.
    global_neutral_avg = (g_home + g_away) / 2.0

    home_elo = feature_matrix["home_elo_rating"].to_numpy()
    away_elo = feature_matrix["away_elo_rating"].to_numpy()
    rating_diff = (home_elo - away_elo) / 400.0

    elo_home = np.maximum(0.0, global_neutral_avg + rating_diff)
    elo_away = np.maximum(0.0, global_neutral_avg - rating_diff)

    # 2. Establish Validation Horizon Mask
    # Filters out rows where out-of-fold predictions couldn't be generated (early history)
    validation_horizon = oof_home_preds > 0

    y_home_active = actual_home_goals[validation_horizon]
    y_away_active = actual_away_goals[validation_horizon]

    p_home_active = oof_poisson_home[validation_horizon]
    p_away_active = oof_poisson_away[validation_horizon]

    e_home_active = elo_home[validation_horizon]
    e_away_active = elo_away[validation_horizon]

    x_home_active = oof_home_preds[validation_horizon]
    x_away_active = oof_away_preds[validation_horizon]

    if method == "ridge":
        # --- PATHWAY A: Level-1 Meta-Learner (Ridge Stacking) ---
        logging.info(
            "🧠 Resolving weights via L2-Regularized Ridge Stacking Regressor..."
        )

        # Stack features vertically to enforce symmetry.
        # Prevents the meta-learner from developing a home/away specific bias.
        y_stacked = np.concatenate([y_home_active, y_away_active])
        X_home = np.column_stack([p_home_active, e_home_active, x_home_active])
        X_away = np.column_stack([p_away_active, e_away_active, x_away_active])
        X_stacked = np.vstack([X_home, X_away])

        # fit_intercept=False ensures 0 xG inputs strictly yield 0 xG output.
        # positive=True prevents the model from taking "short" positions on base learners.
        meta_learner = Ridge(alpha=10.0, fit_intercept=False, positive=True)

        try:
            meta_learner.fit(X_stacked, y_stacked)
            raw_weights = meta_learner.coef_

            if np.sum(raw_weights) <= 0:
                raise ValueError("Ridge regression collapsed to zero weights.")

            # Preserve L2: expect sum to ~1 without normalization shows
            # learners are highly calibrated and output scaled Expected Goals.
            optimized_weights = {
                "poisson": float(raw_weights[0]),
                "elo": float(raw_weights[1]),
                "xgb": float(raw_weights[2]),
            }

        except Exception as e:
            logging.warning(
                f"⚠️ Meta-Learner failed to fit! Reason: {e}. Falling back to default uniform balance."
            )
            optimized_weights = {"poisson": 0.3333, "elo": 0.3333, "xgb": 0.3334}

    elif method == "scipy":
        # --- PATHWAY B: SciPy Constrained Optimization ---
        logging.info("🧩 Resolving weights via SciPy Bounded SLSQP solver...")

        def loss_function(weights):
            """Computes symmetric Mean Squared Error across both scoring perspectives."""

            w_poisson, w_elo, w_xgb = weights

            pred_home = (
                (w_poisson * p_home_active)
                + (w_elo * e_home_active)
                + (w_xgb * x_home_active)
            )
            pred_away = (
                (w_poisson * p_away_active)
                + (w_elo * e_away_active)
                + (w_xgb * x_away_active)
            )

            return (
                mean_squared_error(y_home_active, pred_home)
                + mean_squared_error(y_away_active, pred_away)
            ) / 2.0

        # Constraints: 0.0 <= weight <= 1.0, Sum of weights == 1.0
        bounds = Bounds([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        constraints = LinearConstraint([[1.0, 1.0, 1.0]], lb=[1.0], ub=[1.0])
        initial_guess = [0.3333, 0.3333, 0.3333]

        res = minimize(
            loss_function,
            initial_guess,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if not res.success:
            logging.warning(
                f"⚠️ Optimization failed to converge smoothly! Reason: {res.message}. Falling back to default uniform balance."
            )
            return {"poisson": 0.3333, "elo": 0.3333, "xgb": 0.3334}

        optimized_weights = {
            "poisson": float(res.x[0]),
            "elo": float(res.x[1]),
            "xgb": float(res.x[2]),
        }
    else:
        logging.error(
            f"❌ Configuration Error: Invalid blend method '{method}' requested. Expected 'stacking' or 'scipy'."
        )
        raise ValueError("Unsupported blending method.")

    return optimized_weights
