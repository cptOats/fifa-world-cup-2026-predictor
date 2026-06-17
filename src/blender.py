"""Model Consensus Blending and Meta-Ensemble Optimization Layer.

This module provides the optimization framework for combining distinct modeling
layers (Poisson Ratings, Elo Engine, and Gradient-Boosted Trees). It leverages
out-of-fold (OOF) cross-validation predictions to configure a joint Mean Squared
Error (MSE) loss function, solving for optimal consensus weights using a bounded
and constrained SciPy optimization routine.
"""

import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error


def find_optimal_blend_weights(
    feature_matrix: pd.DataFrame,
    g_home: float,
    g_away: float,
    oof_home_preds: np.ndarray,
    oof_away_preds: np.ndarray,
    oof_poisson_home: np.ndarray,
    oof_poisson_away: np.ndarray,
) -> dict[str, float]:
    r"""Calibrates leak-proof optimal consensus blend weights across all modeling layers.

    Extracts active cross-validation horizons to shield the ensemble layer from
    data leakage. The function constructs a combined baseline for historical point-in-time
    Elo predictions, defines a joint loss function optimizing over collective home and
    away goal MSE, and resolves constraints where individual model weights must fall within
    the boundaries $[0.0, 1.0]$ and strictly sum to $1.0$ ($\sum w_i = 1.0$). If the
    optimization fails to converge, it gracefully defaults to a uniform distribution.

    Args:
        feature_matrix (pd.DataFrame): Master feature matrix tracking historical matches,
            containing rating definitions and score targets.
        g_home (float): Dataset global average score metric for home-side goal references.
        g_away (float): Dataset global average score metric for away-side goal references.
        oof_home_preds (np.ndarray): Continuous out-of-fold validation array for XGBoost home goals.
        oof_away_preds (np.ndarray): Continuous out-of-fold validation array for XGBoost away goals.
        oof_poisson_home (np.ndarray): Continuous out-of-fold validation array for Poisson home goals.
        oof_poisson_away (np.ndarray): Continuous out-of-fold validation array for Poisson away goals.

    Returns:
        dict[str, float]: Consensus weight mapping dictionary containing the keys
            'poisson', 'elo', and 'xgb' paired with their optimal fractional
            coefficients.
    """
    actual_home_goals = feature_matrix["home_score"].to_numpy()
    actual_away_goals = feature_matrix["away_score"].to_numpy()

    # 1. Materialize Point-in-Time Historical Elo Baselines (Vectorized)
    global_neutral_avg = (g_home + g_away) / 2.0

    home_elo = feature_matrix["home_elo_rating"].to_numpy()
    away_elo = feature_matrix["away_elo_rating"].to_numpy()
    rating_diff = (home_elo - away_elo) / 400.0

    elo_home = np.maximum(0.0, global_neutral_avg + rating_diff)
    elo_away = np.maximum(0.0, global_neutral_avg - rating_diff)

    # 2. Establish Validation Horizon Mask
    validation_horizon = oof_home_preds > 0

    y_home_active = actual_home_goals[validation_horizon]
    y_away_active = actual_away_goals[validation_horizon]

    p_home_active = oof_poisson_home[validation_horizon]
    p_away_active = oof_poisson_away[validation_horizon]

    e_home_active = elo_home[validation_horizon]
    e_away_active = elo_away[validation_horizon]

    x_home_active = oof_home_preds[validation_horizon]
    x_away_active = oof_away_preds[validation_horizon]

    # 3. Define Clean Loss Function
    def loss_function(weights):
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

    # 4. Enforce Optimization Bounds and Constraints
    from scipy.optimize import Bounds, LinearConstraint

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

    return optimized_weights
