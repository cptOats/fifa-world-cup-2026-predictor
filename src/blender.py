import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.models import predict_match_score


def find_optimal_blend_weights(
    feature_matrix,
    ratings,
    g_home_avg,
    g_away_avg,
    elo_engine,
    xgb_home,
    xgb_away,
    feature_columns,
):
    """
    Evaluates all three models across the historical feature matrix and uses
    constrained mathematical optimization to find the perfect blend weights.
    """
    print("\n🧮 Initializing Phase 3 Mathematical Optimization Blending...")

    # To keep optimization execution incredibly fast and relevant to modern dynamics,
    # we train the weights on the last 1,500 matches in our leakage-protected matrix
    eval_sample = feature_matrix.tail(1500).copy().reset_index(drop=True)

    actual_home = eval_sample["home_score"].values
    actual_away = eval_sample["away_score"].values

    N = len(eval_sample)

    # 1. Compile Historical Prediction Arrays across all three models
    poisson_home, poisson_away = np.zeros(N), np.zeros(N)
    elo_home, elo_away = np.zeros(N), np.zeros(N)

    print("   ↳ Extracting historical baseline prediction matrices...")
    for idx, row in eval_sample.iterrows():
        h_team = row["home_team"]
        a_team = row["away_team"]

        # Pure Poisson Goal Expectancy Snapshots (unrounded)
        h_rat = ratings.get(h_team, {"attack": 1.0, "defense": 1.0})
        a_rat = ratings.get(a_team, {"attack": 1.0, "defense": 1.0})
        g_neutral = (g_home_avg + g_away_avg) / 2.0
        poisson_home[idx] = h_rat["attack"] * a_rat["defense"] * g_neutral
        poisson_away[idx] = a_rat["attack"] * h_rat["defense"] * g_neutral

        # Pure Elo Goal Expectancy Snapshots (unrounded)
        elo_meta = elo_engine.predict_match(h_team, a_team)
        elo_home[idx] = elo_meta["predicted_home_goals"]
        elo_away[idx] = elo_meta["predicted_away_goals"]

    # XGBoost Continuous Predictions
    X = eval_sample[feature_columns]
    xgb_home_preds = xgb_home.predict(X)
    xgb_away_preds = xgb_away.predict(X)

    # 2. Define the Optimization Loss Function (Mean Squared Error across both sides)
    def objective_function(weights):
        w_poisson, w_elo, w_xgb = weights

        blended_home = (
            (w_poisson * poisson_home) + (w_elo * elo_home) + (w_xgb * xgb_home_preds)
        )
        blended_away = (
            (w_poisson * poisson_away) + (w_elo * elo_away) + (w_xgb * xgb_away_preds)
        )

        mse_home = np.mean((actual_home - blended_home) ** 2)
        mse_away = np.mean((actual_away - blended_away) ** 2)

        return (mse_home + mse_away) / 2.0

    # 3. Enforce Physical Constraints: Weights must sum to 1.0 and stay bounded between 0 and 1
    constraints = {"type": "eq", "fun": lambda w: 1.0 - np.sum(w)}
    bounds = ((0, 1), (0, 1), (0, 1))

    # Flat start initial guess (Equal split distribution)
    initial_weights = [0.333, 0.333, 0.334]

    # 4. Run Sequential Least Squares Programming (SLSQP) Optimizer
    print("   ↳ Executing SLSQP Constrained Minimization solver...")
    result = minimize(
        objective_function,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    opt_poisson, opt_elo, opt_xgb = result.x

    return {"poisson": opt_poisson, "elo": opt_elo, "xgboost": opt_xgb}
