"""
Consensus Copula Calibration Module (Time-Decay & Context-Weighted).

Empirically determines the optimal Bivariate Copula correlation parameter (rho)
by mapping fully blended Out-of-Fold Consensus Expected Goals against
historical draw rates—rigorously adjusting for alpha time-decay and friendly weights.
"""

import glob
import json
import logging
import os

import numpy as np
import pandas as pd
from scipy.stats import norm, poisson

from src.features import compile_master_feature_matrix
from src.model_elo import EloEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)


def get_latest_run_metadata() -> tuple[dict[str, float], float]:
    """
    Extracts optimized ensemble weights and decay alpha from the most
    recent pipeline execution run.
    """
    run_dirs = sorted(
        glob.glob(os.path.join("data", "runs", "run_*")), key=os.path.getmtime
    )
    if not run_dirs:
        raise FileNotFoundError(
            "No run directories found in data/runs/. Please run main.py first."
        )

    latest_run = run_dirs[-1]
    metadata_path = os.path.join(latest_run, "metadata.json")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    weights = metadata.get(
        "ensemble_weights", {"poisson": 0.333, "elo": 0.333, "xgb": 0.334}
    )

    train_vars = metadata.get("training_variables", {})
    decay_alpha = train_vars.get("decay_alpha", 0.00047)

    logging.info(f"💾 Loaded Run Configurations from: {os.path.basename(latest_run)}")
    logging.info(
        f"⚖️  Weights: Poisson {weights['poisson']:.3f} | Elo {weights['elo']:.3f} | XGB {weights['xgb']:.3f}"
    )
    logging.info(f"📉 Active Time-Decay Alpha: {decay_alpha}")

    return weights, decay_alpha


def calibrate_weighted_copula(simulations_per_rho=100, strip_friendlies=False):
    """
    Sweeps a grid of rho values against the fully blended Consensus OOF vector
    applying context and temporal weighting architectures.
    """

    # 1. Fetch Global Poisson Averages for Elo Projections
    poisson_artifacts_path = os.path.join(
        "data", "artifacts", "poisson_artifacts_pure.json"
    )
    if not os.path.exists(poisson_artifacts_path):
        logging.error(
            "❌ Missing pure Poisson artifacts. Ensure main.py ran without Dixon-Coles."
        )
        return

    with open(poisson_artifacts_path, "r") as f:
        p_arts = json.load(f)
        global_neutral_avg = (
            p_arts["global_home_avg"] + p_arts["global_away_avg"]
        ) / 2.0

    # 2. Reconstruct the Master Feature Matrix for Strict Row Alignment
    parquet_path = os.path.join("data", "processed", "clean_historical_matches.parquet")
    df_raw = pd.read_parquet(parquet_path)

    logging.info("📈 Synchronizing sequence ratings through Elo Engine...")
    elo_engine = EloEngine(k_factor=40)
    elo_engine.fit(df_raw)

    feature_matrix, _ = compile_master_feature_matrix(parquet_path, elo_engine)

    # 3. Load Base Learner OOF Arrays
    oof_p_home = np.load(os.path.join("data", "artifacts", "poisson_oof_home_pure.npy"))
    oof_p_away = np.load(os.path.join("data", "artifacts", "poisson_oof_away_pure.npy"))
    oof_x_home = np.load(os.path.join("data", "artifacts", "xgb_oof_home.npy"))
    oof_x_away = np.load(os.path.join("data", "artifacts", "xgb_oof_away.npy"))

    # Vectorize Elo Point-in-Time Baseline Projections
    home_elo = feature_matrix["home_elo_rating"].to_numpy()
    away_elo = feature_matrix["away_elo_rating"].to_numpy()
    oof_e_home = np.maximum(0.0, global_neutral_avg + (home_elo - away_elo) / 400.0)
    oof_e_away = np.maximum(0.0, global_neutral_avg - (home_elo - away_elo) / 400.0)

    # 4. Construct the Blended Consensus OOF Vector
    weights, decay_alpha = get_latest_run_metadata()

    blend_h = (
        (weights["poisson"] * oof_p_home)
        + (weights["elo"] * oof_e_home)
        + (weights["xgb"] * oof_x_home)
    )
    blend_a = (
        (weights["poisson"] * oof_p_away)
        + (weights["elo"] * oof_e_away)
        + (weights["xgb"] * oof_x_away)
    )

    # 5. Build Dynamic Temporal and Contextual Evaluation Weights
    feature_matrix["match_date"] = pd.to_datetime(feature_matrix["match_date"])
    max_date = feature_matrix["match_date"].max()
    days_elapsed = (max_date - feature_matrix["match_date"]).dt.days

    # Calculate exact exponential alpha decay vector
    time_decay = np.exp(-decay_alpha * days_elapsed)
    final_weights = time_decay * feature_matrix["match_weight"].to_numpy()

    # Establish Horizon Mask (Filters early history initialization gaps)
    validation_horizon = oof_x_home > 0

    if strip_friendlies:
        # Optional hard gate logic to completely evict friendly rows from calibration
        logging.info(
            "✂️  Hard filtering active: Evicting all friendly matches from ledger entirely."
        )
        # Reconstruct tournament string matching using raw ledger merge indices if needed,
        # or leverage your native match_weight proxy down-weighting (Recommended)
        is_not_friendly = (feature_matrix["match_weight"] > 0.5).to_numpy()
        validation_horizon = validation_horizon & is_not_friendly

    # Apply final domain masks to arrays
    active_blend_h = blend_h[validation_horizon]
    active_blend_a = blend_a[validation_horizon]
    active_weights = final_weights[validation_horizon]

    actual_home = feature_matrix["home_score"].to_numpy()[validation_horizon]
    actual_away = feature_matrix["away_score"].to_numpy()[validation_horizon]

    # 6. Calculate True Empirical Weighted Draw Rate
    total_active_matches = len(actual_home)
    actual_draw_mask = actual_home == actual_away

    actual_draw_pct = (
        np.sum(actual_draw_mask * active_weights) / np.sum(active_weights)
    ) * 100

    logging.info(f"📊 Active Validation Ledger: {total_active_matches:,} matches.")
    logging.info(f"🎯 Empirical Time-Decay Weighted Draw Rate: {actual_draw_pct:.2f}%")

    # 7. Sweep the Copula Parameter Grid
    rho_grid = np.arange(0.00, 0.16, 0.01)
    results = []

    logging.info(
        "⚙️  Sweeping Bivariate Copula parameter space using mathematical observation weights..."
    )
    rng = np.random.default_rng(seed=1989)

    for rho in rho_grid:
        simulated_draw_rates = []

        for _ in range(simulations_per_rho):
            cov_matrix = [[1.0, rho], [rho, 1.0]]
            z = rng.multivariate_normal(
                [0.0, 0.0], cov_matrix, size=len(active_blend_h)
            )
            u = norm.cdf(z)

            sims_h = np.asarray(poisson.ppf(u[:, 0], active_blend_h), dtype=int)
            sims_a = np.asarray(poisson.ppf(u[:, 1], active_blend_a), dtype=int)

            sim_draw_mask = sims_h == sims_a
            # Calculate simulated rate using the exact same observation weight vector
            weighted_sim_draw_pct = (
                np.sum(sim_draw_mask * active_weights) / np.sum(active_weights)
            ) * 100
            simulated_draw_rates.append(weighted_sim_draw_pct)

        mean_sim_draw_pct = np.mean(simulated_draw_rates)
        error = abs(mean_sim_draw_pct - actual_draw_pct)

        results.append({"rho": rho, "sim_draw_pct": mean_sim_draw_pct, "error": error})

    # 8. Analyze and Report
    results_df = pd.DataFrame(results)
    optimal_row = results_df.loc[results_df["error"].idxmin()]
    optimal_rho = optimal_row["rho"]

    print("\n--- 📈 WEIGHTED CONSENSUS COPULA CALIBRATION RESULTS ---")
    print(f"Target Weighted Draw Rate: {actual_draw_pct:.2f}%\n")
    print(f"{'Rho (ρ)':<10} | {'Weighted Sim Draw %':<20} | {'Absolute Error':<10}")
    print("-" * 50)

    for _, row in results_df.iterrows():
        marker = " <--- OPTIMAL" if row["rho"] == optimal_rho else ""
        print(
            f"{row['rho']:<10.2f} | {row['sim_draw_pct']:<20.2f} | {row['error']:<10.3f}{marker}"
        )

    print("-" * 50)
    logging.info(
        f"✅ Calibration complete. The mathematically optimal draw_copula is: {optimal_rho:.2f}"
    )


if __name__ == "__main__":
    # Toggle strip_friendlies=True to erase friendlies, otherwise applies friendly weighting.
    calibrate_weighted_copula(simulations_per_rho=100, strip_friendlies=False)
