"""
FIFA World Cup 2026 Predictor: Pipeline Orchestrator.

This Directed Acyclic Graph executes the entire end-to-end predictive architecture.
It uses an Extract, Transform, Load framework: spins up machine learning models, blends
an ensemble, runs monte-carlo stochastic simulations, and caches artifacts and run data.
"""

import datetime
import json
import logging
import os
import shutil
from typing import TypedDict

import pandas as pd

from src.blender import find_optimal_blend_weights
from src.features import (
    compile_master_feature_matrix,
    extract_latest_team_form,
    test_point_in_time_leakage,
)
from src.ingest import verify_data_layer
from src.model_elo import EloEngine
from src.model_poisson import (
    train_poisson_oof_predictions,
    train_poisson_ratings,
)
from src.model_xgboost import train_production_xgboost_models
from src.router import (
    simulate_deterministic_group_stage,
    simulate_knockout_waterfall,
)
from src.stochastic import (
    build_expected_stochastic_bracket,
    precompute_sandbox_matchups,
    run_monte_carlo_master,
)
from src.transform import (
    DATACAMP_TO_KAGGLE,
    get_venue_country,
    prepare_historical_features,
)


class TrainingConfig(TypedDict):
    friendly_weight: float
    time_slice_start: str
    start_of_tournament: str
    decay_alpha: float
    cv_folds: int


class MatchRulesConfig(TypedDict):
    et_multiplier: float
    fatigue_factor: float
    card_boost_factor: float
    draw_copula: float


# =====================================================================
# MLOPS FLIGHT CONTROLS
# =====================================================================

# --- MODEL CONFIGURATION ---
MODEL_TYPE: str = "blend"  # "blend", "poisson", "elo", "xgb"
BLEND_METHOD: str = "ridge"  # "ridge", "scipy"
RUN_MONTE_CARLO: bool = True  # Enable for full predictive power
MONTE_CARLO_RUNS: int = 10000  # Recommend 10K+
FORCE_RETRAIN: bool = False  # Deletes model artifacts and OOF arrays

# --- TRAINING VARIABLES ---
TRAINING_VARIABLES: TrainingConfig = {
    "friendly_weight": 0.4,
    "time_slice_start": "1998-01-01",
    "start_of_tournament": "2026-06-11",
    "decay_alpha": 0.00047,
    "cv_folds": 10,
}

# --- TOURNAMENT RULES ---
MATCH_RULES: MatchRulesConfig = {
    "et_multiplier": 1.0 / 3.0,
    "fatigue_factor": 0.80,
    "card_boost_factor": 1.75,
    "draw_copula": 0.08,
}

# =====================================================================

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
blend_suffix = f"_{BLEND_METHOD}" if MODEL_TYPE == "blend" else ""
run_name = f"run_{MODEL_TYPE}{blend_suffix}_{timestamp}"
run_dir = os.path.join("data", "runs", run_name)
os.makedirs(run_dir, exist_ok=True)

# Centralize execution logging natively separating file and terminal streams
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(run_dir, "run_execution.log")),
        logging.StreamHandler(),  # Still prints INFO to terminal, but hides messy data
    ],
)


def main():
    """Executes the end-to-end World Cup prediction pipeline."""

    logging.info("✈️  Running pre-flight checks...")
    test_point_in_time_leakage()

    ARTIFACTS_DIR = os.path.join("data", "artifacts")

    # Cache Management Guard
    if FORCE_RETRAIN:
        logging.info(
            "🧹 FORCE_RETRAIN active. Evicting stale model caches and processed artifacts..."
        )
        if os.path.exists(ARTIFACTS_DIR):
            shutil.rmtree(ARTIFACTS_DIR)
        if os.path.exists(os.path.join("data", "processed")):
            shutil.rmtree(os.path.join("data", "processed"))

    verify_data_layer()
    logging.info("⚙️  Running entity validation and preparing historical features...")
    saved_path = prepare_historical_features(DATACAMP_TO_KAGGLE, TRAINING_VARIABLES)

    modern_df = pd.read_parquet(saved_path)
    group_fixtures = pd.read_csv(os.path.join("data", "raw", "group_fixtures.csv"))

    group_fixtures["home_team"] = group_fixtures["home_team"].replace(
        DATACAMP_TO_KAGGLE
    )
    group_fixtures["away_team"] = group_fixtures["away_team"].replace(
        DATACAMP_TO_KAGGLE
    )

    group_fixtures["venue_country"] = group_fixtures["venue"].apply(get_venue_country)

    logging.info(
        f"🚀 Launching World Cup Prediction Pipeline [Engine: {MODEL_TYPE}{blend_suffix}]"
    )

    # --- MODEL: Continuous Dynamic Elo ---
    logging.info(
        "📈 Synchronizing continuous World Football Elo ratings through time..."
    )
    elo_engine = EloEngine(k_factor=40)
    elo_engine.fit(modern_df)

    feature_matrix, feature_columns = compile_master_feature_matrix(
        os.path.join("data", "processed", "clean_historical_matches.parquet"),
        elo_engine,
    )

    # --- MODEL: Poisson Joint MLE (Pure & Dixon-Coles) ---
    logging.info(
        f"🧮 Resolving {'dixon_coles' if (MODEL_TYPE == 'poisson') else 'pure'}-Poisson Joint Maximum Likelihood Estimations..."
    )
    ratings, g_home, g_away, g_neutral = train_poisson_ratings(
        poisson_alpha=(TRAINING_VARIABLES["decay_alpha"]),
        dixon_coles=(MODEL_TYPE == "poisson"),
    )
    oof_poisson_home, oof_poisson_away = train_poisson_oof_predictions(
        feature_matrix,
        poisson_alpha=TRAINING_VARIABLES["decay_alpha"],
        dixon_coles=(MODEL_TYPE == "poisson"),
        cv_folds=TRAINING_VARIABLES["cv_folds"],
    )

    # --- MODEL: XGBoost Iterative Trees ---
    logging.info("🌲 Training dynamic XGBoost count model...")
    xgb_home, xgb_away, oof_home_preds, oof_away_preds, cv_metrics = (
        train_production_xgboost_models(
            feature_matrix,
            feature_columns,
            alpha=TRAINING_VARIABLES["decay_alpha"],
            cv_folds=TRAINING_VARIABLES["cv_folds"],
        )
    )

    # --- META-ENSEMBLE OPTIMIZATION ---
    if MODEL_TYPE == "blend":
        blend_weights = find_optimal_blend_weights(
            feature_matrix=feature_matrix,
            g_home=g_home,
            g_away=g_away,
            oof_home_preds=oof_home_preds,
            oof_away_preds=oof_away_preds,
            oof_poisson_home=oof_poisson_home,
            oof_poisson_away=oof_poisson_away,
            method=BLEND_METHOD,
        )
    elif MODEL_TYPE == "poisson":
        blend_weights = {"poisson": 1.0, "elo": 0.0, "xgb": 0.0}
    elif MODEL_TYPE == "elo":
        blend_weights = {"poisson": 0.0, "elo": 1.0, "xgb": 0.0}
    elif MODEL_TYPE == "xgb":
        blend_weights = {"poisson": 0.0, "elo": 0.0, "xgb": 1.0}
    else:
        raise ValueError(
            "Unsupported MODEL_TYPE. Choose from 'blend', 'poisson', 'elo', 'xgb'."
        )

    logging.info(
        f"⚖️  Active Execution Weights: Poisson {blend_weights['poisson']:.3f} | Elo {blend_weights['elo']:.3f} | XGBoost {blend_weights['xgb']:.3f}"
    )

    logging.info("📊 Extracting final pre-tournament team form states...")
    raw_teams = set(group_fixtures["home_team"].unique()) | set(
        group_fixtures["away_team"].unique()
    )
    participating_teams: list[str] = [str(team) for team in raw_teams]
    latest_team_form = extract_latest_team_form(feature_matrix, participating_teams)

    # --- GROUP STAGE ROUTING ---
    predicted_fixtures, group_tables, third_place_assignments = (
        simulate_deterministic_group_stage(
            group_fixtures=group_fixtures,
            ratings=ratings,
            g_home=g_home,
            g_away=g_away,
            g_neutral=g_neutral,
            blend_weights=blend_weights,
            match_rules=MATCH_RULES,
            elo_engine=elo_engine,
            xgb_home=xgb_home,
            xgb_away=xgb_away,
            feature_columns=feature_columns,
            latest_team_form=latest_team_form,
        )
    )

    # --- KNOCKOUT WATERFALL ROUTING ---
    logging.info(
        "🌿 Advancing teams and resolving dynamic knockout bracket tree mappings..."
    )
    knockout_matrix = simulate_knockout_waterfall(
        group_tables_df=group_tables,
        third_place_mapping=third_place_assignments,
        ratings=ratings,
        g_home_avg=g_home,
        g_away_avg=g_away,
        g_neutral_avg=g_neutral,
        blend_weights=blend_weights,
        match_rules=MATCH_RULES,
        elo_engine=elo_engine,
        xgb_home=xgb_home,
        xgb_away=xgb_away,
        feature_columns=feature_columns,
        latest_team_form=latest_team_form,
    )

    # Filter and export execution matrices to disk
    master_cols = [
        "match_id",
        "round",
        "venue",
        "venue_country",
        "home_team",
        "away_team",
        "predicted_home_goals",
        "predicted_away_goals",
        "corners",
        "yellow_cards",
        "red_cards",
        "extra_time",
        "penalties",
        "winner_name_meta",
    ]
    master_tournament = pd.concat(
        [predicted_fixtures, knockout_matrix], ignore_index=True
    )[master_cols]

    master_tournament.to_csv(
        os.path.join(run_dir, "deterministic_tournament.csv"), index=False
    )
    group_tables.to_csv(
        os.path.join(run_dir, "deterministic_group_tables.csv"), index=False
    )
    third_places_df = group_tables[group_tables["position"] == 3].copy()
    ranked_thirds_df = third_places_df.sort_values(
        by=["points", "goals_diff", "goals_for"], ascending=[False, False, False]
    ).reset_index(drop=True)
    ranked_thirds_df.to_csv(
        os.path.join(run_dir, "deterministic_third_places.csv"), index=False
    )

    # UI Master Lookup Table compilation
    capability_records = []
    for team in participating_teams:
        p_rat = ratings.get(team, {"attack": 1.0, "defense": 1.0})
        form = latest_team_form.get(team, {})
        elo = elo_engine.get_rating(team)
        short_atk = form.get("ewm_adj_gf_5", 1.2)
        short_def = form.get("ewm_adj_ga_5", 1.2)
        long_atk = form.get("ewm_adj_gf_15", 1.2)
        long_def = form.get("ewm_adj_ga_15", 1.2)
        short_vol_atk = form.get("cv_adj_gf_5", 0.0)
        short_vol_def = form.get("cv_adj_ga_5", 0.0)
        long_vol_atk = form.get("cv_adj_gf_15", 0.0)
        long_vol_def = form.get("cv_adj_ga_15", 0.0)
        record = {
            "Country": team,
            "Elo_Rating": elo,
            "Elo_Momentum": form.get("elo_momentum_5", 0.0),
            "Poisson_Attack": p_rat["attack"],
            "Poisson_Defense": p_rat["defense"],
            "Short_Term_Attack": short_atk,
            "Short_Term_Defense": short_def,
            "Long_Term_Attack": long_atk,
            "Long_Term_Defense": long_def,
            "Attack_Volatility_Short": short_vol_atk,
            "Defense_Volatility_Short": short_vol_def,
            "Attack_Volatility_Long": long_vol_atk,
            "Defense_Volatility_Long": long_vol_def,
        }

        capability_records.append(record)

    capabilities_df = pd.DataFrame(capability_records).sort_values(
        by="Elo_Rating", ascending=False
    )
    capabilities_df.to_csv(
        os.path.join(run_dir, "pre_tournament_capabilities.csv"), index=False
    )

    # Construct and Save the Metadata JSON
    run_metadata = {
        "run_id": run_name,
        "timestamp": timestamp,
        "config": {
            "model_type": MODEL_TYPE,
            "blend_method": BLEND_METHOD if MODEL_TYPE == "blend" else None,
            "run_monte_carlo": RUN_MONTE_CARLO,
            "monte_carlo_runs": MONTE_CARLO_RUNS if RUN_MONTE_CARLO else None,
        },
        "training_variables": TRAINING_VARIABLES,
        "match_rules": MATCH_RULES,
        "ensemble_weights": blend_weights,
        "cross_validation_metrics": cv_metrics,
    }

    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump(run_metadata, f, indent=4)

    final_match = knockout_matrix[knockout_matrix["round"] == "Final"].iloc[0]
    logging.info(
        f"🏆 Deterministic path champion detected: {final_match['winner_name_meta'].upper()}"
    )

    # --- MONTE CARLO STOCHASTIC BLOCK ---
    if RUN_MONTE_CARLO:
        logging.info(
            f"🎲 Spawning {MONTE_CARLO_RUNS:,} Monte Carlo parallel universes..."
        )
        raw_knockout_template = pd.read_csv(
            os.path.join("data", "raw", "knockout_slots.csv")
        )
        raw_knockout_template["venue_country"] = raw_knockout_template["venue"].apply(
            get_venue_country
        )

        prob_dashboard, df_xtables = run_monte_carlo_master(
            group_fixtures=group_fixtures,
            raw_knockout_template=raw_knockout_template,
            ratings=ratings,
            g_home=g_home,
            g_away=g_away,
            g_neutral=g_neutral,
            elo_engine=elo_engine,
            xgb_home=xgb_home,
            xgb_away=xgb_away,
            feature_columns=feature_columns,
            latest_team_form=latest_team_form,
            blend_weights=blend_weights,
            match_rules=MATCH_RULES,
            n_simulations=MONTE_CARLO_RUNS,
        )

        stochastic_champion_row = prob_dashboard.loc[
            prob_dashboard["Champion %"].idxmax()
        ]
        logging.info(
            f"🔮 Stochastic path champion detected: {stochastic_champion_row['Country'].upper()} {stochastic_champion_row['Champion %']:.1f}%"
        )

        prob_dashboard.to_csv(
            os.path.join(run_dir, "stochastic_forecast.csv"), index=False
        )
        df_xtables.to_csv(
            os.path.join(run_dir, "stochastic_group_tables.csv"), index=False
        )

        # Compute and Save Sandbox Pairwise Matrix
        logging.info("⚔️  Spinning up Sandbox pairwise matrix computations...")
        df_sandbox = precompute_sandbox_matchups(
            all_teams=participating_teams,
            ratings=ratings,
            g_home=g_home,
            g_away=g_away,
            g_neutral=g_neutral,
            blend_weights=blend_weights,
            match_rules=MATCH_RULES,
            elo_engine=elo_engine,
            xgb_home=xgb_home,
            xgb_away=xgb_away,
            feature_columns=feature_columns,
            latest_team_form=latest_team_form,
            fat_runs=MONTE_CARLO_RUNS,
        )
        df_sandbox.to_csv(
            os.path.join(run_dir, "stochastic_sandbox_matchups.csv"), index=False
        )

        # Build the Expected Probabilistic Tournament Bracket
        logging.info(
            "🌳 Constructing expected stochastic tournament bracket via xTables..."
        )
        df_stoch_bracket = build_expected_stochastic_bracket(
            df_xtables=df_xtables,
            df_sandbox=df_sandbox,
            raw_knockout_template=raw_knockout_template,
            group_fixtures=group_fixtures,
        )
        df_stoch_bracket.to_csv(
            os.path.join(run_dir, "stochastic_tournament.csv"), index=False
        )

    else:
        logging.info(
            "🎲 Monte Carlo Engine: [DISABLED] skipping stochastic simulation."
        )

    logging.info("🏁 Core pipeline orchestration completed successfully.")


if __name__ == "__main__":
    main()
