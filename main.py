"""World Cup Prediction Pipeline Orchestration Script."""

import datetime
import json
import logging
import os
import shutil

import numpy as np
import pandas as pd

from src.blender import find_optimal_blend_weights
from src.features import compile_master_feature_matrix
from src.ingest import verify_data_layer
from src.match_engine import evaluate_match_consensus
from src.model_elo import EloEngine
from src.model_poisson import (
    train_poisson_oof_predictions,
    train_poisson_ratings,
)
from src.model_xgboost import train_production_xgboost_models
from src.router import (
    allocate_third_places,
    extract_best_third_places,
    resolve_group_tables,
    simulate_knockout_waterfall,
)
from src.stochastic import run_monte_carlo_master
from src.transform import (
    DATACAMP_TO_KAGGLE,
    get_venue_country,
    prepare_historical_features,
)

# --- MODEL CONFIGURATION ---
MODEL_TYPE = "blend"  # "blend", "poisson", "elo", "xgb"
RUN_MONTE_CARLO = True
MONTE_CARLO_RUNS = 10000  # Recommend 10K+
USE_PRIOR_NUDGE = True
NUDGE_STRENGTH = 1.5  # Recommend ~1.5
FORCE_RETRAIN = True

# --- TRAINING VARIABLES ---
TRAINING_VARIABLES = {
    "friendly_weight": 0.4,
    "time_slice_start": "1998-01-01",
    "start_of_tournament": "2026-06-11",
}

# --- TOURNAMENT RULES ---
MATCH_RULES = {
    "et_multiplier": 1.0 / 3.0,
    "fatigue_factor": 0.80,
    "card_boost_factor": 1.75,
}

# --- POWER RATINGS --- source: https://www.datacamp.com/datalab/w/3da1cc64-5670-441e-8e7b-b948a6a29403
TEAM_POWER = {
    "Algeria": 74,
    "Argentina": 95,
    "Australia": 74,
    "Austria": 79,
    "Belgium": 86,
    "Bosnia and Herzegovina": 72,
    "Brazil": 94,
    "Cape Verde": 64,
    "Canada": 75,
    "Colombia": 84,
    "DR Congo": 69,
    "Croatia": 83,
    "Curaçao": 61,
    "Czech Republic": 73,
    "Ivory Coast": 77,
    "Ecuador": 79,
    "Egypt": 76,
    "England": 93,
    "France": 97,
    "Germany": 90,
    "Ghana": 73,
    "Haiti": 62,
    "Iran": 74,
    "Iraq": 69,
    "Japan": 81,
    "Jordan": 65,
    "Mexico": 79,
    "Morocco": 82,
    "Netherlands": 88,
    "New Zealand": 64,
    "Norway": 81,
    "Panama": 67,
    "Paraguay": 76,
    "Portugal": 91,
    "Qatar": 68,
    "Saudi Arabia": 70,
    "Scotland": 73,
    "Senegal": 80,
    "South Africa": 70,
    "South Korea": 77,
    "Spain": 97,
    "Sweden": 78,
    "Switzerland": 82,
    "Tunisia": 71,
    "Turkey": 78,
    "United States": 80,
    "Uruguay": 84,
    "Uzbekistan": 68,
}

# --- PIPELINE INITIALIZATION & LOGGING ---
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
nudge_suffix = f"_nudge{NUDGE_STRENGTH}" if USE_PRIOR_NUDGE else ""
run_name = f"run_{MODEL_TYPE}{nudge_suffix}_{timestamp}"
run_dir = os.path.join("data", "runs", run_name)
os.makedirs(run_dir, exist_ok=True)

# Set up silent terminal logging but verbose file logging
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

    logging.info(
        f"🚀 Launching World Cup Prediction Pipeline [Engine: {MODEL_TYPE}{nudge_suffix}]"
    )

    # --- CACHE MANAGEMENT LAYER ---
    if FORCE_RETRAIN:
        logging.info(
            "🧹 FORCE_RETRAIN active. Evicting stale model caches and processed artifacts..."
        )
        ARTIFACTS_DIR = os.path.join("data", "artifacts")
        if os.path.exists(ARTIFACTS_DIR):
            shutil.rmtree(ARTIFACTS_DIR)
        if os.path.exists(os.path.join("data", "processed")):
            shutil.rmtree(os.path.join("data", "processed"))

    # --- INFRASTRUCTURE GATES ---
    verify_data_layer()
    logging.info("🔄 Running entity validation and preparing historical features...")
    saved_path = prepare_historical_features(DATACAMP_TO_KAGGLE, TRAINING_VARIABLES)

    # --- EXPLICIT DATA INGESTION ---
    modern_df = pd.read_parquet(saved_path)
    group_fixtures = pd.read_csv(os.path.join("data", "raw", "group_fixtures.csv"))

    group_fixtures["home_team"] = group_fixtures["home_team"].replace(
        DATACAMP_TO_KAGGLE
    )
    group_fixtures["away_team"] = group_fixtures["away_team"].replace(
        DATACAMP_TO_KAGGLE
    )

    # Pre-compute venue countries and neutrality flags for Group Stage instantly
    group_fixtures["venue_country"] = group_fixtures["venue"].apply(get_venue_country)
    group_fixtures["is_neutral"] = np.where(
        (group_fixtures["home_team"] == group_fixtures["venue_country"])
        | (group_fixtures["away_team"] == group_fixtures["venue_country"]),
        0,
        1,
    )

    raw_teams = set(group_fixtures["home_team"].unique()) | set(
        group_fixtures["away_team"].unique()
    )
    participating_teams: list[str] = [str(team) for team in raw_teams]

    # --- ENTITY ALIGNMENT GATE FOR POWER RATINGS PRIORS ---
    if USE_PRIOR_NUDGE and TEAM_POWER:
        missing_priors = [
            team for team in participating_teams if team not in TEAM_POWER
        ]
        assert not missing_priors, (
            f"❌ TEAM_POWER String Mismatch! Unmapped tournament teams: {missing_priors}"
        )
        logging.info("🎯 Bayesian Prior Pass: All tournament entities validated.")

    # --- ESTIMATOR TRAINING PLUGINS ---
    logging.info(
        "🧮 Resolving Maximum Likelihood Estimations for Poisson coefficients..."
    )
    ratings, g_home, g_away, g_neutral = train_poisson_ratings()

    logging.info(
        "📈 Synchronizing continuous World Football Elo ratings across time series..."
    )
    elo_engine = EloEngine(k_factor=40)
    elo_engine.fit(modern_df)

    # --- MACHINE LEARNING ENGINE PIPELINE LAYER ---
    logging.info("🌲 Training dynamic XGBoost count models...")
    feature_matrix, feature_columns = compile_master_feature_matrix(
        os.path.join("data", "processed", "clean_historical_matches.parquet"),
        elo_engine,
    )

    xgb_home, xgb_away, oof_home_preds, oof_away_preds, cv_metrics = (
        train_production_xgboost_models(feature_matrix, feature_columns)
    )
    oof_poisson_home, oof_poisson_away = train_poisson_oof_predictions(feature_matrix)

    # CALIBRATE OPTIMAL CONSENSUS WEIGHTS
    if MODEL_TYPE == "blend":
        logging.info("🧩 Optimizing consensus blend via bounded SciPy solver...")
        blend_weights = find_optimal_blend_weights(
            feature_matrix=feature_matrix,
            g_home=g_home,
            g_away=g_away,
            oof_home_preds=oof_home_preds,
            oof_away_preds=oof_away_preds,
            oof_poisson_home=oof_poisson_home,
            oof_poisson_away=oof_poisson_away,
        )
    elif MODEL_TYPE == "poisson":
        blend_weights = {"poisson": 1.0, "elo": 0.0, "xgb": 0.0}
    elif MODEL_TYPE == "elo":
        blend_weights = {"poisson": 0.0, "elo": 1.0, "xgb": 0.0}
    elif MODEL_TYPE == "xgb":
        blend_weights = {"poisson": 0.0, "elo": 0.0, "xgb": 1.0}
    else:
        raise ValueError(
            f"❌ Unsupported MODEL_TYPE: '{MODEL_TYPE}'. Choose from 'blend', 'poisson', 'elo', 'xgb'."
        )

    logging.info(
        f"⚖️  Active Execution Weights: Poisson {blend_weights['poisson']:.3f} | Elo {blend_weights['elo']:.3f} | XGBoost {blend_weights['xgb']:.3f}"
    )

    # State tracking setup
    latest_team_form = {}
    for team in participating_teams:
        team_rows = feature_matrix[
            (feature_matrix["home_team"] == team)
            | (feature_matrix["away_team"] == team)
        ]
        if not team_rows.empty:
            latest_row = team_rows.iloc[-1]
            prefix = "home_team_" if latest_row["home_team"] == team else "away_team_"
            latest_team_form[team] = {
                "ewm_gf_4s": latest_row[f"{prefix}ewm_gf_4s"],
                "ewm_ga_4s": latest_row[f"{prefix}ewm_ga_4s"],
                "ewm_wr_4s": latest_row[f"{prefix}ewm_wr_4s"],
                "ewm_gf_10s": latest_row[f"{prefix}ewm_gf_10s"],
                "ewm_ga_10s": latest_row[f"{prefix}ewm_ga_10s"],
                "ewm_wr_10s": latest_row[f"{prefix}ewm_wr_10s"],
            }
        else:
            latest_team_form[team] = {
                "ewm_gf_4s": 1.2,
                "ewm_ga_4s": 1.2,
                "ewm_wr_4s": 0.35,
                "ewm_gf_10s": 1.2,
                "ewm_ga_10s": 1.2,
                "ewm_wr_10s": 0.35,
            }

    # --- GROUP STAGE SIMULATION ---
    logging.info("⚽ Commencing simulation of full Group Stage schedule...")
    group_results = []

    for _, row in group_fixtures.iterrows():
        match_id = int(row["match_id"])
        group_letter = row["group"]
        home = row["home_team"]
        away = row["away_team"]
        venue_country = row["venue_country"]

        # Call Match Engine
        raw_home, raw_away, p_corners, p_yellows, p_reds = evaluate_match_consensus(
            home_team=home,
            away_team=away,
            venue_country=venue_country,
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
            use_prior_nudge=USE_PRIOR_NUDGE,
            nudge_strength=NUDGE_STRENGTH,
            team_power=TEAM_POWER,
        )

        final_home_goals = int(np.round(max(0, raw_home)))
        final_away_goals = int(np.round(max(0, raw_away)))
        winner_side = (
            "home"
            if final_home_goals > final_away_goals
            else ("away" if final_away_goals > final_home_goals else "draw")
        )

        group_results.append(
            {
                "match_id": match_id,
                "group": group_letter,
                "home_team": home,
                "away_team": away,
                "predicted_home_goals": final_home_goals,
                "predicted_away_goals": final_away_goals,
                "corners": p_corners,
                "yellow_cards": p_yellows,
                "red_cards": p_reds,
                "winning_team": winner_side,
            }
        )

    predicted_fixtures = pd.DataFrame(group_results)

    # Route Bracket Structures
    group_tables = resolve_group_tables(predicted_fixtures)
    top_thirds = extract_best_third_places(group_tables)
    third_place_assignments = allocate_third_places(top_thirds)

    latest_team_form["__meta_weights__"] = blend_weights

    # --- KNOCKOUT WATERFALL SIMULATION ---
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
        use_prior_nudge=USE_PRIOR_NUDGE,
        nudge_strength=NUDGE_STRENGTH,
        team_power=TEAM_POWER,
    )

    logging.info("🗄️ Consolidating schemas into master unified ledger...")
    predicted_fixtures["round"] = "Group " + predicted_fixtures["group"]
    predicted_fixtures["extra_time"] = False
    predicted_fixtures["penalties"] = False
    predicted_fixtures["venue"] = group_fixtures.get("venue", "Neutral")

    predicted_fixtures["winner_name_meta"] = predicted_fixtures.apply(
        lambda r: (
            r["home_team"]
            if r["winning_team"] == "home"
            else (r["away_team"] if r["winning_team"] == "away" else "Draw")
        ),
        axis=1,
    )

    knockout_matrix = knockout_matrix.rename(
        columns={"predicted_home_team": "home_team", "predicted_away_team": "away_team"}
    )

    master_cols = [
        "match_id",
        "round",
        "venue",
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
        [predicted_fixtures[master_cols], knockout_matrix[master_cols]],
        ignore_index=True,
    )

    # Save Core Artifacts
    logging.info(f"💾 Committing unified dataset matrices to: {run_dir}")

    # 1. Save the Master Match Ledger
    master_tournament.to_csv(
        os.path.join(run_dir, "predicted_tournament.csv"), index=False
    )

    # 2. Save the Deterministic League Tables
    group_tables.to_csv(os.path.join(run_dir, "final_group_tables.csv"), index=False)

    # 3. Compile and Save official Wildcard Standings
    third_places_df = group_tables[group_tables["position"] == 3].copy()
    ranked_thirds_df = third_places_df.sort_values(
        by=["points", "goals_diff", "goals_for"], ascending=[False, False, False]
    ).reset_index(drop=True)
    ranked_thirds_df.to_csv(
        os.path.join(run_dir, "third_places_standings.csv"), index=False
    )

    # 4. Compile and Save the Master Team Capabilities Lookup Matrix
    capability_records = []
    for team in participating_teams:
        p_rat = ratings.get(team, {"attack": 1.0, "defense": 1.0})
        form = latest_team_form.get(team, {})

        record = {
            "Country": team,
            "Elo_Rating": elo_engine.get_rating(team),
            "Poisson_Attack": p_rat["attack"],
            "Poisson_Defense": p_rat["defense"],
            "Poisson_Dominance": p_rat["attack"] / max(0.01, p_rat["defense"]),
            "Short_Term_Form_GF": form.get("ewm_gf_4s", 1.2),
            "Short_Term_Form_WR": form.get("ewm_wr_4s", 0.35),
            "Long_Term_Form_GF": form.get("ewm_gf_10s", 1.2),
            "Long_Term_Form_WR": form.get("ewm_wr_10s", 0.35),
        }

        # Append Bayesian Priors Nudge
        if USE_PRIOR_NUDGE:
            record["Nudge_Power_Rating"] = TEAM_POWER.get(team, 75)
            record["Nudge_Raw_Delta"] = (
                TEAM_POWER.get(team, 75) / 100
            ) * NUDGE_STRENGTH
        else:
            record["Nudge_Power_Rating"] = None
            record["Nudge_Raw_Delta"] = None

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
            "run_monte_carlo": RUN_MONTE_CARLO,
            "monte_carlo_runs": MONTE_CARLO_RUNS,
            "use_prior_nudge": USE_PRIOR_NUDGE,
            "nudge_strength": NUDGE_STRENGTH,
        },
        "training_variables": TRAINING_VARIABLES,
        "match_rules": MATCH_RULES,
        "team_power": TEAM_POWER,
        "ensemble_weights": blend_weights,
        "cross_validation_metrics": cv_metrics,
    }

    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump(run_metadata, f, indent=4)

    final_match = knockout_matrix[knockout_matrix["round"] == "Final"].iloc[0]
    logging.info(
        f"🏆 Demodal path champion detected: {final_match['winner_name_meta'].upper()}!"
    )

    # --- PROBABILISTIC SIMULATION LAYER ---
    if RUN_MONTE_CARLO:
        logging.info(
            f"🎲 Spawning {MONTE_CARLO_RUNS:,} Monte Carlo parallel universes..."
        )
        raw_knockout_template = pd.read_csv(
            os.path.join("data", "raw", "knockout_slots.csv")
        )

        # Pre-compute knockout venue countries before entering the parallel engine
        raw_knockout_template["venue_country"] = raw_knockout_template["venue"].apply(
            get_venue_country
        )

        # Capture the metadata dictionary instead of discarding it with an underscore
        prob_dashboard, mc_metadata = run_monte_carlo_master(
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
            use_prior_nudge=USE_PRIOR_NUDGE,
            nudge_strength=NUDGE_STRENGTH,
            team_power=TEAM_POWER,
        )

        prob_dashboard.to_csv(
            os.path.join(run_dir, "monte_carlo_forecast.csv"), index=False
        )

        # Extract the pre-compiled fat artifact list directly and cast to a DataFrame
        fat_matchups = mc_metadata.get("fat_matchups", [])

        pd.DataFrame(fat_matchups).to_csv(
            os.path.join(run_dir, "pre_computed_matchups.csv"), index=False
        )

        logging.info(
            "🔮 Stochastic simulation complete. Pre-computed fat artifacts cached to disk."
        )
    else:
        logging.info(
            "🎲 Monte Carlo Engine: [DISABLED] skipping stochastic simulation."
        )

    logging.info("🏁 Core pipeline orchestration completed successfully.")


if __name__ == "__main__":
    main()
