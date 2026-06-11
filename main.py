import os

import numpy as np
import pandas as pd
import xgboost as xgb

from src.blender import find_optimal_blend_weights
from src.check_names import identify_name_mismatches
from src.elo_model import EloEngine
from src.features import compile_master_feature_matrix
from src.ingest import verify_data_layer
from src.ml_engine import train_production_xgboost_models
from src.models import (
    DATACAMP_TO_KAGGLE,
    predict_match_score,
    print_team_power_rankings,
    train_poisson_oof_predictions,
    train_poisson_ratings,
)
from src.monte_carlo import run_monte_carlo_master
from src.prepare_data import prepare_historical_features
from src.router import (
    allocate_third_places,
    extract_best_third_places,
    resolve_group_tables,
    simulate_knockout_waterfall,
)

# --- MODEL CONFIGURATION TOGGLE ---
MODEL_TYPE = "ensemble"  # Options: "poisson", "elo", "xgboost", "ensemble"

# --- PROBABILISTIC MONTE CARLO TOGGLE ---
RUN_MONTE_CARLO = True  # Bool: Run Monte-Carlo Simulation
MONTE_CARLO_RUNS = 10000  # Total parallel universes to simulate (10k+ recommended)

# --- BAYESIAN EXPERT PRIOR TOGGLE ---
USE_PRIOR_NUDGE = True  # Bool: Apply Bayesian Expert Fine Tuning
NUDGE_STRENGTH = 1.5  # Tuning parameter controlling goal scaling factor

# --- POWER RATINGS TABLE --- source: https://www.datacamp.com/datalab/w/3da1cc64-5670-441e-8e7b-b948a6a29403
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


def main():
    print(
        f"🚀 Initializing World Cup Prediction Pipeline [Active Engine: {MODEL_TYPE.upper()}]..."
    )

    # --- INFRASTRUCTURE GATES ---
    verify_data_layer()
    print("\n--- Running Entity Resolution Check ---")
    identify_name_mismatches()
    print("\n--- Compiling Clean Historical Feature Matrix ---")
    prepare_historical_features()

    # --- EXPLICIT DATA INGESTION ---
    modern_df = pd.read_parquet(
        os.path.join("data", "processed", "clean_historical_matches.parquet")
    )
    group_fixtures = pd.read_csv(os.path.join("data", "raw", "group_fixtures.csv"))

    # Apply Master Entity Resolution Translation Layer
    group_fixtures["home_team"] = group_fixtures["home_team"].replace(
        DATACAMP_TO_KAGGLE
    )
    group_fixtures["away_team"] = group_fixtures["away_team"].replace(
        DATACAMP_TO_KAGGLE
    )
    participating_teams = set(group_fixtures["home_team"].unique()) | set(
        group_fixtures["away_team"].unique()
    )

    # --- ENTITY ALIGNMENT GATE FOR POWER RATINGS PRIORS ---
    if USE_PRIOR_NUDGE and TEAM_POWER:
        missing_priors = [
            team for team in participating_teams if team not in TEAM_POWER
        ]
        assert not missing_priors, (
            f"❌ TEAM_POWER String Mismatch! The following resolved tournament teams are missing keys in power dictionary: {missing_priors}"
        )
        print(
            "🎯 Verification Matrix: All tournament entities successfully verified in power ratings table."
        )

    # --- TELEMETRY PHASE: STATISTICAL POWER RANKINGS ---
    print("\n--- Predictive Core Poisson Model ---")
    ratings, g_home, g_away = train_poisson_ratings()
    print(
        f"    Global Baseline Goal Expectancy: {(g_home + g_away) / 2:.2f} (Neutral) / {g_home:.2f} (Home) / {g_away:.2f} (Away)"
    )
    # Truncate to participating tournament field only
    participating_poisson = {
        team: coefs for team, coefs in ratings.items() if team in participating_teams
    }
    print_team_power_rankings(participating_poisson)

    print("\n📈 Training World Football Elo Engine components...")
    elo_engine = EloEngine(k_factor=40)
    elo_engine.fit(modern_df)

    # Print Formatted Elo Standings Dashboard for Tournament Field
    elo_rankings = sorted(
        [(team, elo_engine.get_rating(team)) for team in participating_teams],
        key=lambda x: x[1],
        reverse=True,
    )
    print("\n📊 World Football Elo Power Rankings (Tournament Field Only):")
    print("=" * 55)
    print(f"{'Rank':<5} | {'Country':<25} | {'Elo Rating':<10}")
    print("-" * 55)
    for rank, (team, score) in enumerate(elo_rankings, 1):
        print(f"{rank:>4}  | {team:<25} | {score:>10.1f}")
    print("=" * 55)

    # --- MACHINE LEARNING ENGINE PIPELINE LAYER ---
    print("\n🌲 Compiling ML Feature Matrices and training Tree Ensembles...")
    feature_matrix, feature_columns = compile_master_feature_matrix(
        os.path.join("data", "processed", "clean_historical_matches.parquet"),
        elo_engine,
    )

    # 1. Capture XGBoost OOF arrays
    xgb_home, xgb_away, oof_home_preds, oof_away_preds = (
        train_production_xgboost_models(feature_matrix, feature_columns)
    )

    # 2. Capture Poisson OOF arrays
    oof_poisson_home, oof_poisson_away = train_poisson_oof_predictions(feature_matrix)

    # CALIBRATE OPTIMAL CONSENSUS WEIGHTS
    blend_weights = find_optimal_blend_weights(
        feature_matrix=feature_matrix,
        ratings=ratings,
        g_home=g_home,
        g_away=g_away,
        elo_engine=elo_engine,
        xgb_home=xgb_home,
        xgb_away=xgb_away,
        feature_columns=feature_columns,
        oof_home_preds=oof_home_preds,
        oof_away_preds=oof_away_preds,
        oof_poisson_home=oof_poisson_home,
        oof_poisson_away=oof_poisson_away,
    )

    # DIAGNOSTIC PRINT: Display the mathematical optimization weights clearly
    print("\n⚖️  ACTIVE CONSENSUS MODEL WEIGHT DISTRIBUTION:")
    print("=" * 55)
    print(f"   📊 Poisson Base Weight (w1) : {blend_weights['poisson']:.4f}")
    print(f"   📈 Elo Engine Weight (w2)   : {blend_weights['elo']:.4f}")
    print(f"   🌲 XGBoost Tree Weight (w3) : {blend_weights['xgboost']:.4f}")
    print(f"   Verified Total Coefficient  : {sum(blend_weights.values()):.2f}")
    print("=" * 55)

    # State tracking: Extract the single most recent form row for every country to use as tournament baseline
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
            # Default fallbacks
            latest_team_form[team] = {
                "ewm_gf_4s": 1.2,
                "ewm_ga_4s": 1.2,
                "ewm_wr_4s": 0.35,
                "ewm_gf_10s": 1.2,
                "ewm_ga_10s": 1.2,
                "ewm_wr_10s": 0.35,
            }

    # Execute Group Stage Simulation
    print("\n--- Executing Tournament Simulation Loop ---")
    group_results = []

    for idx, row in group_fixtures.iterrows():
        match_id = int(row["match_id"])
        group_letter = row["group"]
        home = row["home_team"]
        away = row["away_team"]
        venue = row.get("venue", "Neutral Turf")

        # Compute pure baseline statistical models
        p_home_goals, p_away_goals, p_corners, p_yellows, p_reds, p_winner = (
            predict_match_score(home, away, venue, ratings, g_home, g_away)
        )
        elo_meta = elo_engine.predict_match(home, away)

        # Build live XGBoost match features
        live_match_vector = {
            "home_elo_rating": elo_engine.get_rating(home),
            "away_elo_rating": elo_engine.get_rating(away),
            "elo_differential": elo_engine.get_rating(home)
            - elo_engine.get_rating(away),
            "is_neutral_venue": 1,
            "home_team_ewm_gf_4s": latest_team_form[home]["ewm_gf_4s"],
            "home_team_ewm_ga_4s": latest_team_form[home]["ewm_ga_4s"],
            "home_team_ewm_wr_4s": latest_team_form[home]["ewm_wr_4s"],
            "home_team_ewm_gf_10s": latest_team_form[home]["ewm_gf_10s"],
            "home_team_ewm_ga_10s": latest_team_form[home]["ewm_ga_10s"],
            "home_team_ewm_wr_10s": latest_team_form[home]["ewm_wr_10s"],
            "away_team_ewm_gf_4s": latest_team_form[away]["ewm_gf_4s"],
            "away_team_ewm_ga_4s": latest_team_form[away]["ewm_ga_4s"],
            "away_team_ewm_wr_4s": latest_team_form[away]["ewm_wr_4s"],
            "away_team_ewm_gf_10s": latest_team_form[away]["ewm_gf_10s"],
            "away_team_ewm_ga_10s": latest_team_form[away]["ewm_ga_10s"],
            "away_team_ewm_wr_10s": latest_team_form[away]["ewm_wr_10s"],
        }
        match_df = pd.DataFrame([live_match_vector])[feature_columns]
        xgb_h_pred = xgb_home.predict(match_df)[0]
        xgb_w_pred = xgb_away.predict(match_df)[0]

        # --- RE-ENGINEERED OVERLAY DECISION ROUTER ---
        if MODEL_TYPE == "poisson":
            final_home_goals, final_away_goals = p_home_goals, p_away_goals
            winner_side = p_winner
        elif MODEL_TYPE == "elo":
            final_home_goals, final_away_goals = (
                elo_meta["predicted_home_goals"],
                elo_meta["predicted_away_goals"],
            )
            winner_side = elo_meta["winning_team"]
        elif MODEL_TYPE == "xgboost":
            final_home_goals = int(np.round(xgb_h_pred))
            final_away_goals = int(np.round(xgb_w_pred))
            winner_side = (
                "home"
                if final_home_goals > final_away_goals
                else ("away" if final_away_goals > final_home_goals else "draw")
            )

        elif MODEL_TYPE == "ensemble":
            # 1. Classical Statistical Baselines
            h_poisson_baseline = (
                ratings.get(home, {}).get("attack", 1)
                * ratings.get(away, {}).get("defense", 1)
                * ((g_home + g_away) / 2.0)
            )
            a_poisson_baseline = (
                ratings.get(away, {}).get("attack", 1)
                * ratings.get(home, {}).get("defense", 1)
                * ((g_home + g_away) / 2.0)
            )

            # 2. Empirical Machine Learning Consensus Blend
            blend_home_raw = (
                (blend_weights["poisson"] * h_poisson_baseline)
                + (blend_weights["elo"] * elo_meta["predicted_home_goals"])
                + (blend_weights["xgboost"] * xgb_h_pred)
            )
            blend_away_raw = (
                (blend_weights["poisson"] * a_poisson_baseline)
                + (blend_weights["elo"] * elo_meta["predicted_away_goals"])
                + (blend_weights["xgboost"] * xgb_w_pred)
            )

            # 3. 2026 Power Ratings
            home_power = TEAM_POWER.get(home, 75)
            away_power = TEAM_POWER.get(away, 75)
            power_gap = home_power - away_power

            #  Gentle Bayesian adjustment to fine-tune Data Latency problem
            prior_nudge = (power_gap / 100) * NUDGE_STRENGTH

            blend_home_raw += prior_nudge
            blend_away_raw -= prior_nudge

            # 4. Final Integer Output Compilation
            final_home_goals = int(np.round(max(0, blend_home_raw)))
            final_away_goals = int(np.round(max(0, blend_away_raw)))
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

    # TELEMETRY PHASE: MATCH-BY-MATCH GROUP STAGE DISPLAY
    print("\n📊 Simulating Group Stage:")
    print("=" * 100)
    sorted_fixtures = predicted_fixtures.sort_values(by=["group", "match_id"])
    current_group = ""
    for _, match in sorted_fixtures.iterrows():
        if match["group"] != current_group:
            current_group = match["group"]
            print(f"\n--- GROUP {current_group} ---")
        print(
            f"Match {match['match_id']:>2} | {match['predicted_home_goals']} - {match['predicted_away_goals']} | Corners: {match['corners']:>2} | YC: {match['yellow_cards']:>2} | RC: {match['red_cards']} | Winner: {match['winning_team']:<5} | {match['home_team']:>18} vs {match['away_team']}"
        )
    print("=" * 100)

    # Route Bracket Structures
    group_tables = resolve_group_tables(predicted_fixtures)
    top_thirds = extract_best_third_places(group_tables)
    third_place_assignments = allocate_third_places(top_thirds)

    latest_team_form["__meta_weights__"] = blend_weights
    # Run Sequential Knockout Waterfall
    knockout_matrix = simulate_knockout_waterfall(
        group_tables_df=group_tables,
        third_place_mapping=third_place_assignments,
        ratings=ratings,
        g_home_avg=g_home,
        g_away_avg=g_away,
        model_type=MODEL_TYPE,
        elo_engine=elo_engine,
        xgb_home=xgb_home,
        xgb_away=xgb_away,
        feature_columns=feature_columns,
        latest_team_form=latest_team_form,
    )

    # TELEMETRY PHASE: KNOCKOUT BRACKET DISPLAY
    rounds_to_print = [
        "Round of 32",
        "Round of 16",
        "Quarter-final",
        "Semi-final",
        "Third-place playoff",
        "Final",
    ]
    for r_title in rounds_to_print:
        print(f"\n⚡ {r_title.upper()}:")
        print("=" * 125)
        r_df = knockout_matrix[knockout_matrix["round"] == r_title]
        for _, match in r_df.iterrows():
            print(
                f"Match {match['match_id']:>3} | {match['predicted_home_goals']}-{match['predicted_away_goals']} | Corners: {match['corners']:>2} | YC: {match['yellow_cards']:>2} | PK Shootout: {str(match['penalties']):<5} -> ADVANCES: {match['winner_name_meta']:<16} ({match['predicted_home_team']} vs {match['predicted_away_team']})"
            )
        print("=" * 125)

    final_match = knockout_matrix[knockout_matrix["round"] == "Final"].iloc[0]
    print(
        f"\n{'🏆' * 20} {final_match['winner_name_meta'].upper()} - 2026 FIFA WORLD CUP CHAMPIONS {'🏆' * 20}\n"
    )

    # --- PERSISTENCE LAYER: SAVE ARTIFACTS TO DISK ---
    results_dir = os.path.join("data", "results")
    os.makedirs(results_dir, exist_ok=True)
    predicted_fixtures.to_csv(
        os.path.join(results_dir, "predicted_group_stage.csv"), index=False
    )
    knockout_matrix.to_csv(
        os.path.join(results_dir, "predicted_knockout_bracket.csv"), index=False
    )

    # --- TELEMETRY PHASE: MACRO SENSE-CHECK DASHBOARD ---
    print("\n🔍👀 EXECUTING MACRO SENSE-CHECK (Tournament-Wide Metric Convergence):")
    print("=" * 100)
    g_corners, g_yellows, g_reds = (
        predicted_fixtures["corners"],
        predicted_fixtures["yellow_cards"],
        predicted_fixtures["red_cards"],
    )
    ko_corners, ko_yellows, ko_reds = (
        knockout_matrix["corners"],
        knockout_matrix["yellow_cards"],
        knockout_matrix["red_cards"],
    )
    all_corners = pd.concat([g_corners, ko_corners])
    all_yellows = pd.concat([g_yellows, ko_yellows])
    all_reds = pd.concat([g_reds, ko_reds])

    print(
        f"{'Tournament Phase':<20} | {'Sample Size':<12} | {'Avg Corners':<13} | {'Avg Yellow Cards':<16} | {'Avg Red Cards':<13}"
    )
    print("-" * 100)
    print(
        f"{'Group Stage':<20} | {len(g_corners):<12} | {g_corners.mean():<13.2f} | {g_yellows.mean():<16.2f} | {g_reds.mean():<13.2f}"
    )
    print(
        f"{'Knockout Phase':<20} | {len(ko_corners):<12} | {ko_corners.mean():<13.2f} | {ko_yellows.mean():<16.2f} | {ko_reds.mean():<13.2f}"
    )
    print("-" * 100)
    print(
        f"{'TOURNAMENT TOTAL':<20} | {len(all_corners):<12} | {all_corners.mean():<13.2f} | {all_yellows.mean():<16.2f} | {all_reds.mean():<13.2f}"
    )
    print("=" * 100)

    # Load the raw knockout structure template to hand off to the simulator
    raw_knockout_template = pd.read_csv(
        os.path.join("data", "raw", "knockout_slots.csv")
    )

    # --- PROBABILISTIC SIMULATION LAYER ---
    if RUN_MONTE_CARLO:
        # Load the raw knockout structure template to hand off to the simulator
        raw_knockout_template = pd.read_csv(
            os.path.join("data", "raw", "knockout_slots.csv")
        )

        # Execute the Master Monte Carlo Suite at scale
        prob_dashboard, master_ledgers = run_monte_carlo_master(
            group_fixtures=group_fixtures,
            raw_knockout_template=raw_knockout_template,
            ratings=ratings,
            g_home=g_home,
            g_away=g_away,
            elo_engine=elo_engine,
            xgb_home=xgb_home,
            xgb_away=xgb_away,
            feature_columns=feature_columns,
            latest_team_form=latest_team_form,
            blend_weights=blend_weights,
            n_simulations=MONTE_CARLO_RUNS,
        )

        # PERSIST FORECAST MATRIX TO DISK
        output_dir = os.path.join("data", "results")
        os.makedirs(output_dir, exist_ok=True)

        csv_path = os.path.join(output_dir, "monte_carlo_forecast.csv")
        prob_dashboard.to_csv(csv_path, index=False)

        print(f"\n💾 PERSISTENCE LAYER: Probability matrix cached cleanly to:")
        print(f"   ↳ {csv_path}")

    else:
        print(
            "\n🎲 Monte Carlo Engine: [DISABLED] Skipping probabilistic calculations."
        )


if __name__ == "__main__":
    main()
