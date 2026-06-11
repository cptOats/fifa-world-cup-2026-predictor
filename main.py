import os

import pandas as pd

from src.check_names import identify_name_mismatches
from src.elo_model import EloEngine
from src.ingest import verify_data_layer
from src.models import (
    DATACAMP_TO_KAGGLE,
    predict_match_score,
    print_team_power_rankings,
    train_poisson_ratings,
)
from src.prepare_data import prepare_historical_features
from src.router import (
    allocate_third_places,
    extract_best_third_places,
    resolve_group_tables,
    simulate_knockout_waterfall,
)

# --- MODEL CONFIGURATION TOGGLE ---
MODEL_TYPE = "elo"  # Options: "poisson" or "elo"


def main():
    print("🚀 Initializing World Cup Prediction Pipeline...")

    # --- INFRASTRUCTURE GATES ---
    verify_data_layer()
    print("\n--- Running Entity Resolution Check ---")
    identify_name_mismatches()
    print("\n--- Compiling Clean Historical Feature Matrix ---")
    prepare_historical_features()

    # --- DATA INGESTION ---
    modern_df = pd.read_parquet(
        os.path.join("data", "processed", "clean_historical_matches.parquet")
    )
    group_fixtures = pd.read_csv(os.path.join("data", "raw", "group_fixtures.csv"))

    # TRANSLATION GATE: Standardize fixtures to use Kaggle entities globally
    group_fixtures["home_team"] = group_fixtures["home_team"].replace(
        DATACAMP_TO_KAGGLE
    )
    group_fixtures["away_team"] = group_fixtures["away_team"].replace(
        DATACAMP_TO_KAGGLE
    )

    # Now this set will capture the true historical country names!
    participating_teams = set(group_fixtures["home_team"].unique()) | set(
        group_fixtures["away_team"].unique()
    )

    # --- PREDICTIVE MODELS ---
    print("\n--- Predictive Core Poisson Model ---")
    ratings, g_home, g_away = train_poisson_ratings()
    print(
        f"    Global Baseline Goal Expectancy: {(g_home + g_away) / 2:.2f} (Neutral) / {g_home:.2f} (Home) / {g_away:.2f} (Away)"
    )

    # Filter the Poisson ratings dict before handing it to the print engine
    participating_poisson = {
        team: coefs for team, coefs in ratings.items() if team in participating_teams
    }
    print_team_power_rankings(participating_poisson)

    elo_engine = None
    if MODEL_TYPE == "elo":
        print("\n📈 Training World Football Elo Engine components...")
        elo_engine = EloEngine(k_factor=40)
        elo_engine.fit(modern_df)

        # ELO RANKING: Extract, map, and sort the ratings descending
        elo_rankings = sorted(
            [(team, elo_engine.get_rating(team)) for team in participating_teams],
            key=lambda x: x[1],
            reverse=True,
        )

        # Print structured dashboard
        print("\n📊 World Football Elo Power Rankings (Tournament Field Only):")
        print("=" * 55)
        print(f"{'Rank':<5} | {'Country':<25} | {'Elo Rating':<10}")
        print("-" * 55)
        for rank, (team, score) in enumerate(elo_rankings, 1):
            print(f"{rank:>4}  | {team:<25} | {score:>10.1f}")
        print("=" * 55)

    # Execute Group Stage Simulation
    print("\n--- Executing Tournament Simulation ---")

    group_results = []

    for idx, row in group_fixtures.iterrows():
        match_id = int(row["match_id"])
        group_letter = row["group"]
        home = row["home_team"]
        away = row["away_team"]
        venue = row.get("venue", "Neutral Turf")

        p_home_goals, p_away_goals, p_corners, p_yellows, p_reds, p_winner = (
            predict_match_score(home, away, venue, ratings, g_home, g_away)
        )

        if MODEL_TYPE == "poisson":
            # Pure Dixon-Coles Poisson Path
            final_home_goals = p_home_goals
            final_away_goals = p_away_goals
            winner_side = p_winner

        elif MODEL_TYPE == "elo":
            # Core Elo Scoring Path
            elo_meta = elo_engine.predict_match(home, away)
            final_home_goals = elo_meta["predicted_home_goals"]
            final_away_goals = elo_meta["predicted_away_goals"]
            winner_side = elo_meta["winning_team"]

        # Append to group results dataframe array using consistent variables
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

    # Bind the toggle array directly to the active fixtures pipeline variable
    predicted_fixtures = pd.DataFrame(group_results)

    # Sort by group letter and match sequence for a clean, logical display
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

    # Run Sequential Knockout Waterfall
    knockout_matrix = simulate_knockout_waterfall(
        group_tables_df=group_tables,
        third_place_mapping=third_place_assignments,
        ratings=ratings,
        g_home_avg=g_home,
        g_away_avg=g_away,
        model_type=MODEL_TYPE,
        elo_engine=elo_engine,
    )

    # --- KNOCKOUT BRACKET STAGES ---
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

    group_output_path = os.path.join(results_dir, "predicted_group_stage.csv")
    knockout_output_path = os.path.join(results_dir, "predicted_knockout_bracket.csv")

    predicted_fixtures.to_csv(group_output_path, index=False)
    knockout_matrix.to_csv(knockout_output_path, index=False)

    print("💾 Production simulation ledgers safely saved to disk:")
    print(f"   - Group Stage Ledger: {group_output_path}")
    print(f"   - Knockout Bracket Ledger: {knockout_output_path}")

    # --- MACRO SENSE-CHECK: TOURNAMENT ---
    print("\n🔍👀 EXECUTING MACRO SENSE-CHECK (Tournament-Wide Metric Convergence):")
    print("=" * 100)

    # 1. Extract Group Stage Arrays
    g_corners = predicted_fixtures["corners"]
    g_yellows = predicted_fixtures["yellow_cards"]
    g_reds = predicted_fixtures["red_cards"]

    # 2. Extract Knockout Stage Arrays
    ko_corners = knockout_matrix["corners"]
    ko_yellows = knockout_matrix["yellow_cards"]
    ko_reds = knockout_matrix["red_cards"]

    # 3. Concatenate for Global Tournament Pools
    all_corners = pd.concat([g_corners, ko_corners])
    all_yellows = pd.concat([g_yellows, ko_yellows])
    all_reds = pd.concat([g_reds, ko_reds])

    # 4. Display Formatted Summary Standings Dashboard
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
    print("🎯 Target Verification Calibration Benchmarks:")
    print("   -> World Cup Target Baseline: ~9 Corners per 90min.")
    print("   -> World Cup Target Baseline: ~5 Yellows per 90min.")
    print("=" * 100)


if __name__ == "__main__":
    main()
