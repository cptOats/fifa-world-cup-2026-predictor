import os

import pandas as pd

from src.check_names import identify_name_mismatches
from src.ingest import verify_data_layer
from src.models import (
    print_team_power_rankings,
    simulate_group_stage,
    train_poisson_ratings,
)
from src.prepare_data import prepare_historical_features
from src.router import (
    allocate_third_places,
    extract_best_third_places,
    resolve_group_tables,
    simulate_knockout_waterfall,
)


def main():
    print("🚀 Initializing World Cup Prediction Pipeline...")

    # --- INFRASTRUCTURE GATES---
    verify_data_layer()
    print("\n--- Running Entity Resolution Check ---")
    identify_name_mismatches()
    print("\n--- Compiling Clean Historical Feature Matrix ---")
    prepare_historical_features()

    # --- PREDICTIVE CORE MODEL ---
    print("\n--- Predictive Core Poisson Model ---")
    ratings, g_home, g_away = train_poisson_ratings()
    print(
        f"   Global Baseline Goal Expectancy: {(g_home + g_away) / 2:.2f} (Neutral) / {g_home:.2f} (Home) / {g_away:.2f} (Away)"
    )
    print_team_power_rankings(ratings)

    # Execute Group Stage Simulation
    print("\n--- Executing Tournament Simulation ---")
    predicted_fixtures = simulate_group_stage(ratings, g_home, g_away)

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
            f"Match {match['match_id']:>2} | {match['predicted_home_goals']} - {match['predicted_away_goals']} | Corners: {match['corners']:>2} | YC: {match['yellow_cards']} | RC: {match['red_cards']} | Winner: {match['winning_team']:<5} | {match['home_team']:>18} vs {match['away_team']}"
        )
    print("=" * 100)

    # Route Bracket Structures
    group_tables = resolve_group_tables(predicted_fixtures)
    top_thirds = extract_best_third_places(group_tables)
    third_place_assignments = allocate_third_places(top_thirds)

    # Run Sequential Knockout Waterfall
    knockout_matrix = simulate_knockout_waterfall(
        group_tables, third_place_assignments, ratings, g_home, g_away
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
                f"Match {match['match_id']:>3} | {match['predicted_home_goals']}-{match['predicted_away_goals']} | Corners: {match['corners']:>2} | YC: {match['yellow_cards']} | PK Shootout: {str(match['penalties']):<5} -> ADVANCES: {match['winner_name_meta']:<16} ({match['predicted_home_team']} vs {match['predicted_away_team']})"
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

    # 3. Concatenate for Global Tournament Pools (All 104 Matches)
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
