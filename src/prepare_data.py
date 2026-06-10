import os

import numpy as np
import pandas as pd


def prepare_historical_features():
    print("🧠 Initializing Feature Engineering with Dynamic Weighting Matrix...")

    raw_dir = os.path.join("data", "raw")
    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    # Load Kaggle Raw Data
    results_df = pd.read_csv(os.path.join(raw_dir, "results.csv"))

    # Apply Time Horizon Slice (Modern Era / Post 2018 World Cup)
    results_df["date"] = pd.to_datetime(results_df["date"])
    modern_df = results_df[results_df["date"] >= "2018-08-01"].copy()

    # Assign Dynamic Match Weights: Variable Friendly Weighting [consider 0.33-0.5]
    modern_df["match_weight"] = np.where(
        modern_df["tournament"] == "Friendly", 0.4, 1.0
    )

    # Core Feature Generation
    co_hosts = {"United States", "Mexico", "Canada"}
    modern_df["is_true_home"] = modern_df.apply(
        lambda row: row["home_team"] in co_hosts and row["country"] == row["home_team"],
        axis=1,
    )
    modern_df["total_goals"] = modern_df["home_score"] + modern_df["away_score"]

    # Export complete matrix to Parquet
    output_path = os.path.join(processed_dir, "clean_historical_matches.parquet")
    modern_df.to_parquet(output_path, index=False)

    # Quick diagnostics printout
    total_rows = len(modern_df)
    active_rows = len(modern_df[modern_df["match_weight"] > 0])
    print(f"✅ Complete. Feature store updated at: {output_path}")
    print(
        f"   Preserved {total_rows} total rows ({active_rows} active competitive matches, {total_rows - active_rows} weighted friendly buffers)."
    )


if __name__ == "__main__":
    prepare_historical_features()
