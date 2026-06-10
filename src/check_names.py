import os

import pandas as pd


def identify_name_mismatches():
    raw_dir = os.path.join("data", "raw")

    # Load DataCamp's tournament fixtures
    fixtures_df = pd.read_csv(os.path.join(raw_dir, "group_fixtures.csv"))
    datacamp_teams = set(fixtures_df["home_team"].unique()).union(
        set(fixtures_df["away_team"].unique())
    )

    # Load Kaggle's historical results
    kaggle_df = pd.read_csv(os.path.join(raw_dir, "results.csv"))
    kaggle_teams = set(kaggle_df["home_team"].unique()).union(
        set(kaggle_df["away_team"].unique())
    )

    # Check differences
    mismatches = datacamp_teams - kaggle_teams

    print(f"Total unique teams in DataCamp fixtures: {len(datacamp_teams)}")
    if mismatches:
        print("\n⚠️ MISMATCHES FOUND!")
        for team in sorted(mismatches):
            print(f" - {team}")
    else:
        print(
            "\n✅ Perfect alignment! All DataCamp teams exist in the Kaggle history match ledger."
        )
