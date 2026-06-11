import os

import numpy as np
import pandas as pd

from src.router import (
    allocate_third_places,
    extract_best_third_places,
    resolve_group_tables,
)


def run_monte_carlo_master(
    group_fixtures,
    raw_knockout_template,
    ratings,
    g_home,
    g_away,
    elo_engine,
    xgb_home,
    xgb_away,
    feature_columns,
    latest_team_form,
    blend_weights,
    n_simulations=10000,
):
    """
    Executes N randomized tournament simulations using an optimized global pre-computed
    matchup matrix cache. Eliminates inline Pandas dataframe creation and model prediction overhead.
    """
    print(
        f"\n🎲 Initializing Probabilistic Monte Carlo Engine ({n_simulations:,} runs)..."
    )

    # Initialize NumPy Generator with a fixed seed
    rng = np.random.default_rng(seed=69)

    participating_teams = list(
        set(group_fixtures["home_team"].unique())
        | set(group_fixtures["away_team"].unique())
    )

    # 1. Initialize Global Probability Counters
    metrics = {
        team: {
            "Group Stage Exit": 0,
            "Round of 32": 0,
            "Round of 16": 0,
            "Quarter-final": 0,
            "Semi-final": 0,
            "3rd Place": 0,
            "Finalist": 0,
            "Champion": 0,
        }
        for team in participating_teams
    }

    g_neutral = (g_home + g_away) / 2.0
    ET_MULTIPLIER = 1 / 3
    FATIGUE_FACTOR = 0.80

    # --- VECTORIZED MATCHUP MATRIX PRE-COMPUTATION ---
    print(
        "   ↳ Vectorizing and caching consensus lambda matrices for all possible matchups..."
    )
    matchup_rows = []
    matchup_keys = []

    for h in participating_teams:
        for a in participating_teams:
            if h == a:
                continue
            matchup_keys.append((h, a))
            matchup_rows.append(
                {
                    "home_elo_rating": elo_engine.get_rating(h),
                    "away_elo_rating": elo_engine.get_rating(a),
                    "elo_differential": elo_engine.get_rating(h)
                    - elo_engine.get_rating(a),
                    "is_neutral_venue": 1,
                    "home_team_ewm_gf_4s": latest_team_form[h]["ewm_gf_4s"],
                    "home_team_ewm_ga_4s": latest_team_form[h]["ewm_ga_4s"],
                    "home_team_ewm_wr_4s": latest_team_form[h]["ewm_wr_4s"],
                    "home_team_ewm_gf_10s": latest_team_form[h]["ewm_gf_10s"],
                    "home_team_ewm_ga_10s": latest_team_form[h]["ewm_ga_10s"],
                    "home_team_ewm_wr_10s": latest_team_form[h]["ewm_wr_10s"],
                    "away_team_ewm_gf_4s": latest_team_form[a]["ewm_gf_4s"],
                    "away_team_ewm_ga_4s": latest_team_form[a]["ewm_ga_4s"],
                    "away_team_ewm_wr_4s": latest_team_form[a]["ewm_wr_4s"],
                    "away_team_ewm_gf_10s": latest_team_form[a]["ewm_gf_10s"],
                    "away_team_ewm_ga_10s": latest_team_form[a]["ewm_ga_10s"],
                    "away_team_ewm_wr_10s": latest_team_form[a]["ewm_wr_10s"],
                }
            )

    # Send all 2,256 combinations through XGBoost in one single vectorized batch operation
    matchup_df = pd.DataFrame(matchup_rows)[feature_columns]
    xgb_h_all = xgb_home.predict(matchup_df)
    xgb_a_all = xgb_away.predict(matchup_df)

    # Build the O(1) Consensus Parameter Map
    lambda_cache = {}
    for idx, (h, a) in enumerate(matchup_keys):
        h_poi = (
            ratings.get(h, {}).get("attack", 1)
            * ratings.get(a, {}).get("defense", 1)
            * g_neutral
        )
        a_poi = (
            ratings.get(a, {}).get("attack", 1)
            * ratings.get(h, {}).get("defense", 1)
            * g_neutral
        )
        elo_meta = elo_engine.predict_match(h, a)

        l_h = (
            (blend_weights["poisson"] * h_poi)
            + (blend_weights["elo"] * elo_meta["predicted_home_goals"])
            + (blend_weights["xgboost"] * xgb_h_all[idx])
        )
        l_a = (
            (blend_weights["poisson"] * a_poi)
            + (blend_weights["elo"] * elo_meta["predicted_away_goals"])
            + (blend_weights["xgboost"] * xgb_a_all[idx])
        )

        corners_exp = (
            5.5
            * ratings.get(h, {}).get("attack", 1)
            * ratings.get(a, {}).get("defense", 1)
        ) + (
            5.5
            * ratings.get(a, {}).get("attack", 1)
            * ratings.get(h, {}).get("defense", 1)
        )
        yellows_exp = (
            3.0
            * ratings.get(h, {}).get("defense", 1)
            * ratings.get(a, {}).get("attack", 1)
        ) + (
            3.0
            * ratings.get(a, {}).get("defense", 1)
            * ratings.get(h, {}).get("attack", 1)
        )

        lambda_cache[(h, a)] = (l_h, l_a, corners_exp, yellows_exp)

    # Convert core dataframes to lightweight lists of dicts to kill .iterrows() overhead completely
    group_fixtures_list = group_fixtures[
        ["match_id", "group", "home_team", "away_team"]
    ].to_dict(orient="records")
    knockout_template_list = raw_knockout_template[
        ["match_id", "round", "slot_home", "slot_away"]
    ].to_dict(orient="records")

    # 2. Start the Optimized Master Monte Carlo Loop
    print("   ↳ Executing full tournament simulations...")
    for sim in range(n_simulations):
        group_results = []

        # --- PHASE A: GROUP STAGE SAMPLING ---
        for row in group_fixtures_list:
            m_id, group, home, away = (
                row["match_id"],
                row["group"],
                row["home_team"],
                row["away_team"],
            )
            l_h, l_a, c_exp, y_exp = lambda_cache[(home, away)]

            group_results.append(
                {
                    "match_id": m_id,
                    "group": group,
                    "home_team": home,
                    "away_team": away,
                    "predicted_home_goals": np.random.poisson(l_h),
                    "predicted_away_goals": np.random.poisson(l_a),
                    "corners": int(
                        np.clip(np.round(np.random.normal(c_exp, 1.5)), 4, 18)
                    ),
                    "yellow_cards": int(
                        np.clip(np.round(np.random.normal(y_exp, 1.2)), 0, 10)
                    ),
                    "red_cards": 0,
                }
            )

        sim_fixtures_df = pd.DataFrame(group_results)
        tables = resolve_group_tables(sim_fixtures_df)
        top_thirds = extract_best_third_places(tables)
        third_place_assignments = allocate_third_places(top_thirds)

        winners = tables[tables["position"] == 1].set_index("group")["team"].to_dict()
        runners = tables[tables["position"] == 2].set_index("group")["team"].to_dict()

        for team in participating_teams:
            pos = tables[tables["team"] == team]["position"].values[0]
            if pos == 1 or pos == 2:
                metrics[team]["Round of 32"] += 1
            elif pos == 3 and team in third_place_assignments.values():
                metrics[team]["Round of 32"] += 1
            else:
                metrics[team]["Group Stage Exit"] += 1

        # --- PHASE B: SEQUENTIAL KNOCKOUT WATERFALL SAMPLING ---
        match_winners = {}
        match_losers = {}

        for row in knockout_template_list:
            m_id, r_name, slot_home, slot_away = (
                row["match_id"],
                row["round"],
                row["slot_home"],
                row["slot_away"],
            )

            if "Winner Group" in slot_home:
                home = winners[slot_home.replace("Winner Group ", "").strip()]
            elif "Runner-up Group" in slot_home:
                home = runners[slot_home.replace("Runner-up Group ", "").strip()]
            elif "Best 3rd" in slot_home:
                home = third_place_assignments[m_id]
            elif "Winner Match" in slot_home:
                home = match_winners[
                    int(slot_home.replace("Winner Match ", "").strip())
                ]
            elif "Loser Match" in slot_home:
                home = match_losers[int(slot_home.replace("Loser Match ", "").strip())]
            else:
                home = slot_home

            if "Winner Group" in slot_away:
                away = winners[slot_away.replace("Winner Group ", "").strip()]
            elif "Runner-up Group" in slot_away:
                away = runners[slot_away.replace("Runner-up Group ", "").strip()]
            elif "Best 3rd" in slot_away:
                away = third_place_assignments[m_id]
            elif "Winner Match" in slot_away:
                away = match_winners[
                    int(slot_away.replace("Winner Match ", "").strip())
                ]
            elif "Loser Match" in slot_away:
                away = match_losers[int(slot_away.replace("Loser Match ", "").strip())]
            else:
                away = slot_away

            # Instant O(1) Cache Parameter Retrieval
            l_h, l_a, _, _ = lambda_cache[(home, away)]

            h_goals = np.random.poisson(l_h)
            a_goals = np.random.poisson(l_a)

            if h_goals == a_goals:
                h_goals += np.random.poisson(l_h * (ET_MULTIPLIER * FATIGUE_FACTOR))
                a_goals += np.random.poisson(l_a * (ET_MULTIPLIER * FATIGUE_FACTOR))

                if h_goals == a_goals:
                    h_elo_stat = elo_engine.get_rating(home)
                    a_elo_stat = elo_engine.get_rating(away)
                    winner = (
                        home
                        if np.random.rand() < (h_elo_stat / (h_elo_stat + a_elo_stat))
                        else away
                    )
                else:
                    winner = home if h_goals > a_goals else away
            else:
                winner = home if h_goals > a_goals else away

            loser = away if winner == home else home
            match_winners[m_id] = winner
            match_losers[m_id] = loser

            if r_name == "Round of 32":
                metrics[winner]["Round of 16"] += 1
            elif r_name == "Round of 16":
                metrics[winner]["Quarter-final"] += 1
            elif r_name == "Quarter-final":
                metrics[winner]["Semi-final"] += 1
            elif r_name == "Semi-final":
                metrics[winner]["Finalist"] += 1
            elif r_name == "Third-place playoff":
                metrics[winner]["3rd Place"] += 1
            elif r_name == "Final":
                metrics[winner]["Champion"] += 1
                metrics[loser]["Finalist"] += 1

    # 3. Compile and Format the Master Probability Table
    prob_list = []
    for team, stages in metrics.items():
        prob_list.append(
            {
                "Country": team,
                "R32 %": (stages["Round of 32"] / n_simulations) * 100,
                "R16 %": (stages["Round of 16"] / n_simulations) * 100,
                "QF %": (stages["Quarter-final"] / n_simulations) * 100,
                "SF %": (stages["Semi-final"] / n_simulations) * 100,
                "3rd %": (stages["3rd Place"] / n_simulations) * 100,
                "Final %": (stages["Finalist"] / n_simulations) * 100,
                "Champion %": (stages["Champion"] / n_simulations) * 100,
            }
        )

    prob_df = (
        pd.DataFrame(prob_list)
        .sort_values(by="Champion %", ascending=False)
        .reset_index(drop=True)
    )

    print("\n📊 MONTE CARLO PROBABILISTIC FORECAST:")
    print("=" * 125)
    print(
        f"{'Country':<22} | {'R32 %':<10} | {'R16 %':<10} | {'QF %':<10} | {'SF %':<10} | {'3rd %':<10} | {'Final %':<10} | {'Champion %':<10}"
    )
    print("-" * 125)
    for idx, row in prob_df.iterrows():
        print(
            f"{row['Country']:<22} | {row['R32 %']:<10.1f} | {row['R16 %']:<10.1f} | {row['QF %']:<10.1f} | {row['SF %']:<10.1f} | {row['3rd %']:<10.1f} | {row['Final %']:<10.1f} | {row['Champion %']:.2f}"
        )
    print("=" * 125)

    return prob_df, {}
