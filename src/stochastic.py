"""Stochastic Simulation and Probabilistic Forecasting Layer.

This module houses the master Monte Carlo orchestration engine. It bypasses
repetitive model evaluations during deep iteration runs by pre-computing a vectorized
consensus parameter matrix cache (O(1) lookup complexity). It handles thousands of
randomized tournament simulations to compile explicit survival probabilities from the
Group Stage through to the Final.
"""

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
    g_neutral,
    elo_engine,
    xgb_home,
    xgb_away,
    feature_columns,
    latest_team_form,
    blend_weights,
    n_simulations=10000,
):
    r"""Executes randomized tournament simulations using an optimized global matchup cache.

    Optimizes performance by evaluating all possible matchup combinations (**4,512** vectors)
    across neutral and non-neutral states through the underlying machine learning layers in
    a single vectorized batch before spawning the simulation loop.

    Match outcomes are drawn directly from independent Poisson distributions utilizing
    the pre-calculated ensemble consensus intensities ($\lambda$). Regular integer draws
    in knockout rounds trigger a 30-minute Extra Time Poisson extension containing a decay scalar
    for fatigue, while sudden-death penalty ties are resolved via randomized draws
    weighted by relative team Elo vectors.

    Args:
        group_fixtures (pd.DataFrame): Dataframe tracking initial structural group stage pairs.
        raw_knockout_template (pd.DataFrame): Master scheduling bracket layout spreadsheet.
        ratings (dict): Base historical Poisson attack and defense metrics per team.
        g_home (float): Dataset global average score metric for home-side goal references.
        g_away (float): Dataset global average score metric for away-side goal references.
        g_neutral (float): Dataset global average score metric for neutral venue references.
        elo_engine (EloEngine): Pre-fitted tracking object instance evaluating team Elo scores.
        xgb_home (xgb.XGBRegressor): Pre-fitted tree regressor tracking home goal counts.
        xgb_away (xgb.XGBRegressor): Pre-fitted tree regressor tracking away goal counts.
        feature_columns (list[str]): Explicit string column layout index passed to ML frames.
        latest_team_form (dict): Current rolling exponentially weighted statistics per team.
        blend_weights (dict[str, float]): Calibrated optimal consensus weight coefficients.
        n_simulations (int, optional): Total iteration volume of randomized parallel universes.

    Returns:
        tuple[pd.DataFrame, dict]: Master survival dashboard matrix and an empty metadata log dictionary.
    """
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

    ET_MULTIPLIER = 1 / 3
    FATIGUE_FACTOR = 0.80

    # --- VECTORIZED MATCHUP MATRIX PRE-COMPUTATION ---
    matchup_rows = []
    matchup_keys = []

    #    Extract unique hosting countries directly from active fixtures
    unique_venues = list(group_fixtures["venue_country"].unique())

    for v_country in unique_venues:
        for h in participating_teams:
            for a in participating_teams:
                if h == a:
                    continue

                is_neutral_flag = 0 if (h == v_country or a == v_country) else 1
                matchup_keys.append((h, a, v_country))
                matchup_rows.append(
                    {
                        "home_elo_rating": elo_engine.get_rating(h),
                        "away_elo_rating": elo_engine.get_rating(a),
                        "elo_differential": elo_engine.get_rating(h)
                        - elo_engine.get_rating(a),
                        "is_neutral_venue": is_neutral_flag,
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

    matchup_df = pd.DataFrame(matchup_rows)[feature_columns]
    xgb_h_all = xgb_home.predict(matchup_df)
    xgb_a_all = xgb_away.predict(matchup_df)

    lambda_cache = {}
    for idx, (h, a, v_country) in enumerate(matchup_keys):
        is_neutral_flag = 0 if (h == v_country or a == v_country) else 1

        # Directional Poisson verification inside cache loop
        if h == v_country:
            h_poi = (
                ratings.get(h, {}).get("attack", 1)
                * ratings.get(a, {}).get("defense", 1)
                * g_home
            )
            a_poi = (
                ratings.get(a, {}).get("attack", 1)
                * ratings.get(h, {}).get("defense", 1)
                * g_away
            )
        elif a == v_country:
            h_poi = (
                ratings.get(h, {}).get("attack", 1)
                * ratings.get(a, {}).get("defense", 1)
                * g_away
            )
            a_poi = (
                ratings.get(a, {}).get("attack", 1)
                * ratings.get(h, {}).get("defense", 1)
                * g_home
            )
        else:
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

        elo_meta = elo_engine.predict_elo_match(h, a, is_neutral=is_neutral_flag)

        l_h = (
            (blend_weights["poisson"] * h_poi)
            + (blend_weights["elo"] * elo_meta["predicted_home_goals"])
            + (blend_weights["xgb"] * xgb_h_all[idx])
        )
        l_a = (
            (blend_weights["poisson"] * a_poi)
            + (blend_weights["elo"] * elo_meta["predicted_away_goals"])
            + (blend_weights["xgb"] * xgb_a_all[idx])
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

        lambda_cache[(h, a, v_country)] = (l_h, l_a, corners_exp, yellows_exp)

    # Convert core dataframes to lightweight dictionaries to completely eliminate iteration overhead
    group_fixtures_list = group_fixtures.to_dict(orient="records")
    knockout_template_list = raw_knockout_template.to_dict(orient="records")

    # 2. Start the Optimized Master Monte Carlo Loop
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

            # Extract venue directly to query our O(1) cache
            venue_country = row["venue_country"]
            l_h, l_a, c_exp, y_exp = lambda_cache[(home, away, venue_country)]

            group_results.append(
                {
                    "match_id": m_id,
                    "group": group,
                    "home_team": home,
                    "away_team": away,
                    "predicted_home_goals": rng.poisson(l_h),
                    "predicted_away_goals": rng.poisson(l_a),
                    "corners": int(np.clip(np.round(rng.normal(c_exp, 1.5)), 4, 18)),
                    "yellow_cards": int(
                        np.clip(np.round(rng.normal(y_exp, 1.2)), 0, 10)
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

            # Extract venue directly to pull from the consensus parameters cache
            venue_country = row["venue_country"]
            l_h, l_a, _, _ = lambda_cache[(home, away, venue_country)]

            h_goals = rng.poisson(l_h)
            a_goals = rng.poisson(l_a)

            if h_goals == a_goals:
                h_goals += rng.poisson(l_h * (ET_MULTIPLIER * FATIGUE_FACTOR))
                a_goals += rng.poisson(l_a * (ET_MULTIPLIER * FATIGUE_FACTOR))

                if h_goals == a_goals:
                    h_elo_stat = elo_engine.get_rating(home)
                    a_elo_stat = elo_engine.get_rating(away)
                    winner = (
                        home
                        if rng.random() < (h_elo_stat / (h_elo_stat + a_elo_stat))
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

    return prob_df, {"lambda_cache": lambda_cache}
