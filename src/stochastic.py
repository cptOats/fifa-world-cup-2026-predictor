"""Stochastic Simulation and Probabilistic Forecasting Engine."""

import json
from collections import Counter

import numpy as np
import pandas as pd

from src.match_engine import batch_evaluate_consensus, simulate_stochastic_match
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
    match_rules,
    n_simulations=10000,
    use_prior_nudge=False,
    nudge_strength=1.5,
    team_power=None,
):
    """Executes randomized tournament simulations using an optimized global matchup cache."""

    rng = np.random.default_rng(seed=69)

    participating_teams = list(
        set(group_fixtures["home_team"].unique())
        | set(group_fixtures["away_team"].unique())
    )

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

    # 1. GENERATE ALL POSSIBLE MATCHUP KEYS
    matchup_keys = []
    unique_venues = list(group_fixtures["venue_country"].unique())
    for v_country in unique_venues:
        for h in participating_teams:
            for a in participating_teams:
                if h != a:
                    matchup_keys.append((h, a, v_country))

    # 2. CALL THE CENTRALIZED MATCH ENGINE BATCH PROCESSOR
    lambda_cache = batch_evaluate_consensus(
        matchup_keys=matchup_keys,
        ratings=ratings,
        g_home=g_home,
        g_away=g_away,
        g_neutral=g_neutral,
        blend_weights=blend_weights,
        elo_engine=elo_engine,
        match_rules=match_rules,
        xgb_home=xgb_home,
        xgb_away=xgb_away,
        feature_columns=feature_columns,
        latest_team_form=latest_team_form,
        use_prior_nudge=use_prior_nudge,
        nudge_strength=nudge_strength,
        team_power=team_power,
    )

    group_fixtures_list = group_fixtures.to_dict(orient="records")
    knockout_template_list = raw_knockout_template.to_dict(orient="records")

    # 3. START THE OPTIMIZED MASTER MONTE CARLO LOOP
    for _ in range(n_simulations):
        group_results = []

        # --- PHASE A: GROUP STAGE SAMPLING ---
        for row in group_fixtures_list:
            m_id, group, home, away = (
                row["match_id"],
                row["group"],
                row["home_team"],
                row["away_team"],
            )
            l_h, l_a, c_exp, y_exp = lambda_cache[(home, away, row["venue_country"])]

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

        tables = resolve_group_tables(group_results)
        top_thirds = extract_best_third_places(tables)
        third_place_assignments = allocate_third_places(top_thirds)

        winners = tables[tables["position"] == 1].set_index("group")["team"].to_dict()
        runners = tables[tables["position"] == 2].set_index("group")["team"].to_dict()

        for team in participating_teams:
            pos = tables[tables["team"] == team]["position"].values[0]
            if (
                pos == 1
                or pos == 2
                or (pos == 3 and team in third_place_assignments.values())
            ):
                metrics[team]["Round of 32"] += 1
            else:
                metrics[team]["Group Stage Exit"] += 1

        # --- PHASE B: SEQUENTIAL KNOCKOUT WATERFALL SAMPLING ---
        match_winners, match_losers = {}, {}

        for row in knockout_template_list:
            m_id, r_name, slot_home, slot_away = (
                row["match_id"],
                row["round"],
                row["slot_home"],
                row["slot_away"],
            )

            # (Omitted dict mapping for brevity, same as your original resolution logic)
            home = (
                winners[slot_home.replace("Winner Group ", "").strip()]
                if "Winner Group" in slot_home
                else runners[slot_home.replace("Runner-up Group ", "").strip()]
                if "Runner-up Group" in slot_home
                else third_place_assignments[m_id]
                if "Best 3rd" in slot_home
                else match_winners[int(slot_home.replace("Winner Match ", "").strip())]
                if "Winner Match" in slot_home
                else match_losers[int(slot_home.replace("Loser Match ", "").strip())]
                if "Loser Match" in slot_home
                else slot_home
            )

            away = (
                winners[slot_away.replace("Winner Group ", "").strip()]
                if "Winner Group" in slot_away
                else runners[slot_away.replace("Runner-up Group ", "").strip()]
                if "Runner-up Group" in slot_away
                else third_place_assignments[m_id]
                if "Best 3rd" in slot_away
                else match_winners[int(slot_away.replace("Winner Match ", "").strip())]
                if "Winner Match" in slot_away
                else match_losers[int(slot_away.replace("Loser Match ", "").strip())]
                if "Loser Match" in slot_away
                else slot_away
            )

            l_h, l_a, _, _ = lambda_cache[(home, away, row["venue_country"])]

            # CENTRALIZED STOCHASTIC MATCH RESOLUTION
            h_goals_arr, a_goals_arr = simulate_stochastic_match(
                l_h,
                l_a,
                elo_engine.get_rating(home),
                elo_engine.get_rating(away),
                rng,
                match_rules=match_rules,
                is_knockout=True,
                n_runs=1,
            )

            winner = home if h_goals_arr[0] > a_goals_arr[0] else away
            loser = away if winner == home else home

            match_winners[m_id] = winner
            match_losers[m_id] = loser

            # Tracking stage progression
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

    # 4. COMPILE MASTER PROBABILITY TABLE
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

    # 5. PRE-COMPUTE FAT ARTIFACT FOR THE DASHBOARD SANDBOX
    compiled_matchups = []
    fat_runs = 10000

    for (h, a, cache_neutral), (l_h, l_a, c_exp, y_exp) in lambda_cache.items():
        elo_h = elo_engine.get_rating(h)
        elo_a = elo_engine.get_rating(a)

        # Vectorized Group Stage Probabilities
        grp_h, grp_a = simulate_stochastic_match(
            l_h, l_a, elo_h, elo_a, rng, match_rules, is_knockout=False, n_runs=fat_runs
        )

        grp_win_h = np.sum(grp_h > grp_a) / fat_runs * 100
        grp_win_a = np.sum(grp_a > grp_h) / fat_runs * 100
        grp_draw = np.sum(grp_h == grp_a) / fat_runs * 100

        scores = [f"{sh} - {sa}" for sh, sa in zip(grp_h, grp_a)]
        top_scores = Counter(scores).most_common(3)
        top_str = json.dumps(
            [{"score": s, "pct": (c / fat_runs) * 100} for s, c in top_scores]
        )

        # Vectorized Knockout Stage Probabilities (Automated ET/Pens resolution)
        ko_h, ko_a = simulate_stochastic_match(
            l_h, l_a, elo_h, elo_a, rng, match_rules, is_knockout=True, n_runs=fat_runs
        )

        ko_win_h = np.sum(ko_h > ko_a) / fat_runs * 100
        ko_win_a = np.sum(ko_a > ko_h) / fat_runs * 100

        compiled_matchups.append(
            {
                "home_team": h,
                "away_team": a,
                "is_neutral_venue": cache_neutral,
                "ensemble_lambda_home": l_h,
                "ensemble_lambda_away": l_a,
                "grp_win_home": grp_win_h,
                "grp_draw": grp_draw,
                "grp_win_away": grp_win_a,
                "ko_win_home": ko_win_h,
                "ko_win_away": ko_win_a,
                "top_scorelines_json": top_str,
            }
        )

    return prob_df, {"fat_matchups": compiled_matchups}
