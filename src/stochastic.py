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
):
    """Executes randomized tournament simulations using an optimized global matchup cache."""

    rho_val = match_rules.get("draw_copula", 0.08)
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

    # --- EXPECTED GROUP TABLES (xTABLE) ---
    xtable_ledger = {
        team: {
            "expected_points": 0.0,
            "expected_gd": 0.0,
            "first_place_finishes": 0,
            "second_place_finishes": 0,
            "wildcard_advancements": 0,
            "total_qualifications": 0,
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

    # 2. CALL THE MATCH ENGINE BATCH PROCESSOR
    lambda_cache = batch_evaluate_consensus(
        matchup_keys=matchup_keys,
        ratings=ratings,
        g_home=g_home,
        g_away=g_away,
        g_neutral=g_neutral,
        blend_weights=blend_weights,
        elo_engine=elo_engine,
        xgb_home=xgb_home,
        xgb_away=xgb_away,
        feature_columns=feature_columns,
        latest_team_form=latest_team_form,
    )

    group_fixtures_list = group_fixtures.to_dict(orient="records")
    knockout_template_list = raw_knockout_template.to_dict(orient="records")

    # 3. START THE MONTE CARLO LOOP
    for _ in range(n_simulations):
        group_results = []

        # --- GROUP STAGE SAMPLING ---
        for row in group_fixtures_list:
            m_id, group, home, away = (
                row["match_id"],
                row["group"],
                row["home_team"],
                row["away_team"],
            )
            l_h, l_a, c_exp, y_exp = lambda_cache[(home, away, row["venue_country"])]

            # Use copula draw inflation for group stage
            h_goals, a_goals, _ = simulate_stochastic_match(
                l_h,
                l_a,
                rng,
                match_rules,
                is_knockout=False,
                n_runs=1,
                copula_rho=rho_val,
            )

            group_results.append(
                {
                    "match_id": m_id,
                    "group": group,
                    "home_team": home,
                    "away_team": away,
                    "predicted_home_goals": h_goals[0],
                    "predicted_away_goals": a_goals[0],
                }
            )

        tables = resolve_group_tables(group_results)
        top_thirds = extract_best_third_places(tables)
        third_place_assignments = allocate_third_places(top_thirds)

        # --- RECORD xTABLE POINTS & GOAL DIFFERENCE ---
        for _, row in tables.iterrows():
            team_val = row["team"]
            xtable_ledger[team_val]["expected_points"] += row["points"]
            xtable_ledger[team_val]["expected_gd"] += row["goals_diff"]

        winners = tables[tables["position"] == 1].set_index("group")["team"].to_dict()
        runners = tables[tables["position"] == 2].set_index("group")["team"].to_dict()

        for team in participating_teams:
            pos = tables[tables["team"] == team]["position"].values[0]
            if pos == 1:
                metrics[team]["Round of 32"] += 1
                xtable_ledger[team]["first_place_finishes"] += 1
                xtable_ledger[team]["total_qualifications"] += 1
            elif pos == 2:
                metrics[team]["Round of 32"] += 1
                xtable_ledger[team]["second_place_finishes"] += 1
                xtable_ledger[team]["total_qualifications"] += 1
            elif pos == 3 and team in third_place_assignments.values():
                metrics[team]["Round of 32"] += 1
                xtable_ledger[team]["wildcard_advancements"] += 1
                xtable_ledger[team]["total_qualifications"] += 1
            else:
                metrics[team]["Group Stage Exit"] += 1

        # --- SEQUENTIAL KNOCKOUT WATERFALL SAMPLING ---
        match_winners, match_losers = {}, {}

        for row in knockout_template_list:
            m_id, r_name, slot_home, slot_away = (
                row["match_id"],
                row["round"],
                row["slot_home"],
                row["slot_away"],
            )

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
            h_goals_arr, a_goals_arr, _ = simulate_stochastic_match(
                l_h,
                l_a,
                rng,
                match_rules=match_rules,
                is_knockout=True,
                n_runs=1,
                copula_rho=rho_val,
            )

            winner = home if h_goals_arr[0] > a_goals_arr[0] else away
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

    # 3. GENERATE TARGET DATAFRAMES
    xtable_rows = []
    for team, data in xtable_ledger.items():
        xtable_rows.append(
            {
                "team": team,
                "expected_points": round(data["expected_points"] / n_simulations, 2),
                "expected_gd": round(data["expected_gd"] / n_simulations, 2),
                "group_winner_probability_pct": round(
                    (data["first_place_finishes"] / n_simulations) * 100, 1
                ),
                "group_runner_up_probability_pct": round(
                    (data["second_place_finishes"] / n_simulations) * 100, 1
                ),
                "wildcard_probability_pct": round(
                    (data["wildcard_advancements"] / n_simulations) * 100, 1
                ),
                "total_qualification_pct": round(
                    (data["total_qualifications"] / n_simulations) * 100, 1
                ),
            }
        )
    df_xtables = pd.DataFrame(xtable_rows).sort_values(
        by="expected_points", ascending=False
    )

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
    df_forecast = (
        pd.DataFrame(prob_list)
        .sort_values(by="Champion %", ascending=False)
        .reset_index(drop=True)
    )

    return df_forecast, df_xtables


def precompute_sandbox_matchups(
    all_teams,
    ratings,
    g_home,
    g_away,
    g_neutral,
    blend_weights,
    match_rules,
    elo_engine,
    xgb_home,
    xgb_away,
    feature_columns,
    latest_team_form,
    fat_runs=10000,
    host_nations=None,
):
    """
    Isolated Sandbox Cache Generation.
    Evaluates pairwise permutations on Neutral Turf AND specific Host venues.
    """
    if host_nations is None:
        host_nations = ["United States", "Mexico", "Canada"]

    rng = np.random.default_rng(1989)
    rho_val = match_rules.get("draw_copula", 0.08)

    from itertools import permutations

    matchup_keys = []

    # 1. Base Neutral Permutations (Everyone vs Everyone)
    for h, a in permutations(all_teams, 2):
        matchup_keys.append((h, a, "Neutral"))

    # 2. Host Advantage Permutations (Hosts play everyone at home)
    for host in host_nations:
        if host in all_teams:
            for opp in all_teams:
                if opp != host:
                    matchup_keys.append((host, opp, host))
                    matchup_keys.append((opp, host, host))

    # Clean duplicates just in case
    matchup_keys = list(dict.fromkeys(matchup_keys))

    lambda_cache = batch_evaluate_consensus(
        matchup_keys=matchup_keys,
        ratings=ratings,
        g_home=g_home,
        g_away=g_away,
        g_neutral=g_neutral,
        blend_weights=blend_weights,
        elo_engine=elo_engine,
        xgb_home=xgb_home,
        xgb_away=xgb_away,
        feature_columns=feature_columns,
        latest_team_form=latest_team_form,
    )

    compiled_matchups = []
    for (h, a, venue_country), (l_h, l_a, c_exp, y_exp) in lambda_cache.items():
        grp_h, grp_a, _ = simulate_stochastic_match(
            l_h,
            l_a,
            rng,
            match_rules,
            is_knockout=False,
            n_runs=fat_runs,
            copula_rho=rho_val,
        )

        grp_win_h = float(np.sum(grp_h > grp_a) / fat_runs * 100)
        grp_win_a = float(np.sum(grp_a > grp_h) / fat_runs * 100)
        grp_draw = float(np.sum(grp_h == grp_a) / fat_runs * 100)

        scores = [f"{sh} - {sa}" for sh, sa in zip(grp_h, grp_a)]
        top_scores = Counter(scores).most_common(3)
        top_str = json.dumps(
            [{"score": s, "pct": (c / fat_runs) * 100} for s, c in top_scores]
        )

        _, _, ko_phase = simulate_stochastic_match(
            l_h,
            l_a,
            rng,
            match_rules,
            is_knockout=True,
            n_runs=fat_runs,
            copula_rho=rho_val,
        )

        p_ko_win_90_h = float(np.mean(ko_phase["win_90_h"])) * 100
        p_ko_win_120_h = float(np.mean(ko_phase["win_120_h"])) * 100
        p_ko_win_pen_h = float(np.mean(ko_phase["win_pen_h"])) * 100

        p_ko_win_90_a = float(np.mean(ko_phase["win_90_a"])) * 100
        p_ko_win_120_a = float(np.mean(ko_phase["win_120_a"])) * 100
        p_ko_win_pen_a = float(np.mean(ko_phase["win_pen_a"])) * 100

        ko_win_h = p_ko_win_90_h + p_ko_win_120_h + p_ko_win_pen_h
        ko_win_a = p_ko_win_90_a + p_ko_win_120_a + p_ko_win_pen_a

        compiled_matchups.append(
            {
                "home_team": h,
                "away_team": a,
                "venue_country": venue_country,
                "ensemble_lambda_home": round(l_h, 3),
                "ensemble_lambda_away": round(l_a, 3),
                "grp_win_home": round(grp_win_h, 1),
                "grp_draw": round(grp_draw, 1),
                "grp_win_away": round(grp_win_a, 1),
                "ko_win_90_home_pct": round(p_ko_win_90_h, 1),
                "ko_win_120_home_pct": round(p_ko_win_120_h, 1),
                "ko_win_pen_home_pct": round(p_ko_win_pen_h, 1),
                "ko_win_90_away_pct": round(p_ko_win_90_a, 1),
                "ko_win_120_away_pct": round(p_ko_win_120_a, 1),
                "ko_win_pen_away_pct": round(p_ko_win_pen_a, 1),
                "ko_win_home": round(ko_win_h, 1),
                "ko_win_away": round(ko_win_a, 1),
                "top_scorelines_json": top_str,
            }
        )

    return pd.DataFrame(compiled_matchups)


def build_expected_stochastic_bracket(
    df_xtables, df_sandbox, raw_knockout_template, group_fixtures
):
    """
    Builds the 'Most Likely' probabilistic knockout bracket using xPts
    from the Monte Carlo simulation and >50% threshold stochastic sandbox outcomes.
    """

    # 1. Map teams to groups using the static fixtures
    team_to_group = {}
    for _, row in group_fixtures.iterrows():
        team_to_group[row["home_team"]] = row["group"]
        team_to_group[row["away_team"]] = row["group"]

    df_xt = df_xtables.copy()
    df_xt["group"] = df_xt["team"].map(team_to_group)

    # 2. Sort by expected points and expected GD to determine group ranks
    df_xt = df_xt.sort_values(
        by=["group", "expected_points", "expected_gd"], ascending=[True, False, False]
    )
    df_xt["position"] = df_xt.groupby("group").cumcount() + 1

    winners = df_xt[df_xt["position"] == 1].set_index("group")["team"].to_dict()
    runners = df_xt[df_xt["position"] == 2].set_index("group")["team"].to_dict()

    # 3. Resolve the Top 4 Third-Place Teams using xPts
    thirds = df_xt[df_xt["position"] == 3].copy()

    # Temporarily rename columns to trick the deterministic router into accepting xPts
    top_thirds = (
        thirds.rename(
            columns={"expected_points": "points", "expected_gd": "goals_diff"}
        )
        .sort_values(by=["points", "goals_diff"], ascending=[False, False])
        .head(8)
    )

    third_place_assignments = allocate_third_places(top_thirds)

    # 4. Traverse the bracket using stochastic win probabilities
    match_winners = {}
    match_losers = {}
    bracket_rows = []

    knockout_list = raw_knockout_template.to_dict(orient="records")

    for row in knockout_list:
        m_id = row["match_id"]
        r_name = row["round"]
        v_country = row.get("venue_country", "Neutral")
        slot_home = row["slot_home"]
        slot_away = row["slot_away"]

        # Match Routing Resolution
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

        # If neither team is the host nation, it's structurally neutral turf
        if home != v_country and away != v_country:
            lookup_venue = "Neutral"
        else:
            lookup_venue = v_country

        # Query Sandbox for Absolute Stochastic Edge
        sb_row = df_sandbox[
            (df_sandbox["home_team"] == home)
            & (df_sandbox["away_team"] == away)
            & (df_sandbox["venue_country"] == lookup_venue)
        ]
        is_swapped = False

        if sb_row.empty:
            sb_row = df_sandbox[
                (df_sandbox["home_team"] == away)
                & (df_sandbox["away_team"] == home)
                & (df_sandbox["venue_country"] == lookup_venue)
            ]
            is_swapped = True

        if not sb_row.empty:
            sb_data = sb_row.iloc[0]
            prob_h = (
                float(sb_data["ko_win_away"])
                if is_swapped
                else float(sb_data["ko_win_home"])
            )
            prob_a = (
                float(sb_data["ko_win_home"])
                if is_swapped
                else float(sb_data["ko_win_away"])
            )

            # Retrieve base xG for UI aesthetics
            l_h = (
                sb_data["ensemble_lambda_away"]
                if is_swapped
                else sb_data["ensemble_lambda_home"]
            )
            l_a = (
                sb_data["ensemble_lambda_home"]
                if is_swapped
                else sb_data["ensemble_lambda_away"]
            )
        else:
            prob_h, prob_a = 50.0, 50.0
            l_h, l_a = 1.0, 1.0

        # Determine Absolute Probabilistic Winner
        winner = home if prob_h >= prob_a else away
        loser = away if winner == home else home

        match_winners[m_id] = winner
        match_losers[m_id] = loser

        # Format cosmetic UI goal integers to respect the stochastic winner
        pred_h_goals, pred_a_goals = int(round(l_h)), int(round(l_a))
        if pred_h_goals == pred_a_goals:
            if winner == home:
                pred_h_goals += 1
            else:
                pred_a_goals += 1
        elif pred_h_goals > pred_a_goals and winner == away:
            pred_a_goals = pred_h_goals + 1
        elif pred_a_goals > pred_h_goals and winner == home:
            pred_h_goals = pred_a_goals + 1

        bracket_rows.append(
            {
                "match_id": m_id,
                "round": r_name,
                "venue": row["venue"],
                "venue_country": v_country,
                "home_team": home,
                "away_team": away,
                "predicted_home_goals": pred_h_goals,
                "predicted_away_goals": pred_a_goals,
                "extra_time": False,
                "penalties": False,
                "winner_name_meta": winner,
            }
        )

    return pd.DataFrame(bracket_rows)
