"""
Stochastic Simulation and Probabilistic Forecasting Engine.

Executes optimized Monte Carlo parallel permutations. Converts pre-computed baseline
predictive intensities via Bivariate Copulas to track exact probability distributions
across 'xTables' (xPts/xGD) and tournament survival metrics.
"""

import json
import logging
import os
from collections import Counter
from typing import cast

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
    start_of_tournament: str | None = None,
):
    """Executes randomized tournament simulations caching multi-dimensional pairwise keys natively."""

    rho_val = match_rules.get("draw_copula", 0.08)
    rng = np.random.default_rng(seed=69)

    participating_teams = list(
        set(group_fixtures["home_team"].unique())
        | set(group_fixtures["away_team"].unique())
    )

    # HASH MAP CONFIGURATION: Pre-build an O(1) shootout winner cache
    shootout_cache = {}
    shootouts_path = os.path.join("data", "raw", "shootouts.csv")
    if os.path.exists(shootouts_path):
        try:
            st_df = pd.read_csv(shootouts_path)
            for _, st_row in st_df.iterrows():
                h_team = str(st_row["home_team"]).strip().lower()
                a_team = str(st_row["away_team"]).strip().lower()
                so_winner = str(st_row["winner"]).strip().lower()

                # Cache both permutations without the brittle date dependency
                shootout_cache[(h_team, a_team)] = so_winner
                shootout_cache[(a_team, h_team)] = so_winner
        except (
                    OSError,
                    pd.errors.EmptyDataError,
                    pd.errors.ParserError,
                    KeyError,
                    AttributeError,
                ) as e:
                    logging.debug(
                        "Failed to process shootout cache file '%s': %s",
                        shootouts_path,
                        e,
                    )

    # HASH MAP CONFIGURATION: Pre-build live knockout results cache
    live_ko_cache = {}
    results_path = os.path.join("data", "raw", "results.csv")

    if os.path.exists(results_path):
        try:
            actual_tournament_start = pd.to_datetime("2026-06-11")
            res_df = pd.read_csv(results_path)
            res_df["date"] = pd.to_datetime(res_df["date"])

            # Filter strictly to the active tournament window
            cutoff_date = pd.to_datetime(start_of_tournament) if start_of_tournament else None

            mask = (
                (res_df["tournament"] == "FIFA World Cup")
                & (res_df["date"] >= actual_tournament_start)
                & (pd.notna(res_df["home_score"]))
            )

            if cutoff_date is not None:
                mask &= (res_df["date"] < cutoff_date)

            live_matches = res_df[mask]

            for _, m_row in live_matches.iterrows():
                h_norm = str(m_row["home_team"]).strip().lower()
                a_norm = str(m_row["away_team"]).strip().lower()

                # Store structural score tuples mapped to normalized team name keys
                live_ko_cache[(h_norm, a_norm)] = (
                    int(m_row["home_score"]),
                    int(m_row["away_score"]),
                )
                live_ko_cache[(a_norm, h_norm)] = (
                    int(m_row["away_score"]),
                    int(m_row["home_score"]),
                )
        except (ValueError, TypeError, KeyError, AttributeError) as e:
            logging.debug("Failed to populate live knockout cache: %s", e)

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

    # 1. GENERATE ALL POSSIBLE MATCHUP KEYS AND BATCH EVALUATE
    matchup_keys = []
    unique_venues = list(group_fixtures["venue_country"].unique())
    for v_country in unique_venues:
        for h in participating_teams:
            for a in participating_teams:
                if h != a:
                    matchup_keys.append((h, a, v_country))

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

    # 2. BEGIN MONTE CARLO ITERATION
    for _ in range(n_simulations):
        group_results = []

        # --- GROUP STAGE SIMULATIONS ---
        for row in group_fixtures_list:
            m_id, group, home, away = (
                row["match_id"],
                row["group"],
                row["home_team"],
                row["away_team"],
            )
            # Try to fetch actual match data
            act_h = row.get("actual_home_score")
            act_a = row.get("actual_away_score")

            if (
                pd.notna(act_h)
                and pd.notna(act_a)
                and str(act_h).strip() != ""
                and str(act_a).strip() != ""
            ):
                # Match is completed: absorb the exact scoreline
                sim_h_final = int(float(act_h))
                sim_a_final = int(float(act_a))
            else:
                # Match is unplayed: predict with match_engine
                l_h, l_a, _, _ = lambda_cache[(home, away, row["venue_country"])]

                h_goals, a_goals, _ = simulate_stochastic_match(
                    l_h,
                    l_a,
                    rng,
                    match_rules,
                    is_knockout=False,
                    n_runs=1,
                    copula_rho=rho_val,
                )
                sim_h_final = h_goals[0]
                sim_a_final = a_goals[0]

            group_results.append(
                {
                    "match_id": m_id,
                    "group": group,
                    "home_team": home,
                    "away_team": away,
                    "predicted_home_goals": sim_h_final,
                    "predicted_away_goals": sim_a_final,
                }
            )

        tables = resolve_group_tables(group_results)
        top_thirds = extract_best_third_places(tables)
        third_place_assignments = allocate_third_places(top_thirds)

        # Record Expected Points (xPts)
        for row in tables.itertuples():
            team_val = str(row.team)
            xtable_ledger[team_val]["expected_points"] += float(cast(float, row.points))
            xtable_ledger[team_val]["expected_gd"] += float(cast(float, row.goals_diff))

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

        match_winners, match_losers = {}, {}

        # --- KNOCKOUT STAGE SIMULATIONS ---
        for row in knockout_template_list:
            m_id, r_name, slot_home, slot_away = (
                row["match_id"],
                row["round"],
                row["slot_home"],
                row["slot_away"],
            )

            def _resolve_team(
                slot,
                m_id=m_id,
                winners=winners,
                runners=runners,
                third_place_assignments=third_place_assignments,
                match_winners=match_winners,
                match_losers=match_losers,
            ):
                """Parse textual placeholder tags to route physical entities into the node."""
                if "Winner Group" in slot:
                    return winners[slot.replace("Winner Group ", "").strip()]
                if "Runner-up Group" in slot:
                    return runners[slot.replace("Runner-up Group ", "").strip()]
                if "Best 3rd" in slot:
                    return third_place_assignments[m_id]
                if "Winner Match" in slot:
                    return match_winners[int(slot.replace("Winner Match ", "").strip())]
                if "Loser Match" in slot:
                    return match_losers[int(slot.replace("Loser Match ", "").strip())]
                return slot

            home = _resolve_team(slot_home)
            away = _resolve_team(slot_away)

            # Normalize strings cleanly prior to O(1) cache validation queries
            norm_home = str(home).strip().lower()
            norm_away = str(away).strip().lower()

            # Try to fetch actual match data
            act_h = row.get("actual_home_score")
            act_a = row.get("actual_away_score")

            # Query the cache maps using the normalized string tokens
            if (
                pd.isna(act_h) or pd.isna(act_a) or str(act_h).strip() == ""
            ) and (norm_home, norm_away) in live_ko_cache:
                # Unpack exactly 2 items to match the cache definition
                act_h, act_a = live_ko_cache[(norm_home, norm_away)]

            if (
                pd.notna(act_h)
                and pd.notna(act_a)
                and str(act_h).strip() != ""
                and str(act_a).strip() != ""
            ):
                final_h_sim = int(float(act_h))
                final_a_sim = int(float(act_a))

                if final_h_sim > final_a_sim:
                    winner, loser = home, away
                elif final_a_sim > final_h_sim:
                    winner, loser = away, home
                else:
                    # Query the 2-tuple shootout layout without the ghost date parameter
                    shootout_winner = shootout_cache.get((norm_home, norm_away))
                    if shootout_winner == norm_home:
                        winner, loser = home, away
                    elif shootout_winner == norm_away:
                        winner, loser = away, home
                    else:
                        if elo_engine.get_rating(home) >= elo_engine.get_rating(away):
                            winner, loser = home, away
                        else:
                            winner, loser = away, home
            else:
                # Match is unplayed: predict with match_engine
                l_h, l_a, _, _ = lambda_cache[(home, away, row["venue_country"])]

                h_goals_arr, a_goals_arr, _ = simulate_stochastic_match(
                    l_h,
                    l_a,
                    rng,
                    match_rules=match_rules,
                    is_knockout=True,
                    n_runs=1,
                    copula_rho=rho_val,
                )

                h_g = int(h_goals_arr[0])
                a_g = int(a_goals_arr[0])

                if h_g > a_g:
                    winner, loser = home, away
                elif a_g > h_g:
                    winner, loser = away, home
                else:
                    # DRAW DETECTED: Trigger Shrunk Penalty Shootout Model
                    elo_h = elo_engine.get_rating(home)
                    elo_a = elo_engine.get_rating(away)

                    # Compute standard Elo win expectation
                    p_raw = 1.0 / (1.0 + 10.0 ** ((elo_a - elo_h) / 400.0))

                    # Apply an 85% shrinkage coefficient pulling the probability toward 50/50
                    gamma = 0.85
                    p_shootout = 0.5 + (1.0 - gamma) * (p_raw - 0.5)

                    # Roll the dice against the shrunk distribution
                    if rng.random() < p_shootout:
                        winner, loser = home, away
                    else:
                        winner, loser = away, home

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

    # GENERATE TARGET DATAFRAMES
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
    """Isolates all pairwise permutations into an explicit Sandbox UI cache limit."""

    host_nations = host_nations or ["United States", "Mexico", "Canada"]
    rng = np.random.default_rng(1989)
    rho_val = match_rules.get("draw_copula", 0.08)

    from itertools import permutations

    matchup_keys = [(h, a, "Neutral") for h, a in permutations(all_teams, 2)]
    for host in [h for h in host_nations if h in all_teams]:
        matchup_keys.extend([(host, opp, host) for opp in all_teams if opp != host])
        matchup_keys.extend([(opp, host, host) for opp in all_teams if opp != host])

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
    for (h, a, venue_country), (l_h, l_a, _, _) in lambda_cache.items():
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
    df_xtables, df_sandbox, raw_knockout_template, group_fixtures, start_of_tournament: str | None = None
):
    """Builds a 'Most Likely' UI probabilistic bracket mapping directly from xPts logic.

    Hardened to seamlessly integrate real-world match outcomes and shootout caches.
    """

    team_to_group = {}
    for _, row in group_fixtures.iterrows():
        team_to_group[row["home_team"]] = row["group"]
        team_to_group[row["away_team"]] = row["group"]

    df_xt = df_xtables.copy()
    df_xt["group"] = df_xt["team"].map(team_to_group)
    df_xt = df_xt.sort_values(
        by=[
            "group",
            "group_winner_probability_pct",
            "group_runner_up_probability_pct",
            "expected_points",
        ],
        ascending=[True, False, False, False],
    )
    df_xt["position"] = df_xt.groupby("group").cumcount() + 1

    winners = df_xt[df_xt["position"] == 1].set_index("group")["team"].to_dict()
    runners = df_xt[df_xt["position"] == 2].set_index("group")["team"].to_dict()

    thirds = df_xt[df_xt["position"] == 3].copy()
    top_thirds = thirds.sort_values(
        by=["wildcard_probability_pct", "expected_points", "expected_gd"],
        ascending=[False, False, False],
    ).head(8)

    top_thirds = top_thirds.rename(
        columns={"expected_points": "points", "expected_gd": "goals_diff"}
    )
    third_place_assignments = allocate_third_places(top_thirds)

    # FACTUAL CAPTURE LAYER: Ingest historical knockout data templates
    shootout_cache = {}
    shootouts_path = os.path.join("data", "raw", "shootouts.csv")
    if os.path.exists(shootouts_path):
        try:
            st_df = pd.read_csv(shootouts_path)
            for _, st_row in st_df.iterrows():
                h_team = str(st_row["home_team"]).strip().lower()
                a_team = str(st_row["away_team"]).strip().lower()
                so_winner = str(st_row["winner"]).strip().lower()
                shootout_cache[(h_team, a_team)] = so_winner
                shootout_cache[(a_team, h_team)] = so_winner
        except (
                    OSError,
                    pd.errors.EmptyDataError,
                    pd.errors.ParserError,
                    KeyError,
                    AttributeError,
                ) as e:
                    logging.debug(
                        "Failed to parse shootout cache file '%s': %s",
                        shootouts_path,
                        e,
                    )

    live_ko_cache = {}
    results_path = os.path.join("data", "raw", "results.csv")
    if os.path.exists(results_path):
        try:
            actual_tournament_start = pd.to_datetime("2026-06-11")
            res_df = pd.read_csv(results_path)
            res_df["date"] = pd.to_datetime(res_df["date"])

            cutoff_date = pd.to_datetime(start_of_tournament) if start_of_tournament else None

            mask = (
                (res_df["tournament"] == "FIFA World Cup")
                & (res_df["date"] >= actual_tournament_start)
                & (pd.notna(res_df["home_score"]))
            )

            if cutoff_date is not None:
                mask &= (res_df["date"] < cutoff_date)

            live_matches = res_df[mask]

            for _, m_row in live_matches.iterrows():
                h_norm = str(m_row["home_team"]).strip().lower()
                a_norm = str(m_row["away_team"]).strip().lower()
                live_ko_cache[(h_norm, a_norm)] = (
                    int(m_row["home_score"]),
                    int(m_row["away_score"]),
                )
                live_ko_cache[(a_norm, h_norm)] = (
                    int(m_row["away_score"]),
                    int(m_row["home_score"]),
                )
        except (ValueError, TypeError, KeyError, AttributeError) as e:
            logging.debug("Failed to populate live knockout cache: %s", e)

    match_winners, match_losers, bracket_rows = {}, {}, []
    knockout_list = raw_knockout_template.to_dict(orient="records")

    for row in knockout_list:
        m_id = row["match_id"]
        r_name = row["round"]
        v_country = row.get("venue_country", "Neutral")
        slot_home = row["slot_home"]
        slot_away = row["slot_away"]

        def _resolve_team(
            slot,
            m_id=m_id,
            winners=winners,
            runners=runners,
            third_place_assignments=third_place_assignments,
            match_winners=match_winners,
            match_losers=match_losers,
        ):
            if "Winner Group" in slot:
                return winners[slot.replace("Winner Group ", "").strip()]
            if "Runner-up Group" in slot:
                return runners[slot.replace("Runner-up Group ", "").strip()]
            if "Best 3rd" in slot:
                return third_place_assignments[m_id]
            if "Winner Match" in slot:
                return match_winners[int(slot.replace("Winner Match ", "").strip())]
            if "Loser Match" in slot:
                return match_losers[int(slot.replace("Loser Match ", "").strip())]
            return slot

        home = _resolve_team(slot_home)
        away = _resolve_team(slot_away)

        norm_home = str(home).strip().lower()
        norm_away = str(away).strip().lower()

        # Check for real-world score overrides
        act_h = row.get("actual_home_score")
        act_a = row.get("actual_away_score")

        if (
            pd.isna(act_h) or pd.isna(act_a) or str(act_h).strip() == ""
        ) and (norm_home, norm_away) in live_ko_cache:
            act_h, act_a = live_ko_cache[(norm_home, norm_away)]

        if (
            pd.notna(act_h)
            and pd.notna(act_a)
            and str(act_h).strip() != ""
            and str(act_a).strip() != ""
        ):
            # Match is historical reality: Pull true data values
            pred_h_goals = int(float(act_h))
            pred_a_goals = int(float(act_a))

            if pred_h_goals > pred_a_goals:
                winner = home
                is_et, is_pen = False, False
            elif pred_a_goals > pred_h_goals:
                winner = away
                is_et, is_pen = False, False
            else:
                # Tie detected: extract penalty shootout record
                is_et, is_pen = True, True
                shootout_winner = shootout_cache.get((norm_home, norm_away))
                if shootout_winner == norm_home:
                    winner = home
                elif shootout_winner == norm_away:
                    winner = away
                else:
                    winner = home  # Safe fallback
        else:
            # Match is unplayed: fallback to sandbox probabilistic lookups
            is_et, is_pen = False, False
            if home != v_country and away != v_country:
                lookup_venue = "Neutral"
            else:
                lookup_venue = v_country

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

            winner = home if prob_h >= prob_a else away

            pred_h_goals, pred_a_goals = round(l_h), round(l_a)
            if pred_h_goals == pred_a_goals:
                if winner == home:
                    pred_h_goals += 1
                else:
                    pred_a_goals += 1
            elif pred_h_goals > pred_a_goals and winner == away:
                pred_a_goals = pred_h_goals + 1
            elif pred_a_goals > pred_h_goals and winner == home:
                pred_h_goals = pred_a_goals + 1

        match_winners[m_id], match_losers[m_id] = (
            winner,
            away if winner == home else home,
        )

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
                "extra_time": is_et,
                "penalties": is_pen,
                "winner_name_meta": winner,
            }
        )

    return pd.DataFrame(bracket_rows)
