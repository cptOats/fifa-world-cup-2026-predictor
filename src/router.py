"""
Tournament Group Standings and Knockout Stage Routing Engine.

Evaluates multi-key ascending/descending pandas sorts to dictate group standings.
It uses a recursive backtracking search to accurately map the complex logic behind
the official 8 "Best Third Place" wildcard slots into the Round of 32 constraints.
"""

import json
import logging
import os

import pandas as pd

from src.match_engine import evaluate_match_consensus, simulate_deterministic_match


def resolve_group_tables(predicted_fixtures_list_or_df):
    """Ultra-fast, native Python implementation of the 2026 Group Stage Resolver."""

    is_df = isinstance(predicted_fixtures_list_or_df, pd.DataFrame)
    records = (
        predicted_fixtures_list_or_df.to_dict("records")
        if is_df
        else predicted_fixtures_list_or_df
    )

    # 1. Base table compilation using native high-speed dictionaries
    teams_data = {}
    group_matches = {}

    for row in records:
        grp = row["group"]
        home = row["home_team"]
        away = row["away_team"]
        h_score = row["predicted_home_goals"]
        a_score = row["predicted_away_goals"]

        if grp not in group_matches:
            group_matches[grp] = []
        group_matches[grp].append(row)

        for team in (home, away):
            if team not in teams_data:
                teams_data[team] = {
                    "group": grp,
                    "team": team,
                    "points": 0,
                    "goals_for": 0,
                    "goals_against": 0,
                    "goals_diff": 0,
                    "h2h_pts": 0,
                    "h2h_gd": 0,
                    "h2h_gf": 0,
                }

        teams_data[home]["goals_for"] += h_score
        teams_data[home]["goals_against"] += a_score
        teams_data[away]["goals_for"] += a_score
        teams_data[away]["goals_against"] += h_score

        teams_data[home]["goals_diff"] += h_score - a_score
        teams_data[away]["goals_diff"] += a_score - h_score

        if h_score > a_score:
            teams_data[home]["points"] += 3
        elif a_score > h_score:
            teams_data[away]["points"] += 3
        else:
            teams_data[home]["points"] += 1
            teams_data[away]["points"] += 1

    # 2. Native Head-to-Head processing ONLY for point deadlocks
    for grp, matches in group_matches.items():
        pts_map = {}
        for team, data in teams_data.items():
            if data["group"] == grp:
                p = data["points"]
                if p not in pts_map:
                    pts_map[p] = []
                pts_map[p].append(team)

        for p, tied_teams in pts_map.items():
            if len(tied_teams) > 1:
                # If tied, simulate the mini-league between the deadlocked teams
                for m in matches:
                    if m["home_team"] in tied_teams and m["away_team"] in tied_teams:
                        h, a = m["home_team"], m["away_team"]
                        hg, ag = m["predicted_home_goals"], m["predicted_away_goals"]

                        teams_data[h]["h2h_gf"] += hg
                        teams_data[a]["h2h_gf"] += ag
                        teams_data[h]["h2h_gd"] += hg - ag
                        teams_data[a]["h2h_gd"] += ag - hg

                        if hg > ag:
                            teams_data[h]["h2h_pts"] += 3
                        elif ag > hg:
                            teams_data[a]["h2h_pts"] += 3
                        else:
                            teams_data[h]["h2h_pts"] += 1
                            teams_data[a]["h2h_pts"] += 1

    # 3. Compile and execute the 2026 multi-key tuple sort natively
    compiled_list = list(teams_data.values())
    compiled_list.sort(
        key=lambda x: (
            x["group"],
            -x["points"],
            -x["h2h_pts"],
            -x["h2h_gd"],
            -x["h2h_gf"],
            -x["goals_diff"],
            -x["goals_for"],
        )
    )

    # 4. Assign positions
    current_group = None
    pos = 1
    for row in compiled_list:
        if row["group"] != current_group:
            current_group = row["group"]
            pos = 1
        row["position"] = pos
        pos += 1

    # 5. Return as a single DataFrame to maintain downstream stochastic contracts
    return pd.DataFrame(compiled_list)


def extract_best_third_places(group_tables_df):
    """Isolates and returns the top 8 advancing wildcards from all 3rd place finishes."""

    third_places = group_tables_df[group_tables_df["position"] == 3].copy()
    ranked_thirds = third_places.sort_values(
        by=["points", "goals_diff", "goals_for"], ascending=[False, False, False]
    ).reset_index(drop=True)

    return ranked_thirds.head(8).copy()


def allocate_third_places(advancing_thirds_df):
    """Maps qualifying third-place teams using the official 495-row FIFA Annex C table."""

    # Load the file dynamically inside the function
    annex_c_path = os.path.join("data", "raw", "annex_c_matrix.json")
    try:
        with open(annex_c_path, "r") as f:
            matrix_lookup = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            "🛑 CRITICAL RUNTIME ERROR: FIFA Annex C matrix file is missing from data/raw/annex_c_matrix.json. "
            "Ensure your ingestion layer executes completely before routing."
        )

    # Create the sorted 8-character string (e.g., 'BDEFIJKL')
    teams = list(zip(advancing_thirds_df["group"], advancing_thirds_df["team"]))
    qualified_groups_key = "".join(sorted([g for g, _ in teams]))

    # O(1) Hash Map Lookup into the official 495-row table
    if qualified_groups_key in matrix_lookup:
        matrix_slots = matrix_lookup[qualified_groups_key]
        group_to_team = {g: t for g, t in teams}

        # Convert the group letters back to physical country names for the bracket
        # Cast match JSON keys back to integers for contract alignment
        return {
            int(match_id): group_to_team[grp] for match_id, grp in matrix_slots.items()
        }
    else:
        raise ValueError(
            f"CRITICAL ERROR: Combination {qualified_groups_key} not found in FIFA Annex C! "
            f"Verify your raw JSON compilation output."
        )


def generate_round_of_32_draw(
    group_tables_df: pd.DataFrame, third_place_mapping: dict[int, str]
) -> pd.DataFrame:
    """Reads the template layout and substitutes placeholders with actual country names for the Round of 32."""
    processed_dir = os.path.join("data", "processed")
    knockout_template = pd.read_csv(
        os.path.join(processed_dir, "clean_knockout_slots.csv")
    )
    r32_df = knockout_template[knockout_template["round"] == "Round of 32"].copy()

    winners = (
        group_tables_df[group_tables_df["position"] == 1]
        .set_index("group")["team"]
        .to_dict()
    )
    runners_up = (
        group_tables_df[group_tables_df["position"] == 2]
        .set_index("group")["team"]
        .to_dict()
    )

    home_teams = []
    away_teams = []

    for _, row in r32_df.iterrows():
        match_id = int(row["match_id"])
        slot_home = row["slot_home"]
        slot_away = row["slot_away"]

        def _resolve_team(slot):
            """Parse textual placeholder tags to route the physical entities into the node."""

            if "Winner Group" in slot:
                return winners[slot.replace("Winner Group ", "").strip()]
            if "Runner-up Group" in slot:
                return runners_up[slot.replace("Runner-up Group ", "").strip()]
            if "Best 3rd" in slot:
                return third_place_mapping[match_id]
            return slot

        home_teams.append(_resolve_team(slot_home))
        away_teams.append(_resolve_team(slot_away))

    r32_df["home_team"] = home_teams
    r32_df["away_team"] = away_teams

    return r32_df[["match_id", "round", "venue", "home_team", "away_team"]].reset_index(
        drop=True
    )


def simulate_deterministic_group_stage(
    group_fixtures,
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
):
    """Executes deterministic iteration across all group stage blueprints."""

    group_results = []

    for _, row in group_fixtures.iterrows():
        match_id = int(row["match_id"])
        group_letter = row["group"]
        home = row["home_team"]
        away = row["away_team"]
        venue_country = row["venue_country"]
        venue = row.get("venue", "Neutral")

        # Extract real-world overrides if present
        act_h = row.get("actual_home_score")
        act_a = row.get("actual_away_score")

        if (
            pd.notna(act_h)
            and pd.notna(act_a)
            and str(act_h).strip() != ""
            and str(act_a).strip() != ""
        ):
            # Match is completed: absorb the exact scoreline
            final_home = int(float(act_h))
            final_away = int(float(act_a))
            winner_side = (
                "home"
                if final_home > final_away
                else ("away" if final_away > final_home else "draw")
            )
            t_corn, t_yell, t_red = (
                None,
                None,
                None,
            )  # Explicitly null out proxy metric stats for completed matches
        else:
            # Match is unplayed: predict with match_engine
            raw_home, raw_away, p_corners, p_yellows, p_reds = evaluate_match_consensus(
                home_team=home,
                away_team=away,
                venue_country=venue_country,
                ratings=ratings,
                g_home_avg=g_home,
                g_away_avg=g_away,
                g_neutral_avg=g_neutral,
                blend_weights=blend_weights,
                elo_engine=elo_engine,
                xgb_home=xgb_home,
                xgb_away=xgb_away,
                feature_columns=feature_columns,
                latest_team_form=latest_team_form,
            )

            final_home, final_away, winner_side, t_corn, t_yell, t_red, _, _ = (
                simulate_deterministic_match(
                    raw_home,
                    raw_away,
                    p_corners,
                    p_yellows,
                    p_reds,
                    match_rules,
                    is_knockout=False,
                )
            )

        group_results.append(
            {
                "match_id": match_id,
                "group": group_letter,
                "home_team": home,
                "away_team": away,
                "predicted_home_goals": final_home,
                "predicted_away_goals": final_away,
                "corners": t_corn,
                "yellow_cards": t_yell,
                "red_cards": t_red,
                "winning_team": winner_side,
                "venue": venue,
                "venue_country": venue_country,
            }
        )

    predicted_fixtures = pd.DataFrame(group_results)
    group_tables = resolve_group_tables(predicted_fixtures)
    top_thirds = extract_best_third_places(group_tables)
    third_place_assignments = allocate_third_places(top_thirds)

    predicted_fixtures["round"] = "Group " + predicted_fixtures["group"]
    predicted_fixtures["extra_time"] = False
    predicted_fixtures["penalties"] = False
    predicted_fixtures["winner_name_meta"] = predicted_fixtures.apply(
        lambda r: (
            r["home_team"]
            if r["winning_team"] == "home"
            else (r["away_team"] if r["winning_team"] == "away" else "Draw")
        ),
        axis=1,
    )

    return predicted_fixtures, group_tables, third_place_assignments


def simulate_knockout_waterfall(
    group_tables_df: pd.DataFrame,
    third_place_mapping: dict[int, str],
    ratings: dict[str, dict[str, float]],
    g_home_avg: float,
    g_away_avg: float,
    g_neutral_avg: float,
    blend_weights: dict[str, float],
    match_rules,
    elo_engine=None,
    xgb_home=None,
    xgb_away=None,
    feature_columns: list[str] | None = None,
    latest_team_form: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    """Simulates the knockout bracket tree sequentially, seamlessly prioritizing real-world match overrides and shootout resolutions."""

    if feature_columns is None or latest_team_form is None or elo_engine is None:
        raise ValueError(
            "Type Guard: feature_columns list cannot be None inside the routing layer."
        )

    processed_dir = os.path.join("data", "processed")
    knockout_template = pd.read_csv(
        os.path.join(processed_dir, "clean_knockout_slots.csv")
    )

    # HASH MAP CONFIGURATION: Pre-build an O(1) shootout winner cache
    shootout_cache = {}
    shootouts_path = os.path.join("data", "raw", "shootouts.csv")
    if os.path.exists(shootouts_path):
        try:
            st_df = pd.read_csv(shootouts_path)
            for _, st_row in st_df.iterrows():
                st_date = str(st_row["date"]).strip()
                h_team = str(st_row["home_team"]).strip()
                a_team = str(st_row["away_team"]).strip()
                so_winner = str(st_row["winner"]).strip()

                # Cache both team permutations anchored to the exact match date
                shootout_cache[(st_date, h_team, a_team)] = so_winner
                shootout_cache[(st_date, a_team, h_team)] = so_winner
        except Exception:
            pass

    match_winners, match_losers = {}, {}
    winners = (
        group_tables_df[group_tables_df["position"] == 1]
        .set_index("group")["team"]
        .to_dict()
    )
    runners_up = (
        group_tables_df[group_tables_df["position"] == 2]
        .set_index("group")["team"]
        .to_dict()
    )
    knockout_results = []

    for _, row in knockout_template.iterrows():
        match_id = int(row["match_id"])
        r_name = row["round"]
        venue = row["venue"]
        slot_home = row["slot_home"]
        slot_away = row["slot_away"]
        venue_country = row["venue_country"]

        def _resolve_team(slot):
            """Parse textual placeholder tags to route the physical entities into the node."""
            if "Winner Group" in slot:
                return winners[slot.replace("Winner Group ", "").strip()]
            if "Runner-up Group" in slot:
                return runners_up[slot.replace("Runner-up Group ", "").strip()]
            if "Best 3rd" in slot:
                return third_place_mapping[match_id]
            if "Winner Match" in slot:
                return match_winners[int(slot.replace("Winner Match ", "").strip())]
            if "Loser Match" in slot:
                return match_losers[int(slot.replace("Loser Match ", "").strip())]
            return slot

        home_team, away_team = _resolve_team(slot_home), _resolve_team(slot_away)

        # Look for real-world overrides from disk template
        act_h = row.get("actual_home_score")
        act_a = row.get("actual_away_score")

        # Track the exact date string of the historical match being processed
        match_date = None
        if pd.isna(act_h) or pd.isna(act_a) or str(act_h).strip() == "":
            results_path = os.path.join("data", "raw", "results.csv")
            parquet_path = os.path.join(
                "data", "processed", "clean_historical_matches.parquet"
            )

            if os.path.exists(results_path) and os.path.exists(parquet_path):
                max_hist_date = pd.to_datetime(
                    pd.read_parquet(parquet_path)["date"]
                ).max()
                actual_tournament_start = pd.to_datetime("2026-06-11")
                res_df = pd.read_csv(results_path)
                res_df["date"] = pd.to_datetime(res_df["date"])

                match_lookup = res_df[
                    (res_df["tournament"] == "FIFA World Cup")
                    & (res_df["date"] >= actual_tournament_start)
                    & (res_df["date"] <= max_hist_date)
                    & (
                        (
                            (res_df["home_team"] == home_team)
                            & (res_df["away_team"] == away_team)
                        )
                        | (
                            (res_df["home_team"] == away_team)
                            & (res_df["away_team"] == home_team)
                        )
                    )
                ]

                if not match_lookup.empty:
                    m_row = match_lookup.iloc[0]
                    # Capture the exact date string from the verified historical row
                    match_date = str(m_row["date"].strftime("%Y-%m-%d")).strip()
                    if m_row["home_team"] == home_team:
                        act_h, act_a = m_row["home_score"], m_row["away_score"]
                    else:
                        act_h, act_a = m_row["away_score"], m_row["home_score"]

        if (
            pd.notna(act_h)
            and pd.notna(act_a)
            and str(act_h).strip() != ""
            and str(act_a).strip() != ""
        ):
            # Match is completed: absorb the exact scoreline
            f_home = int(float(act_h))
            f_away = int(float(act_a))

            if f_home > f_away:
                winner_side = "home"
                is_et, is_pen = False, False
            elif f_away > f_home:
                winner_side = "away"
                is_et, is_pen = False, False
            else:
                # TIE RESOLUTION: It's an ET draw, consult the fast shootout cache map
                is_et, is_pen = True, True
                shootout_winner = shootout_cache.get((home_team, away_team))

                if shootout_winner == home_team:
                    winner_side = "home"
                elif shootout_winner == away_team:
                    winner_side = "away"
                else:
                    # Ultimate fallback safety if shootout row is completely missing from Kaggle
                    logging.warning(
                        f"⚠️ Shootout Missing: Draw recorded for {home_team} vs {away_team} on {match_date}, "
                        "but no shootout entry matched this specific date. Defaulting to Elo advantage."
                    )
                    winner_side = (
                        "home"
                        if elo_engine.get_rating(home_team)
                        >= elo_engine.get_rating(away_team)
                        else "away"
                    )

            t_corn, t_yell, t_red = (
                None,
                None,
                None,
            )  # Explicitly null out proxy metric stats for completed matches
        else:
            # Match is unplayed: predict with match_engine
            raw_home, raw_away, p_corners, p_yellows, p_reds = evaluate_match_consensus(
                home_team=home_team,
                away_team=away_team,
                venue_country=venue_country,
                ratings=ratings,
                g_home_avg=g_home_avg,
                g_away_avg=g_away_avg,
                g_neutral_avg=g_neutral_avg,
                blend_weights=blend_weights,
                elo_engine=elo_engine,
                xgb_home=xgb_home,
                xgb_away=xgb_away,
                feature_columns=feature_columns,
                latest_team_form=latest_team_form,
            )

            f_home, f_away, winner_side, t_corn, t_yell, t_red, is_et, is_pen = (
                simulate_deterministic_match(
                    raw_home,
                    raw_away,
                    p_corners,
                    p_yellows,
                    p_reds,
                    match_rules,
                    is_knockout=True,
                )
            )

        advance_winner = home_team if winner_side == "home" else away_team
        advance_loser = away_team if winner_side == "home" else home_team

        match_winners[match_id] = advance_winner
        match_losers[match_id] = advance_loser

        knockout_results.append(
            {
                "match_id": match_id,
                "round": r_name,
                "home_team": home_team,
                "away_team": away_team,
                "predicted_home_goals": f_home,
                "predicted_away_goals": f_away,
                "corners": t_corn,
                "yellow_cards": t_yell,
                "red_cards": t_red,
                "extra_time": is_et,
                "penalties": is_pen,
                "winner_name_meta": advance_winner,
                "venue": venue,
                "venue_country": venue_country,
            }
        )

    return pd.DataFrame(knockout_results)
