"""Tournament Group Standings and Knockout Stage Routing Engine."""

import logging
import os

import numpy as np
import pandas as pd

from src.match_engine import evaluate_match_consensus
from src.transform import get_venue_country

# The official tournament group dependencies for the 8 third-place wildcard slots
THIRD_PLACE_CONSTRAINTS = {
    75: {"A", "B", "C", "D", "F"},
    78: {"C", "D", "F", "G", "H"},
    79: {"C", "E", "F", "H", "I"},
    80: {"E", "H", "I", "J", "K"},
    81: {"A", "E", "H", "I", "J"},
    82: {"B", "E", "F", "I", "J"},
    85: {"E", "F", "G", "I", "J"},
    88: {"D", "E", "I", "J", "L"},
}


def resolve_group_tables(predicted_fixtures_list_or_df):
    """Compiles tables natively. Accepts either a List of Dicts (Fast) or DataFrame (Legacy)."""

    # Check if we were passed a DataFrame (from deterministic) or List of Dicts (from Stochastic)
    is_df = isinstance(predicted_fixtures_list_or_df, pd.DataFrame)
    records = (
        predicted_fixtures_list_or_df.to_dict("records")
        if is_df
        else predicted_fixtures_list_or_df
    )

    table_records = {}

    for row in records:
        group = row["group"]
        home = row["home_team"]
        away = row["away_team"]
        home_score = row["predicted_home_goals"]
        away_score = row["predicted_away_goals"]

        for team in [home, away]:
            if team not in table_records:
                table_records[team] = {
                    "group": group,
                    "team": team,
                    "points": 0,
                    "goals_for": 0,
                    "goals_against": 0,
                    "goals_diff": 0,
                }

        table_records[home]["goals_for"] += home_score
        table_records[home]["goals_against"] += away_score
        table_records[away]["goals_for"] += away_score
        table_records[away]["goals_against"] += home_score

        if home_score > away_score:
            table_records[home]["points"] += 3
        elif away_score > home_score:
            table_records[away]["points"] += 3
        else:
            table_records[home]["points"] += 1
            table_records[away]["points"] += 1

    compiled_list = list(table_records.values())
    for stats in compiled_list:
        stats["goals_diff"] = stats["goals_for"] - stats["goals_against"]

    # Native Python Multi-key Sort (Equivalent to Pandas Ascending/Descending)
    # Group (Asc), Points (Desc), Goal Diff (Desc), Goals For (Desc)
    compiled_list.sort(
        key=lambda x: (x["group"], -x["points"], -x["goals_diff"], -x["goals_for"])
    )

    # Add position integers
    current_group = None
    pos = 1
    for row in compiled_list:
        if row["group"] != current_group:
            current_group = row["group"]
            pos = 1
        row["position"] = pos
        pos += 1

    return pd.DataFrame(compiled_list)


def extract_best_third_places(group_tables_df):
    """Isolates all 12 third-place finishers and extracts the top 8 wildcards."""

    third_places = group_tables_df[group_tables_df["position"] == 3].copy()
    ranked_thirds = third_places.sort_values(
        by=["points", "goals_diff", "goals_for"], ascending=[False, False, False]
    ).reset_index(drop=True)

    return ranked_thirds.head(8).copy()


def allocate_third_places(advancing_thirds_df):
    """Maps qualifying third-place teams to unique knockout match slots."""

    teams = list(zip(advancing_thirds_df["group"], advancing_thirds_df["team"]))
    slot_ids = list(THIRD_PLACE_CONSTRAINTS.keys())

    def backtrack(
        team_idx: int, current_assignment: dict[int, str]
    ) -> dict[int, str] | None:
        """Finds a valid slot assignment using a backtracking search algorithm."""
        # Base case: All teams have been successfully assigned
        if team_idx == len(teams):
            return current_assignment

        group, team_name = teams[team_idx]

        # Try assigning the current team to an available slot
        for slot in slot_ids:
            if slot not in current_assignment:
                # Check if the team's group is allowed in this slot
                if group in THIRD_PLACE_CONSTRAINTS[slot]:
                    next_assignment = current_assignment.copy()
                    next_assignment[slot] = team_name

                    # Recurse for the next team
                    result = backtrack(team_idx + 1, next_assignment)
                    if result is not None:
                        return result

        return None

    # Kick off the recursive backtracking search
    assignment = backtrack(0, {})

    # GREEDY FALLBACK: If a chaotic Monte Carlo universe breaks the official matrix
    if assignment is None:
        logging.debug(
            "Wildcard constraint broken by stochastic upset. Applying greedy fallback."
        )
        assignment = {}
        available_slots = list(slot_ids)

        # Force assign teams to whatever slots are left
        for _, team_name in teams:
            if available_slots:
                assignment[available_slots.pop(0)] = team_name

    return assignment


def generate_round_of_32_draw(group_tables_df, third_place_mapping):
    """Reads the template layout and substitutes placeholders with actual country names."""

    raw_dir = os.path.join("data", "raw")
    knockout_template = pd.read_csv(os.path.join(raw_dir, "knockout_slots.csv"))

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
        match_id = row["match_id"]
        slot_home = row["slot_home"]
        slot_away = row["slot_away"]

        if "Winner Group" in slot_home:
            grp = slot_home.replace("Winner Group ", "").strip()
            home_teams.append(winners[grp])
        elif "Runner-up Group" in slot_home:
            grp = slot_home.replace("Runner-up Group ", "").strip()
            home_teams.append(runners_up[grp])
        else:
            home_teams.append(third_place_mapping[match_id])

        if "Winner Group" in slot_away:
            grp = slot_away.replace("Winner Group ", "").strip()
            away_teams.append(winners[grp])
        elif "Runner-up Group" in slot_away:
            grp = slot_away.replace("Runner-up Group ", "").strip()
            away_teams.append(runners_up[grp])
        else:
            away_teams.append(third_place_mapping[match_id])

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
    elo_engine,
    xgb_home,
    xgb_away,
    feature_columns,
    latest_team_form,
    use_prior_nudge,
    nudge_strength,
    team_power,
):
    """
    Simulates the deterministic group stage fixtures using model consensus.
    Resolves league tables, extracts wildcards, and assigns tournament brackets.
    """

    group_results = []

    for _, row in group_fixtures.iterrows():
        match_id = int(row["match_id"])
        group_letter = row["group"]
        home = row["home_team"]
        away = row["away_team"]
        venue_country = row["venue_country"]

        # Call Match Engine
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
            use_prior_nudge=use_prior_nudge,
            nudge_strength=nudge_strength,
            team_power=team_power,
        )

        final_home_goals = int(np.round(max(0, raw_home)))
        final_away_goals = int(np.round(max(0, raw_away)))
        winner_side = (
            "home"
            if final_home_goals > final_away_goals
            else ("away" if final_away_goals > final_home_goals else "draw")
        )

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

    predicted_fixtures = pd.DataFrame(group_results)

    # Route Bracket Structures
    group_tables = resolve_group_tables(predicted_fixtures)
    top_thirds = extract_best_third_places(group_tables)
    third_place_assignments = allocate_third_places(top_thirds)

    # Mutate the latest team form tracker to embed ensemble weights metadata
    latest_team_form["__meta_weights__"] = blend_weights

    return predicted_fixtures, group_tables, third_place_assignments


def simulate_knockout_waterfall(
    group_tables_df,
    third_place_mapping,
    ratings,
    g_home_avg,
    g_away_avg,
    g_neutral_avg,
    blend_weights,
    match_rules,
    elo_engine=None,
    xgb_home=None,
    xgb_away=None,
    feature_columns=None,
    latest_team_form=None,
    use_prior_nudge=False,
    nudge_strength=1.5,
    team_power=None,
):
    """Simulates the knockout bracket tree sequentially from Round of 32 down to the Final."""

    assert feature_columns is not None, (
        "❌ Type Enforcement: feature_columns list cannot be None inside the routing layer."
    )
    assert latest_team_form is not None, (
        "❌ Type Enforcement: latest_team_form map cannot be None inside the routing layer."
    )

    et_multiplier = match_rules["et_multiplier"]
    fatigue_factor = match_rules["fatigue_factor"]

    raw_dir = os.path.join("data", "raw")
    knockout_template = pd.read_csv(os.path.join(raw_dir, "knockout_slots.csv"))

    knockout_template["venue_country"] = knockout_template["venue"].apply(
        get_venue_country
    )

    match_winners = {}
    match_losers = {}

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

        if "Winner Group" in slot_home:
            home_team = winners[slot_home.replace("Winner Group ", "").strip()]
        elif "Runner-up Group" in slot_home:
            home_team = runners_up[slot_home.replace("Runner-up Group ", "").strip()]
        elif "Best 3rd" in slot_home:
            home_team = third_place_mapping[match_id]
        elif "Winner Match" in slot_home:
            home_team = match_winners[
                int(slot_home.replace("Winner Match ", "").strip())
            ]
        elif "Loser Match" in slot_home:
            home_team = match_losers[int(slot_home.replace("Loser Match ", "").strip())]
        else:
            home_team = slot_home

        if "Winner Group" in slot_away:
            away_team = winners[slot_away.replace("Winner Group ", "").strip()]
        elif "Runner-up Group" in slot_away:
            away_team = runners_up[slot_away.replace("Runner-up Group ", "").strip()]
        elif "Best 3rd" in slot_away:
            away_team = third_place_mapping[match_id]
        elif "Winner Match" in slot_away:
            away_team = match_winners[
                int(slot_away.replace("Winner Match ", "").strip())
            ]
        elif "Loser Match" in slot_away:
            away_team = match_losers[int(slot_away.replace("Loser Match ", "").strip())]
        else:
            away_team = slot_away

        # Call the unified match engine to calculate baseline 90-min capability curves
        raw_home, raw_away, tot_corners_90, tot_yellows_90, tot_reds_90 = (
            evaluate_match_consensus(
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
                use_prior_nudge=use_prior_nudge,
                nudge_strength=nudge_strength,
                team_power=team_power,
            )
        )

        pred_home_90 = int(np.round(raw_home))
        pred_away_90 = int(np.round(raw_away))

        # --- TIMELINE RESOLUTION GATE (Normal vs Extra Time) ---
        is_extra_time = False
        is_penalty = False

        if pred_home_90 > pred_away_90:
            final_home_goals, final_away_goals = pred_home_90, pred_away_90
            advance_winner, advance_loser = home_team, away_team
            winner_side = "home"
            tot_corners = int(np.clip(np.round(tot_corners_90), 4, 16))
            tot_yellows = int(np.clip(np.round(tot_yellows_90), 1, 9))
            tot_reds = int(np.clip(np.round(tot_reds_90), 0, 3))

        elif pred_away_90 > pred_home_90:
            final_home_goals, final_away_goals = pred_home_90, pred_away_90
            advance_winner, advance_loser = away_team, home_team
            winner_side = "away"
            tot_corners = int(np.clip(np.round(tot_corners_90), 4, 16))
            tot_yellows = int(np.clip(np.round(tot_yellows_90), 1, 9))
            tot_reds = int(np.clip(np.round(tot_reds_90), 0, 3))

        else:
            # Regulation Integer Draw -> Triggers Additive Extra Time
            is_extra_time = True

            raw_home_120 = raw_home * (1 + (et_multiplier * fatigue_factor))
            raw_away_120 = raw_away * (1 + (et_multiplier * fatigue_factor))

            final_home_goals = int(np.round(raw_home_120))
            final_away_goals = int(np.round(raw_away_120))

            tot_corners = int(
                np.clip(
                    np.round(tot_corners_90 * (1 + (et_multiplier * fatigue_factor))),
                    5,
                    18,
                )
            )
            tot_yellows = int(
                np.clip(
                    np.round(tot_yellows_90 * (1 + (et_multiplier * fatigue_factor))),
                    1,
                    12,
                )
            )
            tot_reds = int(
                np.clip(
                    np.round(tot_reds_90 * (1 + (et_multiplier * fatigue_factor))),
                    0,
                    4,
                )
            )

            if final_home_goals > final_away_goals:
                advance_winner, advance_loser = home_team, away_team
                winner_side = "home"
            elif final_away_goals > final_home_goals:
                advance_winner, advance_loser = away_team, home_team
                winner_side = "away"
            else:
                # 120 mins Integer Draw -> Penalty Shootout
                is_penalty = True
                if raw_home >= raw_away:
                    advance_winner, advance_loser = home_team, away_team
                    winner_side = "home"
                else:
                    advance_winner, advance_loser = away_team, home_team
                    winner_side = "away"

        match_winners[match_id] = advance_winner
        match_losers[match_id] = advance_loser

        knockout_results.append(
            {
                "match_id": match_id,
                "round": r_name,
                "venue": venue,
                "predicted_home_team": home_team,
                "predicted_away_team": away_team,
                "predicted_home_goals": final_home_goals,
                "predicted_away_goals": final_away_goals,
                "corners": tot_corners,
                "yellow_cards": tot_yellows,
                "red_cards": tot_reds,
                "match_winner": winner_side,
                "extra_time": is_extra_time,
                "penalties": is_penalty,
                "winner_name_meta": advance_winner,
            }
        )

    return pd.DataFrame(knockout_results)
