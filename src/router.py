"""Tournament Group Standings and Knockout Stage Routing Engine."""

import logging
import os

import pandas as pd

from src.match_engine import evaluate_match_consensus, simulate_deterministic_match
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
    match_rules,
    elo_engine,
    xgb_home,
    xgb_away,
    feature_columns,
    latest_team_form,
):
    """Simulates the deterministic group stage fixtures using model consensus."""

    group_results = []

    for _, row in group_fixtures.iterrows():
        match_id = int(row["match_id"])
        group_letter = row["group"]
        home = row["home_team"]
        away = row["away_team"]
        venue_country = row["venue_country"]
        venue = row.get("venue", "Neutral")

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
        )

        # Resolve final integer logic
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

    # Data Frame Adjustments
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
    """Simulates the knockout bracket tree sequentially from Round of 32 down to the Final."""

    # Swap out 'assert' for explicit conditional guards to force type-narrowing
    if feature_columns is None:
        raise ValueError(
            "Type Guard: feature_columns list cannot be None inside the routing layer."
        )

    if latest_team_form is None:
        raise ValueError(
            "Type Guard: latest_team_form map cannot be None inside the routing layer."
        )

    raw_dir = os.path.join("data", "raw")
    knockout_template = pd.read_csv(os.path.join(raw_dir, "knockout_slots.csv"))
    knockout_template["venue_country"] = knockout_template["venue"].apply(
        get_venue_country
    )

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

        # Resolve team slots
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

        # 1. Get raw continuous intensities
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

        # Resolve timeline logic
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
