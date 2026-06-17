"""Tournament Standings Resolution and Knockout Simulation Routing Layer.

This module provides the core logic for translating raw simulated match results into
structured tournament brackets. It handles group table point compilations, resolves
the complex bipartite matching constraints for wildcards (best third-place teams),
and simulates the sequential knockout waterfall phase including extra time and penalty
shootouts.
"""

import os

import numpy as np
import pandas as pd

from src.poisson import predict_poisson_match
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


def resolve_group_tables(predicted_fixtures_df):
    """Compiles the 12 group stage standings tables from simulated match results.

    Aggregates points, goals scored (goals_for), goals conceded (goals_against),
    and goal differentials for every team across all group fixtures. The resulting
    tables are sorted dynamically using official tournament tiebreaker rules.

    Args:
        predicted_fixtures_df (pd.DataFrame): Pandas DataFrame containing simulated
            group matches with columns `group`, `home_team`, `away_team`,
            `predicted_home_goals`, and `predicted_away_goals`.

    Returns:
        pd.DataFrame: Sorted group standings containing columns for points, goal matrix,
            and an explicitly assigned structural `position` rank (1 to 4).
    """
    table_records = {}

    for _, row in predicted_fixtures_df.iterrows():
        group = row["group"]
        home = row["home_team"]
        away = row["away_team"]

        home_score = int(row["predicted_home_goals"])
        away_score = int(row["predicted_away_goals"])

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

    compiled_list = []
    for team, stats in table_records.items():
        stats["goals_diff"] = stats["goals_for"] - stats["goals_against"]
        compiled_list.append(stats)

    tables = pd.DataFrame(compiled_list)

    tables = tables.sort_values(
        by=["group", "points", "goals_diff", "goals_for"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)

    tables["position"] = tables.groupby("group").cumcount() + 1
    return tables


def extract_best_third_places(group_tables_df):
    """Isolates all 12 third-place finishers and extracts the top 8 wildcards.

    Filters the master standings table to isolate teams finishing in 3rd place
    within their respective groups, sorts them based on tournament point metrics,
    and returns the top 8 qualifying wildcard slots.

    Args:
        group_tables_df (pd.DataFrame): Compiled group standings including points,
            goal differentials, and position records.

    Returns:
        pd.DataFrame: A filtered and sorted DataFrame containing only the 8 best
            third-place teams advancing to the knockout round.
    """
    third_places = group_tables_df[group_tables_df["position"] == 3].copy()
    ranked_thirds = third_places.sort_values(
        by=["points", "goals_diff", "goals_for"], ascending=[False, False, False]
    ).reset_index(drop=True)

    return ranked_thirds.head(8).copy()


def allocate_third_places(advancing_thirds_df):
    """Maps qualifying third-place teams to unique knockout match slots.

    Ingests the 8 advancing third-place groups sorted by tournament criteria and
    maps them directly to their designated bracket pathways using an optimized,
    static tournament configuration array. This completely avoids runtime
    combinatorial backtracking steps.

    Args:
        advancing_thirds_df (pd.DataFrame): Dataframe containing the 8 advancing
            third-place teams alongside their native group identifiers.

    Returns:
        dict[int, str]: A dictionary map linking specific knockout `match_id` integers
            directly to the assigned team name strings.
    """
    teams = list(zip(advancing_thirds_df["group"], advancing_thirds_df["team"]))
    slot_ids = list(THIRD_PLACE_CONSTRAINTS.keys())

    def backtrack(
        team_idx: int, current_assignment: dict[int, str]
    ) -> dict[int, str] | None:
        """Finds a valid slot assignment using a backtracking search algorithm.

        This helper function recursively iterates through available slots to
        assign the current team, ensuring that the team's group satisfies the
        predefined third-place constraints for that slot.

        Args:
            team_idx (int): The index of the team currently being assigned from
                the outer `teams` list.
            current_assignment (dict): A dictionary mapping slot IDs to team names,
                representing the successful assignments made up to this point.

        Returns:
            dict | None: A complete dictionary of slot-to-team assignments if a
            valid configuration is found; None if the current path leads to a dead
            end and backtracking is required.
        """
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

    # Kick off the recursive backtracking search starting at index 0
    assignment = backtrack(0, {})
    if assignment is None:
        raise ValueError(
            "Fatal: Could not find a valid slot assignment for this combination of 3rd place teams."
        )

    return assignment


def generate_round_of_32_draw(group_tables_df, third_place_mapping):
    """Reads the template layout and substitutes placeholders with actual country names.

    Maps structural template placeholders (e.g., 'Winner Group A', 'Runner-up Group B')
    and resolved third-place wildcard paths to build the initial static pairs for the
    Round of 32 knockout bracket.

    Args:
        group_tables_df (pd.DataFrame): Master parsed group stage standings data.
        third_place_mapping (dict[int, str]): Map linking wildcard `match_id` positions
            to allocated team name strings.

    Returns:
        pd.DataFrame: Staged Round of 32 fixture matrix displaying explicit `home_team`
            and `away_team` country pairs.
    """
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


def simulate_knockout_waterfall(
    group_tables_df,
    third_place_mapping,
    ratings,
    g_home_avg,
    g_away_avg,
    g_neutral_avg,
    blend_weights,
    elo_engine=None,
    xgb_home=None,
    xgb_away=None,
    feature_columns=None,
    latest_team_form=None,
):
    """Simulates the knockout bracket tree sequentially from Round of 32 down to the Final."""

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

    FATIGUE_FACTOR = 0.80
    CARD_BOOST_FACTOR = 1.75
    ET_MULTIPLIER = 1 / 3

    for _, row in knockout_template.iterrows():
        match_id = int(row["match_id"])
        r_name = row["round"]
        venue = row["venue"]
        slot_home = row["slot_home"]
        slot_away = row["slot_away"]

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

        # VENUE NEUTRALITY RESOLUTION
        venue_country = row["venue_country"]
        is_neutral = (
            0 if (home_team == venue_country or away_team == venue_country) else 1
        )

        (
            lambda_home_poisson,
            lambda_away_poisson,
            raw_corners_90,
            raw_yellows_90,
            raw_reds_90,
        ) = predict_poisson_match(
            home_team,
            away_team,
            venue_country,
            ratings,
            g_home_avg,
            g_away_avg,
            g_neutral_avg,
        )

        # MIRRORED XGBOOST FEATURE CONSTRAINTS
        form_registry = latest_team_form if latest_team_form is not None else {}
        fallback_form = {
            "ewm_gf_4s": 1.2,
            "ewm_ga_4s": 1.2,
            "ewm_wr_4s": 0.35,
            "ewm_gf_10s": 1.2,
            "ewm_ga_10s": 1.2,
            "ewm_wr_10s": 0.35,
        }
        h_form = form_registry.get(home_team, fallback_form) or fallback_form
        a_form = form_registry.get(away_team, fallback_form) or fallback_form

        def _get_ko_vec(h_t, a_t, h_f, a_f, is_neut):
            """Helper function to fetch knockout vectors."""
            return {
                "home_elo_rating": elo_engine.get_rating(h_t) if elo_engine else 1500.0,
                "away_elo_rating": elo_engine.get_rating(a_t) if elo_engine else 1500.0,
                "elo_differential": (
                    elo_engine.get_rating(h_t) - elo_engine.get_rating(a_t)
                )
                if elo_engine
                else 0.0,
                "is_neutral_venue": is_neut,
                "home_team_ewm_gf_4s": h_f["ewm_gf_4s"],
                "home_team_ewm_ga_4s": h_f["ewm_ga_4s"],
                "home_team_ewm_wr_4s": h_f["ewm_wr_4s"],
                "home_team_ewm_gf_10s": h_f["ewm_gf_10s"],
                "home_team_ewm_ga_10s": h_f["ewm_ga_10s"],
                "home_team_ewm_wr_10s": h_f["ewm_wr_10s"],
                "away_team_ewm_gf_4s": a_f["ewm_gf_4s"],
                "away_team_ewm_ga_4s": a_f["ewm_ga_4s"],
                "away_team_ewm_wr_4s": a_f["ewm_wr_4s"],
                "away_team_ewm_gf_10s": a_f["ewm_gf_10s"],
                "away_team_ewm_ga_10s": a_f["ewm_ga_10s"],
                "away_team_ewm_wr_10s": a_f["ewm_wr_10s"],
            }

        df_fwd = pd.DataFrame(
            [_get_ko_vec(home_team, away_team, h_form, a_form, is_neutral)]
        )[feature_columns]
        df_swp = pd.DataFrame(
            [_get_ko_vec(away_team, home_team, a_form, h_form, is_neutral)]
        )[feature_columns]

        h_fwd = xgb_home.predict(df_fwd)[0] if xgb_home else 0.0
        a_fwd = xgb_away.predict(df_fwd)[0] if xgb_away else 0.0
        h_swp = xgb_home.predict(df_swp)[0] if xgb_home else 0.0
        a_swp = xgb_away.predict(df_swp)[0] if xgb_away else 0.0

        if home_team == venue_country:
            xgb_h_pred, xgb_w_pred = h_fwd, a_fwd
        elif away_team == venue_country:
            xgb_h_pred, xgb_w_pred = a_swp, h_swp
        else:
            xgb_h_pred = (h_fwd + a_swp) / 2.0
            xgb_w_pred = (a_fwd + h_swp) / 2.0

        # --- THE UNIFIED CONSENSUS COGNITIVE RUNNER ---
        assert elo_engine is not None
        b_w = blend_weights
        elo_meta = elo_engine.predict_elo_match(
            home_team, away_team, is_neutral=is_neutral
        )

        # Unified weight-driven continuous intensity extraction
        raw_home = (
            (b_w["poisson"] * lambda_home_poisson)
            + (b_w["elo"] * float(elo_meta["lambda_home"]))
            + (b_w["xgb"] * xgb_h_pred)
        )
        raw_away = (
            (b_w["poisson"] * lambda_away_poisson)
            + (b_w["elo"] * float(elo_meta["lambda_away"]))
            + (b_w["xgb"] * xgb_w_pred)
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
            tot_corners = int(np.clip(np.round(raw_corners_90), 4, 16))
            tot_yellows = int(np.clip(np.round(raw_yellows_90), 1, 9))
            tot_reds = int(np.clip(np.round(raw_yellows_90), 0, 3))
        elif pred_away_90 > pred_home_90:
            final_home_goals, final_away_goals = pred_home_90, pred_away_90
            advance_winner, advance_loser = away_team, home_team
            winner_side = "away"
            tot_corners = int(np.clip(np.round(raw_corners_90), 4, 16))
            tot_yellows = int(np.clip(np.round(raw_yellows_90), 1, 9))
            tot_reds = int(np.clip(np.round(raw_yellows_90), 0, 3))
        else:
            # Regulation Integer Draw -> Triggers Additive Extra Time
            is_extra_time = True

            # Natively degrade the continuous expectations using your fatigue scalars
            et_home = int(np.round(raw_home * (ET_MULTIPLIER * FATIGUE_FACTOR)))
            et_away = int(np.round(raw_away * (ET_MULTIPLIER * FATIGUE_FACTOR)))

            final_home_goals = pred_home_90 + et_home
            final_away_goals = pred_away_90 + et_away

            tot_corners = int(
                np.clip(
                    np.round(raw_corners_90 * (1 + (ET_MULTIPLIER * FATIGUE_FACTOR))),
                    5,
                    18,
                )
            )
            tot_yellows = int(
                np.clip(
                    np.round(
                        raw_yellows_90 * (1 + (ET_MULTIPLIER * CARD_BOOST_FACTOR))
                    ),
                    1,
                    12,
                )
            )
            tot_reds = int(
                np.clip(
                    np.round(raw_reds_90 * (1 + (ET_MULTIPLIER * CARD_BOOST_FACTOR))),
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
                # 120 mins Integer Draw -> Resolves via Penalty Shootout using raw floats to determine winner
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
