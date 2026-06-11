import os

import numpy as np
import pandas as pd

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
    """Compiles the 12 group stage standings tables from simulated results."""
    # Track point tallies and goal arrays via team state structures
    table_records = {}

    for _, row in predicted_fixtures_df.iterrows():
        group = row["group"]
        home = row["home_team"]
        away = row["away_team"]

        # Parse new DataCamp-compliant goal tracking columns
        home_score = int(row["predicted_home_goals"])
        away_score = int(row["predicted_away_goals"])

        # Initialize team state profile schemas if missing from index records
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

        # Accumulate absolute volume goals
        table_records[home]["goals_for"] += home_score
        table_records[home]["goals_against"] += away_score
        table_records[away]["goals_for"] += away_score
        table_records[away]["goals_against"] += home_score

        # Distribute competition points based on match outcome orientations
        if home_score > away_score:
            table_records[home]["points"] += 3
        elif away_score > home_score:
            table_records[away]["points"] += 3
        else:
            table_records[home]["points"] += 1
            table_records[away]["points"] += 1

    # Unpack the compiled structures into a list for DataFrame parsing
    compiled_list = []
    for team, stats in table_records.items():
        # Calculate final net goal differential vectors
        stats["goals_diff"] = stats["goals_for"] - stats["goals_against"]
        compiled_list.append(stats)

    tables = pd.DataFrame(compiled_list)

    # Tie-breakers: Points -> Goal Diff -> Goals For
    tables = tables.sort_values(
        by=["group", "points", "goals_diff", "goals_for"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)

    # Assign intra-group seeding ranks (1 through 4)
    tables["position"] = tables.groupby("group").cumcount() + 1
    return tables


def extract_best_third_places(group_tables_df):
    """Isolates all 12 third-place finishers and extracts the top 8."""
    third_places = group_tables_df[group_tables_df["position"] == 3].copy()
    ranked_thirds = third_places.sort_values(
        by=["points", "goals_diff", "goals_for"], ascending=[False, False, False]
    ).reset_index(drop=True)

    return ranked_thirds.head(8).copy()


def allocate_third_places(advancing_thirds_df):
    """Solves the bipartite matching problem to map the 8 advancing third-place teams to their unique match slots based on constraint sets."""
    # Convert advancing teams into a list of tuples: (group, team_name)
    teams = list(zip(advancing_thirds_df["group"], advancing_thirds_df["team"]))
    slot_ids = list(THIRD_PLACE_CONSTRAINTS.keys())

    def backtrack(team_idx, current_assignment):
        # Base Case: All 8 teams successfully matched to slots
        if team_idx == len(teams):
            return current_assignment

        group, team_name = teams[team_idx]

        # Evaluate available slots for this team's group letter
        for slot in slot_ids:
            if slot not in current_assignment:
                if group in THIRD_PLACE_CONSTRAINTS[slot]:
                    # Try assigning this team to the slot
                    next_assignment = current_assignment.copy()
                    next_assignment[slot] = team_name

                    # Recursively try to match the remaining teams
                    result = backtrack(team_idx + 1, next_assignment)
                    if result is not None:
                        return result
        return None

    assignment = backtrack(0, {})
    if assignment is None:
        raise ValueError(
            "Fatal: Could not find a valid slot assignment for this combination of 3rd place teams."
        )

    return assignment


def generate_round_of_32_draw(group_tables_df, third_place_mapping):
    """Reads the template layout and substitutes placeholders with actual country names."""
    raw_dir = os.path.join("data", "raw")
    knockout_template = pd.read_csv(os.path.join(raw_dir, "knockout_slots.csv"))

    # Isolate only the Round of 32 fixtures (Matches 73 to 88)
    r32_df = knockout_template[knockout_template["round"] == "Round of 32"].copy()

    # Create maps for winners and runners-up
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

        # Resolve Home Slot
        if "Winner Group" in slot_home:
            grp = slot_home.replace("Winner Group ", "").strip()
            home_teams.append(winners[grp])
        elif "Runner-up Group" in slot_home:
            grp = slot_home.replace("Runner-up Group ", "").strip()
            home_teams.append(runners_up[grp])
        else:
            home_teams.append(third_place_mapping[match_id])

        # Resolve Away Slot
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
    model_type="poisson",
    elo_engine=None,
):
    """Simulates the knockout bracket sequentially, evaluating regular time (90m), extra time (120m) with fatigue adjustments, and penalty shootouts."""
    from src.models import get_dixon_coles_score, get_venue_country

    raw_dir = os.path.join("data", "raw")
    knockout_template = pd.read_csv(os.path.join(raw_dir, "knockout_slots.csv"))

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

    # Fatigue Factor: Extra time goal scoring efficiency and corner occurrences drop
    FATIGUE_FACTOR = 0.80
    # Card Boost Factor: Extra time (yellow) cards spike due to tired fouls and compounding dissent
    CARD_BOOST_FACTOR = 1.75
    ET_MULTIPLIER = 1 / 3

    print(f"\n🔮 Simulating Knockout Waterfall (Model Engine: {model_type.upper()})...")

    for _, row in knockout_template.iterrows():
        match_id = int(row["match_id"])
        r_name = row["round"]
        venue = row["venue"]
        slot_home = row["slot_home"]
        slot_away = row["slot_away"]

        # Resolve Team Identities
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

        # 90min Expected Goal Intensities & Parameters
        home_rating = ratings.get(home_team, {"attack": 1.0, "defense": 1.0})
        away_rating = ratings.get(away_team, {"attack": 1.0, "defense": 1.0})

        venue_country = get_venue_country(venue)
        g_neutral = (g_home_avg + g_away_avg) / 2.0

        if home_team == venue_country:
            lambda_home_90 = home_rating["attack"] * away_rating["defense"] * g_home_avg
            lambda_away_90 = away_rating["attack"] * home_rating["defense"] * g_away_avg
        elif away_team == venue_country:
            lambda_home_90 = home_rating["attack"] * away_rating["defense"] * g_away_avg
            lambda_away_90 = away_rating["attack"] * home_rating["defense"] * g_home_avg
        else:
            lambda_home_90 = home_rating["attack"] * away_rating["defense"] * g_neutral
            lambda_away_90 = away_rating["attack"] * home_rating["defense"] * g_neutral

        # 1. EXECUTE MODEL INTERCEPT TOGGLE FOR 90-MINUTE GOALS
        if model_type == "poisson":
            pred_home_90, pred_away_90 = get_dixon_coles_score(
                lambda_home_90, lambda_away_90
            )
        elif model_type == "elo":
            # Extract win expectancy from Elo component logic to map back continuous intensities
            w_home, _ = elo_engine.calculate_expected_score(
                elo_engine.get_rating(home_team), elo_engine.get_rating(away_team)
            )
            # Re-generate aligned continuous distributions for ET mapping scaling
            lambda_home_90 = max(0.1, 1.35 + 2.2 * (w_home - 0.5))
            lambda_away_90 = max(0.1, 1.35 + 2.2 * ((1.0 - w_home) - 0.5))

            # Extract score integers natively from the instantiated EloEngine module
            elo_res = elo_engine.predict_match(home_team, away_team)
            pred_home_90 = elo_res["predicted_home_goals"]
            pred_away_90 = elo_res["predicted_away_goals"]

        # 2. Compute Raw 90min Continuous Float Baselines for Secondary Metrics
        raw_corners_90 = (5.5 * home_rating["attack"] * away_rating["defense"]) + (
            5.5 * away_rating["attack"] * home_rating["defense"]
        )

        raw_yellows_90 = (3.0 * home_rating["defense"] * away_rating["attack"]) + (
            3.0 * away_rating["defense"] * home_rating["attack"]
        )

        # 3. TIMELINE RESOLUTION GATE
        is_penalty = False
        tot_reds = 0

        if pred_home_90 > pred_away_90:
            final_home_goals, final_away_goals = pred_home_90, pred_away_90
            advance_winner, advance_loser = home_team, away_team
            winner_side = "home"
            tot_corners = int(np.clip(np.round(raw_corners_90), 5, 16))
            tot_yellows = int(np.clip(np.round(raw_yellows_90), 1, 9))

        elif pred_away_90 > pred_home_90:
            final_home_goals, final_away_goals = pred_home_90, pred_away_90
            advance_winner, advance_loser = away_team, home_team
            winner_side = "away"
            tot_corners = int(np.clip(np.round(raw_corners_90), 5, 16))
            tot_yellows = int(np.clip(np.round(raw_yellows_90), 1, 9))

        else:
            # 90min Integer Draw
            lambda_home_120 = lambda_home_90 * (1 + (ET_MULTIPLIER * FATIGUE_FACTOR))
            lambda_away_120 = lambda_away_90 * (1 + (ET_MULTIPLIER * FATIGUE_FACTOR))

            # 120min Goal Resolution
            if model_type == "poisson":
                pred_home_120, pred_away_120 = get_dixon_coles_score(
                    lambda_home_120, lambda_away_120
                )
            elif model_type == "elo":
                pred_home_120 = int(np.round(lambda_home_120))
                pred_away_120 = int(np.round(lambda_away_120))

            # 120min Inflate Stats
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

            if pred_home_120 > pred_away_120:
                final_home_goals, final_away_goals = pred_home_120, pred_away_120
                advance_winner, advance_loser = home_team, away_team
                winner_side = "home"
            elif pred_away_120 > pred_home_120:
                final_home_goals, final_away_goals = pred_home_120, pred_away_120
                advance_winner, advance_loser = away_team, home_team
                winner_side = "away"
            else:
                # 120min Draw -> Penalty Shootout
                final_home_goals, final_away_goals = pred_home_120, pred_away_120
                is_penalty = True
                if lambda_home_120 >= lambda_away_120:
                    advance_winner, advance_loser = home_team, away_team
                    winner_side = "home"
                else:
                    advance_winner, advance_loser = away_team, home_team
                    winner_side = "away"

        # Update lookup tables for subsequent tournament rounds
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
                "penalties": is_penalty,
                "winner_name_meta": advance_winner,
            }
        )

    return pd.DataFrame(knockout_results)
