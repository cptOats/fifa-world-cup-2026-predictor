"""Dynamic Elo Rating Model."""

import numpy as np
import pandas as pd


class EloEngine:
    """A thermodynamic rating system that evaluates and tracks football team capabilities."""

    def __init__(self, k_factor=40, default_elo=1500):
        """Initializes the Elo engine with baseline scaling constraints."""
        self.k_factor = k_factor
        self.default_elo = default_elo
        self.ratings = {}

    def get_rating(self, team):
        """Safely fetches a team's current rating, initializing it if absent."""

        if team not in self.ratings:
            self.ratings[team] = self.default_elo
        return self.ratings[team]

    def _calculate_expected_score(
        self, r_home, r_away, is_neutral=0, home_advantage=100
    ):
        """Computes the logistic win expectancy for a matchup incorporating venue states."""

        # Symmetrical neutrality override: Host premium collapses to 0 on neutral grounds
        actual_home_adv = 0 if is_neutral == 1 else home_advantage

        w_home = 1 / (10 ** (-(r_home + actual_home_adv - r_away) / 400) + 1)
        w_away = 1.0 - w_home
        return w_home, w_away

    def _get_goal_margin_multiplier(self, home_goals, away_goals):
        """Calculates the standard World Football Elo goal differential index scalar."""

        goal_diff = abs(home_goals - away_goals)
        if goal_diff <= 1:
            return 1.0
        elif goal_diff == 2:
            return 1.5
        else:
            return (11.0 + goal_diff) / 8.0

    def fit(self, historical_matches_df: pd.DataFrame):
        """Processes a match ledger chronologically to update team ratings step-by-step."""

        # Structural safeguard: Enforce strict sequence resolution across rolling timelines
        sorted_matches = historical_matches_df.sort_values(by="date").copy()

        for _, row in sorted_matches.iterrows():
            home = row["home_team"]
            away = row["away_team"]
            h_goals = int(row["home_score"])
            a_goals = int(row["away_score"])
            is_neutral = int(row.get("neutral", 0))
            match_weight = float(row.get("match_weight", 1.0))

            # 1. Fetch current ratings before the whistle blows
            r_home = self.get_rating(home)
            r_away = self.get_rating(away)

            # 2. Compute probability expectations passing the neutral flag
            w_home, w_away = self._calculate_expected_score(
                r_home, r_away, is_neutral=is_neutral
            )

            # 3. Map match outcomes (Win = 1.0, Draw = 0.5, Loss = 0.0)
            if h_goals > a_goals:
                actual_home, actual_away = 1.0, 0.0
            elif a_goals > h_goals:
                actual_home, actual_away = 0.0, 1.0
            else:
                actual_home, actual_away = 0.5, 0.5

            # 4. Compute the goal margin index scale factor
            g_factor = self._get_goal_margin_multiplier(h_goals, a_goals)

            # 5. Execute the delta adjustment update step
            current_k = self.k_factor * match_weight
            self.ratings[home] += current_k * g_factor * (actual_home - w_home)
            self.ratings[away] += current_k * g_factor * (actual_away - w_away)

    def predict_elo_match(
        self, home_team, away_team, is_neutral=0, baseline_goals=1.35, alpha=2.2
    ):
        """Translates final Elo delta vectors back into discrete integer score lines."""
        r_home = self.get_rating(home_team)
        r_away = self.get_rating(away_team)

        # Added parameter signature injection to prevent unbound variable crashes
        w_home, _ = self._calculate_expected_score(
            r_home, r_away, is_neutral=is_neutral
        )

        # Convert the continuous probability advantage into expected goal lambdas
        lambda_home = max(0.1, baseline_goals + alpha * (w_home - 0.5))
        lambda_away = max(0.1, baseline_goals + alpha * (0.5 - w_home))

        pred_home = int(np.round(lambda_home))
        pred_away = int(np.round(lambda_away))

        return {
            "predicted_home_goals": pred_home,
            "predicted_away_goals": pred_away,
            "lambda_home": lambda_home,
            "lambda_away": lambda_away,
            "winner": "home"
            if pred_home > pred_away
            else ("away" if pred_away > pred_home else "draw"),
        }
