"""
Dynamic Elo Rating Model.

A zero-sum thermodynamic rating system that tracks entity capabilities chronologically.
It models historical team form independent of structured feature matrices, ensuring
strict sequence resolution step-by-step through the historical ledger.
"""

import numpy as np
import pandas as pd


class EloEngine:
    """Maintains and updates point-in-time sequence ratings for international squads."""

    def __init__(self, k_factor=40, default_elo=1500):
        """
        Initializes the Elo engine with baseline scaling constraints.

        Args:
            k_factor (int): Volatility index governing rating swings per match.
            default_elo (int): Baseline entry rating for new entities.
        """

        self.k_factor = k_factor
        self.default_elo = default_elo
        self.ratings = {}

    def get_rating(self, team: str) -> float:
        """Safely fetches a team's active rating, initializing it if absent."""

        if team not in self.ratings:
            self.ratings[team] = self.default_elo
        return self.ratings[team]

    def _calculate_expected_score(
        self,
        r_home: float,
        r_away: float,
        is_neutral: int = 0,
        home_advantage: float = 100,
    ) -> tuple[float, float]:
        """
        Computes the logistic win expectancy probability curve.

        Args:
            is_neutral (int): Flag (1) to collapse the host premium mathematically.
        """

        actual_home_adv = 0 if is_neutral == 1 else home_advantage

        w_home = 1 / (10 ** (-(r_home + actual_home_adv - r_away) / 400) + 1)
        w_away = 1.0 - w_home
        return w_home, w_away

    def _get_goal_margin_multiplier(self, home_goals: int, away_goals: int) -> float:
        """Applies the standard World Football Elo goal differential index scalar."""

        goal_diff = abs(home_goals - away_goals)
        if goal_diff <= 1:
            return 1.0
        elif goal_diff == 2:
            return 1.5
        else:
            return (11.0 + goal_diff) / 8.0

    def fit(self, historical_matches_df: pd.DataFrame):
        """Processes the match ledger chronologically to update sequence ratings."""

        # Structural safeguard: Enforce strictly chronological sequence resolution
        sorted_matches = historical_matches_df.sort_values(by="date").copy()

        for _, row in sorted_matches.iterrows():
            home = row["home_team"]
            away = row["away_team"]
            h_goals = int(row["home_score"])
            a_goals = int(row["away_score"])
            is_neutral = int(row.get("neutral", 0))
            match_weight = float(row.get("match_weight", 1.0))

            # Fetch our newly generated match_outcome
            outcome = row.get("match_outcome", "Unknown")

            # 1. Fetch ratings strictly *before* the match is played
            r_home = self.get_rating(home)
            r_away = self.get_rating(away)

            # 2. Compute probabilities and map targets
            w_home, w_away = self._calculate_expected_score(
                r_home, r_away, is_neutral=is_neutral
            )

            # Map actual results using the fractional shootout reality
            if h_goals > a_goals:
                actual_home, actual_away = 1.0, 0.0
            elif a_goals > h_goals:
                actual_home, actual_away = 0.0, 1.0
            else:
                # The game tied in regulation/ET. How was it resolved?
                if outcome == "Home_Win":
                    actual_home, actual_away = 0.75, 0.25
                elif outcome == "Away_Win":
                    actual_home, actual_away = 0.25, 0.75
                else:
                    actual_home, actual_away = 0.5, 0.5  # A true draw

            g_factor = self._get_goal_margin_multiplier(h_goals, a_goals)

            # 3. Execute the delta adjustment update step
            current_k = self.k_factor * match_weight
            self.ratings[home] += current_k * g_factor * (actual_home - w_home)
            self.ratings[away] += current_k * g_factor * (actual_away - w_away)

    def predict_elo_match(
        self,
        home_team: str,
        away_team: str,
        is_neutral: int = 0,
        baseline_goals: float = 1.35,
        alpha: float = 2.2,
    ) -> dict[str, float | int | str]:
        """Translates final Elo delta vectors back into continuous goal intensities (lambdas)."""

        r_home = self.get_rating(home_team)
        r_away = self.get_rating(away_team)

        w_home, _ = self._calculate_expected_score(
            r_home, r_away, is_neutral=is_neutral
        )

        # Map the continuous probability advantage into expected goal intensities
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
