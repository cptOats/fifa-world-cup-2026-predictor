"""World Football Elo Rating and Match Prediction Engine.

This module provides a class-based implementation of the custom World Football Elo
rating system. It tracks continuous team capability values across chronological
historical datasets, evaluates logistic win expectancies, applies goal-differential
multiplier scalars ($G$), and maps finalized ratings back into discrete expected score lines.
"""

import numpy as np


class EloEngine:
    """A thermodynamic rating system that evaluates and tracks football team capabilities.

    This engine implements a rolling rating mechanism where points are traded between
    opponents based on match outcomes relative to pre-match logistic expectancies. It
    supports dynamic goal margin weighting index adjustments to prevent rating dilution
    during high-scoring matches.

    Attributes:
        k_factor (int): Controls the maximum point volatility scale per match update.
        default_elo (int): Initial baseline score assigned to unrated team entities.
        ratings (dict[str, float]): Dynamic ledger mapping country name strings to
            their running calculated Elo rating floats.
    """

    def __init__(self, k_factor=40, default_elo=1500):
        """Initializes the Elo engine with baseline scaling constraints."""
        self.k_factor = k_factor
        self.default_elo = default_elo
        self.ratings = {}

    def get_rating(self, team):
        """Safely fetches a team's current rating, initializing it if absent.

        Args:
            team (str): Standardized country string name of the target team.

        Returns:
            float: The current running Elo rating score assigned to the team.
        """
        if team not in self.ratings:
            self.ratings[team] = self.default_elo
        return self.ratings[team]

    def _calculate_expected_score(self, r_home, r_away):
        """Computes the logistic win expectancy for a matchup on neutral turf.

        Args:
            r_home (float): Pre-match Elo rating float assigned to the home side.
            r_away (float): Pre-match Elo rating float assigned to the away side.

        Returns:
            tuple[float, float]: Win expectation probabilities for home and away sides.
        """
        w_home = 1 / (10 ** (-(r_home - r_away) / 400) + 1)
        w_away = 1.0 - w_home
        return w_home, w_away

    def _get_goal_margin_multiplier(self, home_goals, away_goals):
        """Calculates the standard World Football Elo goal differential index scalar.

        Args:
            home_goals (int): Integer score achieved by the designated home side.
            away_goals (int): Integer score achieved by the designated away side.

        Returns:
            float: The calculated multiplier factor ($G$) used to scale ratings updates.
        """
        goal_diff = abs(home_goals - away_goals)
        if goal_diff <= 1:
            return 1.0
        elif goal_diff == 2:
            return 1.5
        else:
            return (11.0 + goal_diff) / 8.0

    def fit(self, historical_matches_df):
        """Processes a match ledger chronologically to update team ratings step-by-step.

        Args:
            historical_matches_df (pd.DataFrame): Dataframe containing historical results.
        """
        # Ensure chronological processing to keep the thermodynamic updates valid
        sorted_matches = historical_matches_df.sort_values(by="date").copy()

        for _, row in sorted_matches.iterrows():
            home = row["home_team"]
            away = row["away_team"]
            h_goals = int(row["home_score"])
            a_goals = int(row["away_score"])

            # 1. Fetch current ratings before the whistle blows
            r_home = self.get_rating(home)
            r_away = self.get_rating(away)

            # 2. Compute probability expectations
            w_home, w_away = self._calculate_expected_score(r_home, r_away)

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
            self.ratings[home] += self.k_factor * g_factor * (actual_home - w_home)
            self.ratings[away] += self.k_factor * g_factor * (actual_away - w_away)

    def predict_elo_match(self, home_team, away_team, baseline_goals=1.35, alpha=2.2):
        """Translates final Elo delta vectors back into discrete integer score lines.

        Args:
            home_team (str): Standardized country string name of the home team.
            away_team (str): Standardized country string name of the away team.
            baseline_goals (float, optional): Average expected single-side goals parameter.
            alpha (float, optional): Sensitivity multiplier mapping Elo gaps to goal counts.

        Returns:
            dict[str, Any]: Results dictionary tracking predicted goals and winner markers.
        """
        r_home = self.get_rating(home_team)
        r_away = self.get_rating(away_team)

        w_home, _ = self._calculate_expected_score(r_home, r_away)

        # Convert the continuous probability advantage into expected goal lambdas
        lambda_home = max(0.1, baseline_goals + alpha * (w_home - 0.5))
        lambda_away = max(0.1, baseline_goals + alpha * (0.5 - w_home))

        pred_home = int(np.round(lambda_home))
        pred_away = int(np.round(lambda_away))

        if pred_home > pred_away:
            winner = "home"
        elif pred_away > pred_home:
            winner = "away"
        else:
            winner = "draw"

        return {
            "predicted_home_goals": pred_home,
            "predicted_away_goals": pred_away,
            "winning_team": winner,
        }
