import numpy as np
import pandas as pd


class EloEngine:
    def __init__(self, k_factor=40, default_elo=1500):
        """
        Initializes the Elo engine.
        K-factor controls the maximum volatility per match update (40 is standard for internationals).
        """
        self.k_factor = k_factor
        self.default_elo = default_elo
        self.ratings = {}

    def get_rating(self, team):
        """Safely fetches a team's current rating, initializing at 1500 if missing."""
        if team not in self.ratings:
            self.ratings[team] = self.default_elo
        return self.ratings[team]

    def calculate_expected_score(self, r_home, r_away):
        """Computes the logistic win expectancy for the home side on neutral turf."""
        w_home = 1 / (10 ** (-(r_home - r_away) / 400) + 1)
        w_away = 1.0 - w_home
        return w_home, w_away

    def get_goal_margin_multiplier(self, home_goals, away_goals):
        """Calculates the standard World Football Elo goal differential index (G)."""
        goal_diff = abs(home_goals - away_goals)
        if goal_diff <= 1:
            return 1.0
        elif goal_diff == 2:
            return 1.5
        else:
            return (11.0 + goal_diff) / 8.0

    def fit(self, historical_matches_df):
        """
        Processes a dataframe chronologically, updating team ratings step-by-step
        up to the day before the tournament.
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
            w_home, w_away = self.calculate_expected_score(r_home, r_away)

            # 3. Map match outcomes (Win = 1.0, Draw = 0.5, Loss = 0.0)
            if h_goals > a_goals:
                actual_home, actual_away = 1.0, 0.0
            elif a_goals > h_goals:
                actual_home, actual_away = 0.0, 1.0
            else:
                actual_home, actual_away = 0.5, 0.5

            # 4. Compute the goal margin index scale factor
            g_factor = self.get_goal_margin_multiplier(h_goals, a_goals)

            # 5. Execute the delta adjustment update step
            self.ratings[home] += self.k_factor * g_factor * (actual_home - w_home)
            self.ratings[away] += self.k_factor * g_factor * (actual_away - w_away)

    def predict_match(self, home_team, away_team, baseline_goals=1.35, alpha=2.2):
        """
        Translates final Elo delta vectors back into discrete integer scorelines
        for DataCamp frame injection.
        """
        r_home = self.get_rating(home_team)
        r_away = self.get_rating(away_team)

        w_home, _ = self.calculate_expected_score(r_home, r_away)

        # Convert the continuous probability advantage into expected goal lambdas
        lambda_home = max(0.1, baseline_goals + alpha * (w_home - 0.5))
        lambda_away = max(0.1, baseline_goals + alpha * ((1.0 - w_home) - 0.5))

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
