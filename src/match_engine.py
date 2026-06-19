"""Match Evaluation Engine."""

import numpy as np
import pandas as pd

from src.model_poisson import predict_poisson_match


def simulate_stochastic_match(
    lambda_h: float,
    lambda_a: float,
    elo_h: float,
    elo_a: float,
    rng: np.random.Generator,
    match_rules: dict[str, float],
    is_knockout: bool = False,
    n_runs: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Universal vectorized match simulator for Monte Carlo distributions.
    Handles Regulation, Extra Time fatigue scaling, and Elo-weighted penalty shootouts.
    """
    et_multiplier = match_rules["et_multiplier"]
    fatigue_factor = match_rules["fatigue_factor"]

    sims_h = rng.poisson(lambda_h, n_runs)
    sims_a = rng.poisson(lambda_a, n_runs)

    if not is_knockout:
        return sims_h, sims_a

    draw_mask = sims_h == sims_a
    n_draws = np.sum(draw_mask)

    if n_draws > 0:
        # 1. Add Extra Time with compound fatigue scaling
        sims_h[draw_mask] += rng.poisson(
            lambda_h * et_multiplier * fatigue_factor, n_draws
        )
        sims_a[draw_mask] += rng.poisson(
            lambda_a * et_multiplier * fatigue_factor, n_draws
        )

        # 2. Resolve Penalties for remaining draws
        shootout_mask = draw_mask & (sims_h == sims_a)
        n_shootouts = np.sum(shootout_mask)

        if n_shootouts > 0:
            # Symmetrical probabilistic tie-breaker based on relative Elo strength
            prob_h_win = elo_h / (elo_h + elo_a)
            sims_h[shootout_mask] += np.where(
                rng.random(n_shootouts) < prob_h_win, 1, 0
            )

    return sims_h, sims_a


def _resolve_consensus_math(
    home_team: str,
    away_team: str,
    venue_country: str,
    ratings: dict[str, dict[str, float]],
    g_home_avg: float,
    g_away_avg: float,
    g_neutral_avg: float,
    blend_weights: dict[str, float],
    elo_engine,
    xgb_h_pred: float,
    xgb_w_pred: float,
    use_prior_nudge: bool,
    nudge_strength: float,
    team_power: dict[str, int] | None,
) -> tuple[float, float, float, float, float]:
    """Pure mathematical resolver for obtaining model outputs without DataFrame overhead."""

    # 1. Resolve Dynamic Venue Neutrality
    is_neutral = 0 if (home_team == venue_country or away_team == venue_country) else 1

    # 2. Extract Poisson Intensity and Proxy Metrics
    lambda_home_poisson, lambda_away_poisson, p_corners, p_yellows, p_reds = (
        predict_poisson_match(
            home_team,
            away_team,
            venue_country,
            ratings,
            g_home_avg,
            g_away_avg,
            g_neutral_avg,
        )
    )

    # 3. Extract ELO Intensity
    elo_meta = elo_engine.predict_elo_match(home_team, away_team, is_neutral=is_neutral)

    # 4. Compute Consensus Expectations Vector
    raw_home = (
        (blend_weights["poisson"] * lambda_home_poisson)
        + (blend_weights["elo"] * float(elo_meta["lambda_home"]))
        + (blend_weights["xgb"] * xgb_h_pred)
    )
    raw_away = (
        (blend_weights["poisson"] * lambda_away_poisson)
        + (blend_weights["elo"] * float(elo_meta["lambda_away"]))
        + (blend_weights["xgb"] * xgb_w_pred)
    )

    # 5. Apply Bayesian Prior Nudge to the Continuous Intensities
    if use_prior_nudge and team_power:
        prior_nudge = (
            (team_power.get(home_team, 75) - team_power.get(away_team, 75))
            / 100
            * nudge_strength
        )
        raw_home = max(0.1, raw_home + prior_nudge)
        raw_away = max(0.1, raw_away - prior_nudge)

    return raw_home, raw_away, p_corners, p_yellows, p_reds


def evaluate_match_consensus(
    home_team: str,
    away_team: str,
    venue_country: str,
    ratings: dict[str, dict[str, float]],
    g_home_avg: float,
    g_away_avg: float,
    g_neutral_avg: float,
    blend_weights: dict[str, float],
    elo_engine,
    xgb_home,
    xgb_away,
    feature_columns: list[str],
    latest_team_form: dict[str, dict[str, float]],
    use_prior_nudge: bool = False,
    nudge_strength: float = 1.5,
    team_power: dict[str, int] | None = None,
) -> tuple[float, float, float, float, float]:
    """Computes goal intensities for a single match."""

    is_neutral = 0 if (home_team == venue_country or away_team == venue_country) else 1

    def _get_xgb_vec(h_t, a_t):
        return {
            "home_elo_rating": elo_engine.get_rating(h_t),
            "away_elo_rating": elo_engine.get_rating(a_t),
            "elo_differential": elo_engine.get_rating(h_t) - elo_engine.get_rating(a_t),
            "is_neutral_venue": is_neutral,
            "home_team_ewm_gf_4s": latest_team_form[h_t]["ewm_gf_4s"],
            "home_team_ewm_ga_4s": latest_team_form[h_t]["ewm_ga_4s"],
            "home_team_ewm_wr_4s": latest_team_form[h_t]["ewm_wr_4s"],
            "home_team_ewm_gf_10s": latest_team_form[h_t]["ewm_gf_10s"],
            "home_team_ewm_ga_10s": latest_team_form[h_t]["ewm_ga_10s"],
            "home_team_ewm_wr_10s": latest_team_form[h_t]["ewm_wr_10s"],
            "away_team_ewm_gf_4s": latest_team_form[a_t]["ewm_gf_4s"],
            "away_team_ewm_ga_4s": latest_team_form[a_t]["ewm_ga_4s"],
            "away_team_ewm_wr_4s": latest_team_form[a_t]["ewm_wr_4s"],
            "away_team_ewm_gf_10s": latest_team_form[a_t]["ewm_gf_10s"],
            "away_team_ewm_ga_10s": latest_team_form[a_t]["ewm_ga_10s"],
            "away_team_ewm_wr_10s": latest_team_form[a_t]["ewm_wr_10s"],
        }

    df_fwd = pd.DataFrame([_get_xgb_vec(home_team, away_team)])[feature_columns]
    df_swp = pd.DataFrame([_get_xgb_vec(away_team, home_team)])[feature_columns]

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

    # Pass the isolated XGBoost predictions into the shared math resolver
    return _resolve_consensus_math(
        home_team,
        away_team,
        venue_country,
        ratings,
        g_home_avg,
        g_away_avg,
        g_neutral_avg,
        blend_weights,
        elo_engine,
        xgb_h_pred,
        xgb_w_pred,
        use_prior_nudge,
        nudge_strength,
        team_power,
    )


def batch_evaluate_consensus(
    matchup_keys: list[tuple[str, str, str]],
    ratings: dict[str, dict[str, float]],
    g_home: float,
    g_away: float,
    g_neutral: float,
    blend_weights: dict[str, float],
    elo_engine,
    xgb_home,
    xgb_away,
    feature_columns: list[str],
    latest_team_form: dict[str, dict[str, float]],
    use_prior_nudge: bool = False,
    nudge_strength: float = 1.5,
    team_power: dict[str, int] | None = None,
) -> dict[tuple[str, str, str], tuple[float, float, float, float]]:
    """Vectorized batch inference for thousands of theoretical matchups."""

    rows_fwd, rows_swp = [], []

    def _build_xgb_dict(h, a, is_neut):
        return {
            "home_elo_rating": elo_engine.get_rating(h),
            "away_elo_rating": elo_engine.get_rating(a),
            "elo_differential": elo_engine.get_rating(h) - elo_engine.get_rating(a),
            "is_neutral_venue": is_neut,
            "home_team_ewm_gf_4s": latest_team_form[h]["ewm_gf_4s"],
            "home_team_ewm_ga_4s": latest_team_form[h]["ewm_ga_4s"],
            "home_team_ewm_wr_4s": latest_team_form[h]["ewm_wr_4s"],
            "home_team_ewm_gf_10s": latest_team_form[h]["ewm_gf_10s"],
            "home_team_ewm_ga_10s": latest_team_form[h]["ewm_ga_10s"],
            "home_team_ewm_wr_10s": latest_team_form[h]["ewm_wr_10s"],
            "away_team_ewm_gf_4s": latest_team_form[a]["ewm_gf_4s"],
            "away_team_ewm_ga_4s": latest_team_form[a]["ewm_ga_4s"],
            "away_team_ewm_wr_4s": latest_team_form[a]["ewm_wr_4s"],
            "away_team_ewm_gf_10s": latest_team_form[a]["ewm_gf_10s"],
            "away_team_ewm_ga_10s": latest_team_form[a]["ewm_ga_10s"],
            "away_team_ewm_wr_10s": latest_team_form[a]["ewm_wr_10s"],
        }

    for h, a, v_country in matchup_keys:
        is_neutral_flag = 0 if (h == v_country or a == v_country) else 1
        rows_fwd.append(_build_xgb_dict(h, a, is_neutral_flag))
        rows_swp.append(_build_xgb_dict(a, h, is_neutral_flag))

    df_fwd = pd.DataFrame(rows_fwd)[feature_columns]
    df_swp = pd.DataFrame(rows_swp)[feature_columns]

    # Batch execute both perspectives through XGBoost
    xgb_h_fwd = xgb_home.predict(df_fwd)
    xgb_a_fwd = xgb_away.predict(df_fwd)
    xgb_h_swp = xgb_home.predict(df_swp)
    xgb_a_swp = xgb_away.predict(df_swp)

    lambda_cache = {}

    # Iterate through the pre-calculated XGBoost arrays to resolve the final math
    for idx, (h, a, v_country) in enumerate(matchup_keys):
        # Isolate the correct XGBoost prediction logic for this specific row
        if h == v_country:
            xgb_h_val, xgb_a_val = xgb_h_fwd[idx], xgb_a_fwd[idx]
        elif a == v_country:
            xgb_h_val, xgb_a_val = xgb_a_swp[idx], xgb_h_swp[idx]
        else:
            xgb_h_val = (xgb_h_fwd[idx] + xgb_a_swp[idx]) / 2.0
            xgb_a_val = (xgb_a_fwd[idx] + xgb_h_swp[idx]) / 2.0

        # Pass the extracted XGBoost values into the shared pure math resolver
        l_h, l_a, c_exp, y_exp, _ = _resolve_consensus_math(
            h,
            a,
            v_country,
            ratings,
            g_home,
            g_away,
            g_neutral,
            blend_weights,
            elo_engine,
            xgb_h_val,
            xgb_a_val,
            use_prior_nudge,
            nudge_strength,
            team_power,
        )

        lambda_cache[(h, a, v_country)] = (l_h, l_a, c_exp, y_exp)

    return lambda_cache
