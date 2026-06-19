"""Match Evaluation Engine."""

import numpy as np
import pandas as pd
from scipy.stats import norm, poisson

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
    copula_rho: float = 0.08,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """
    Universal vectorized match simulator for Monte Carlo distributions.
    Features Bivariate Copula Draw Inflation and Phase-of-Victory Masking.
    """
    et_multiplier = match_rules["et_multiplier"]
    fatigue_factor = match_rules["fatigue_factor"]

    # --- THE COPULA: BIVARIATE DRAW INFLATION ---
    if copula_rho > 0.0:
        # 1. Generate correlated standard normals
        cov_matrix = [[1.0, copula_rho], [copula_rho, 1.0]]
        z = rng.multivariate_normal([0.0, 0.0], cov_matrix, size=n_runs)

        # 2. Convert standard normals to uniform probabilities
        u = norm.cdf(z)

        # 3. Map uniforms to Poisson marginals via Point Percentile Function (Inverse CDF)
        sims_h = poisson.ppf(u[:, 0], lambda_h).astype(int)
        sims_a = poisson.ppf(u[:, 1], lambda_a).astype(int)
    else:
        # Fallback to independent framework
        sims_h = rng.poisson(lambda_h, n_runs)
        sims_a = rng.poisson(lambda_a, n_runs)

    # Initialize empty tracking vectors for all possible outcomes
    phase_meta = {
        "win_90_h": np.zeros(n_runs, dtype=bool),
        "win_90_a": np.zeros(n_runs, dtype=bool),
        "win_120_h": np.zeros(n_runs, dtype=bool),
        "win_120_a": np.zeros(n_runs, dtype=bool),
        "win_pen_h": np.zeros(n_runs, dtype=bool),
        "win_pen_a": np.zeros(n_runs, dtype=bool),
    }

    if not is_knockout:
        return sims_h, sims_a, phase_meta

    # --- PHASE A: REGULATION (90 MINUTES) ---
    phase_meta["win_90_h"] = sims_h > sims_a
    phase_meta["win_90_a"] = sims_a > sims_h

    draw_mask = sims_h == sims_a
    n_draws = int(np.sum(draw_mask))

    shootout_mask = np.zeros(n_runs, dtype=bool)
    n_shootouts = 0

    if n_draws > 0:
        # 1. Add Extra Time goals with compound fatigue scaling
        et_h = rng.poisson(lambda_h * et_multiplier * fatigue_factor, n_draws)
        et_a = rng.poisson(lambda_a * et_multiplier * fatigue_factor, n_draws)

        sims_h[draw_mask] += et_h
        sims_a[draw_mask] += et_a

        # --- PHASE B: EXTRA TIME (120 MINUTES) ---
        phase_meta["win_120_h"] = draw_mask & (sims_h > sims_a)
        phase_meta["win_120_a"] = draw_mask & (sims_a > sims_h)

        shootout_mask = draw_mask & (sims_h == sims_a)
        n_shootouts = int(np.sum(shootout_mask))

    if n_shootouts > 0:
        prob_h_win = lambda_h / (lambda_h + lambda_a)

        pen_h_win = rng.random(n_shootouts) < prob_h_win
        pen_a_win = np.logical_not(pen_h_win)

        sims_h[shootout_mask] += np.where(pen_h_win, 1, 0)
        sims_a[shootout_mask] += np.where(pen_a_win, 1, 0)

        # --- PHASE C: PENALTY SHOOTOUT ---
        phase_meta["win_pen_h"] = shootout_mask & (sims_h > sims_a)
        phase_meta["win_pen_a"] = shootout_mask & (sims_a > sims_h)

    return sims_h, sims_a, phase_meta


def simulate_deterministic_match(
    raw_home: float,
    raw_away: float,
    tot_corners_90: float,
    tot_yellows_90: float,
    tot_reds_90: float,
    match_rules: dict[str, float],
    is_knockout: bool = False,
) -> tuple[int, int, str, int, int, int, bool, bool]:
    """
    Resolves a deterministic match timeline (90m -> 120m -> Penalties).
    Returns integer goals, discipline metrics, and progression flags.
    """
    pred_home_90 = int(np.round(raw_home))
    pred_away_90 = int(np.round(raw_away))

    is_extra_time = False
    is_penalty = False

    # --- PHASE A: REGULATION (OR GROUP STAGE) ---
    if not is_knockout or pred_home_90 != pred_away_90:
        final_home_goals = pred_home_90
        final_away_goals = pred_away_90
        tot_corners = int(np.clip(np.round(tot_corners_90), 4, 16))
        tot_yellows = int(np.clip(np.round(tot_yellows_90), 1, 9))
        tot_reds = int(np.clip(np.round(tot_reds_90), 0, 3))

        if final_home_goals > final_away_goals:
            winner_side = "home"
        elif final_away_goals > final_home_goals:
            winner_side = "away"
        else:
            winner_side = "draw"

        return (
            final_home_goals,
            final_away_goals,
            winner_side,
            tot_corners,
            tot_yellows,
            tot_reds,
            is_extra_time,
            is_penalty,
        )

    # --- PHASE B: EXTRA TIME (KNOCKOUT DRAWS ONLY) ---
    is_extra_time = True
    et_multiplier = match_rules.get("et_multiplier", 0.333)
    fatigue_factor = match_rules.get("fatigue_factor", 0.8)

    raw_home_120 = raw_home * (1 + (et_multiplier * fatigue_factor))
    raw_away_120 = raw_away * (1 + (et_multiplier * fatigue_factor))

    final_home_goals = int(np.round(raw_home_120))
    final_away_goals = int(np.round(raw_away_120))

    tot_corners = int(
        np.clip(
            np.round(tot_corners_90 * (1 + (et_multiplier * fatigue_factor))), 5, 18
        )
    )
    tot_yellows = int(
        np.clip(
            np.round(tot_yellows_90 * (1 + (et_multiplier * fatigue_factor))), 1, 12
        )
    )
    tot_reds = int(
        np.clip(np.round(tot_reds_90 * (1 + (et_multiplier * fatigue_factor))), 0, 4)
    )

    if final_home_goals > final_away_goals:
        winner_side = "home"
    elif final_away_goals > final_home_goals:
        winner_side = "away"
    else:
        # --- PHASE C: PENALTIES ---
        is_penalty = True
        # Utilize the comprehensive floating-point ensemble blend as the ultimate tie-breaker
        if raw_home >= raw_away:
            winner_side = "home"
        else:
            winner_side = "away"

    return (
        final_home_goals,
        final_away_goals,
        winner_side,
        tot_corners,
        tot_yellows,
        tot_reds,
        is_extra_time,
        is_penalty,
    )


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
