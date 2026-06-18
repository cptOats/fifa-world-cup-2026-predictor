"""Machine Learning: Gradient-Boosted Count Tree (XGBoost) Model."""

import os

import numpy as np
import xgboost as xgb
from sklearn.metrics import d2_tweedie_score
from sklearn.model_selection import TimeSeriesSplit


def train_production_xgboost_models(feature_matrix, feature_columns, alpha=0.00047):
    """Trains production XGBoost goal-count models using chronological validation."""

    # 1. Isolate Features, Targets, and Temporal Anchor
    X = feature_matrix[feature_columns]
    y_home = feature_matrix["home_score"]
    y_away = feature_matrix["away_score"]

    # 2. Chronological Backtest Audit with Out-of-Fold Tracking
    tscv = TimeSeriesSplit(n_splits=3)

    oof_home_preds = np.zeros(len(feature_matrix))
    oof_away_preds = np.zeros(len(feature_matrix))
    cv_metrics = {}

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_h_train, y_h_test = y_home.iloc[train_idx], y_home.iloc[test_idx]
        y_a_train, y_a_test = y_away.iloc[train_idx], y_away.iloc[test_idx]

        # Calculate fold-specific decay relative to the training fold's maximum date
        train_dates = feature_matrix["match_date"].iloc[train_idx]
        fold_max_date = train_dates.max()
        days_elapsed_fold = (fold_max_date - train_dates).dt.days
        time_decay_fold = np.exp(-alpha * days_elapsed_fold)

        # Generate compound weights
        w_train = time_decay_fold * feature_matrix["match_weight"].iloc[train_idx]

        # Fit Home validation model
        h_cv_model = xgb.XGBRegressor(
            objective="count:poisson",
            n_estimators=60,
            max_depth=3,
            learning_rate=0.05,
            random_state=1989,
        )
        h_cv_model.fit(X_train, y_h_train, sample_weight=w_train)
        oof_home_preds[test_idx] = h_cv_model.predict(X_test)

        # Fit Away validation model
        a_cv_model = xgb.XGBRegressor(
            objective="count:poisson",
            n_estimators=60,
            max_depth=3,
            learning_rate=0.05,
            random_state=1989,
        )
        a_cv_model.fit(X_train, y_a_train, sample_weight=w_train)
        oof_away_preds[test_idx] = a_cv_model.predict(X_test)

        # Calculate Deviance R-squared
        dev_r2_h = d2_tweedie_score(y_h_test, oof_home_preds[test_idx], power=1)
        dev_r2_a = d2_tweedie_score(y_a_test, oof_away_preds[test_idx], power=1)

        cv_metrics[f"fold_{fold}"] = {
            "home_deviance_r2": dev_r2_h,
            "away_deviance_r2": dev_r2_a,
        }

    # 3. Production Model Fit (Anchored to the absolute latest pre-tournament data cliff)
    days_elapsed_prod = (
        feature_matrix["match_date"].max() - feature_matrix["match_date"]
    ).dt.days
    time_decay_prod = np.exp(-alpha * days_elapsed_prod)

    # Apply compound weights to production models
    w_prod = time_decay_prod * feature_matrix["match_weight"]

    xgb_params = {
        "objective": "count:poisson",
        "eval_metric": "poisson-nloglik",
        "n_estimators": 120,
        "max_depth": 4,
        "learning_rate": 0.04,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 1989,
    }

    model_home = xgb.XGBRegressor(**xgb_params)
    model_home.fit(X, y_home, sample_weight=w_prod)

    model_away = xgb.XGBRegressor(**xgb_params)
    model_away.fit(X, y_away, sample_weight=w_prod)

    # 4. Save Core Model Artifacts
    ARTIFACTS_DIR = os.path.join("data", "artifacts")
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    model_home.save_model(os.path.join(ARTIFACTS_DIR, "xgb_home_core.json"))
    model_away.save_model(os.path.join(ARTIFACTS_DIR, "xgb_away_core.json"))

    return model_home, model_away, oof_home_preds, oof_away_preds, cv_metrics
