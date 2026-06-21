"""
Machine Learning: Gradient-Boosted Count Tree (XGBoost) Model.

Implements Count Poisson objective regression using time-decay weighting (alpha).
Includes strict TimeSeriesSplit cross-validation to isolate true out-of-fold
predictions and calculate Deviance R-Squared (D2 Tweedie) metrics for model observability.
"""

import json
import logging
import os

import numpy as np
import xgboost as xgb
from sklearn.metrics import d2_tweedie_score
from sklearn.model_selection import TimeSeriesSplit


def train_production_xgboost_models(
    feature_matrix, feature_columns, alpha=0.00047, cv_folds=3
):
    """
    Trains XGBoost structures with automated disk caching for validation states.

    Returns:
        tuple: (model_home, model_away, oof_home_preds, oof_away_preds, cv_metrics)
    """

    ARTIFACTS_DIR = os.path.join("data", "artifacts")
    XGB_OOF_H_PATH = os.path.join(ARTIFACTS_DIR, "xgb_oof_home.npy")
    XGB_OOF_A_PATH = os.path.join(ARTIFACTS_DIR, "xgb_oof_away.npy")
    XGB_METRICS_PATH = os.path.join(ARTIFACTS_DIR, "xgb_cv_metrics.json")

    # 1. Cache Hit Check (Fast-path avoidance of CV overhead)
    if (
        os.path.exists(XGB_OOF_H_PATH)
        and os.path.exists(XGB_OOF_A_PATH)
        and os.path.exists(XGB_METRICS_PATH)
    ):
        logging.info(
            "💾 Cached XGBoost Out-of-Fold arrays detected. Skipping cross-validation loops..."
        )
        oof_home_preds = np.load(XGB_OOF_H_PATH)
        oof_away_preds = np.load(XGB_OOF_A_PATH)
        with open(XGB_METRICS_PATH, "r") as f:
            cv_metrics = json.load(f)

        model_home = xgb.XGBRegressor()
        model_home.load_model(os.path.join(ARTIFACTS_DIR, "xgb_home_core.json"))
        model_away = xgb.XGBRegressor()
        model_away.load_model(os.path.join(ARTIFACTS_DIR, "xgb_away_core.json"))
        return model_home, model_away, oof_home_preds, oof_away_preds, cv_metrics

    # 2. Sequential Cross Validation Architecture
    X = feature_matrix[feature_columns]
    y_home = feature_matrix["home_score"]
    y_away = feature_matrix["away_score"]
    tscv = TimeSeriesSplit(n_splits=cv_folds)

    oof_home_preds = np.zeros(len(feature_matrix))
    oof_away_preds = np.zeros(len(feature_matrix))
    cv_metrics = {}

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        logging.info(
            f"🔄 Processing XGBoost Out-of-Fold Cross-Validation (Fold {fold}/{cv_folds})..."
        )

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_h_train, y_h_test = y_home.iloc[train_idx], y_home.iloc[test_idx]
        y_a_train, y_a_test = y_away.iloc[train_idx], y_away.iloc[test_idx]

        # Calculate fold-specific exponential decay relative to the training set boundary
        train_dates = feature_matrix["match_date"].iloc[train_idx]
        fold_max_date = train_dates.max()
        days_elapsed_fold = (fold_max_date - train_dates).dt.days
        time_decay_fold = np.exp(-alpha * days_elapsed_fold)
        w_train = time_decay_fold * feature_matrix["match_weight"].iloc[train_idx]

        # Home Context Tree
        h_cv_model = xgb.XGBRegressor(
            objective="count:poisson",
            n_estimators=60,
            max_depth=3,
            learning_rate=0.05,
            random_state=1989,
        )
        h_cv_model.fit(X_train, y_h_train, sample_weight=w_train)
        oof_home_preds[test_idx] = h_cv_model.predict(X_test)

        # Away Context Tree
        a_cv_model = xgb.XGBRegressor(
            objective="count:poisson",
            n_estimators=60,
            max_depth=3,
            learning_rate=0.05,
            random_state=1989,
        )
        a_cv_model.fit(X_train, y_a_train, sample_weight=w_train)
        oof_away_preds[test_idx] = a_cv_model.predict(X_test)

        # Calculate Deviance R-squared for Count Poisson distributions
        dev_r2_h = d2_tweedie_score(y_h_test, oof_home_preds[test_idx], power=1)
        dev_r2_a = d2_tweedie_score(y_a_test, oof_away_preds[test_idx], power=1)
        cv_metrics[f"fold_{fold}"] = {
            "home_deviance_r2": dev_r2_h,
            "away_deviance_r2": dev_r2_a,
        }

    # 3. Fit Production Models on all available data
    days_elapsed_prod = (
        feature_matrix["match_date"].max() - feature_matrix["match_date"]
    ).dt.days
    time_decay_prod = np.exp(-alpha * days_elapsed_prod)
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

    # 4. Save Artifacts
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    model_home.save_model(os.path.join(ARTIFACTS_DIR, "xgb_home_core.json"))
    model_away.save_model(os.path.join(ARTIFACTS_DIR, "xgb_away_core.json"))
    np.save(XGB_OOF_H_PATH, oof_home_preds)
    np.save(XGB_OOF_A_PATH, oof_away_preds)
    with open(XGB_METRICS_PATH, "w") as f:
        json.dump(cv_metrics, f, indent=4)

    return model_home, model_away, oof_home_preds, oof_away_preds, cv_metrics
