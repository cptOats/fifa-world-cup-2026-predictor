import os

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import d2_tweedie_score
from sklearn.model_selection import TimeSeriesSplit


def train_production_xgboost_models(feature_matrix, feature_columns, alpha=0.002):
    """
    Chronological cross-validation audit calculating out-of-fold predictions,
    applies exponential time-decay sample weighting, and serializes production models.
    """
    print("\n🏋️‍♂️ Initializing Gradient-Boosted Tree Training Matrix...")

    # 1. Isolate Features, Targets, and Temporal Anchor
    X = feature_matrix[feature_columns]
    y_home = feature_matrix["home_score"]
    y_away = feature_matrix["away_score"]

    # Calculate exponential decay weights based on days elapsed from the most recent match
    days_elapsed = (
        feature_matrix["match_date"].max() - feature_matrix["match_date"]
    ).dt.days
    sample_weights = np.exp(-alpha * days_elapsed)

    # 2. Chronological Backtest Audit with Out-of-Fold Tracking
    tscv = TimeSeriesSplit(n_splits=3)
    print("📋 Executing Chronological Cross-Validation Audit with Time-Decay...")

    oof_home_preds = np.zeros(len(feature_matrix))
    oof_away_preds = np.zeros(len(feature_matrix))

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_h_train, y_h_test = y_home.iloc[train_idx], y_home.iloc[test_idx]
        y_a_train, y_a_test = y_away.iloc[train_idx], y_away.iloc[test_idx]
        w_train = sample_weights.iloc[train_idx]

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
        print(
            f"   ↳ Fold {fold} -> Home Deviance R²: {dev_r2_h:.3f} | Away Deviance R²: {dev_r2_a:.3f}"
        )

    # 3. Define Production Hyperparameters
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

    print("\n🚀 Fitting final production models across full historical horizon...")

    # Train production models
    model_home = xgb.XGBRegressor(**xgb_params)
    model_home.fit(X, y_home, sample_weight=sample_weights)

    model_away = xgb.XGBRegressor(**xgb_params)
    model_away.fit(X, y_away, sample_weight=sample_weights)

    # 4. Save Core Model Artifacts
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    path_home = os.path.join(models_dir, "xgb_home_core.json")
    path_away = os.path.join(models_dir, "xgb_away_core.json")

    model_home.save_model(path_home)
    model_away.save_model(path_away)

    print(
        "💾 Machine learning models successfully serialized to the persistence layer:"
    )
    print(f"   - Home Goal Engine: {path_home}")
    print(f"   - Away Goal Engine: {path_away}")

    return model_home, model_away, oof_home_preds, oof_away_preds
