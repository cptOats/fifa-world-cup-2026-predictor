import os

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit


def train_production_xgboost_models(feature_matrix, feature_columns):
    """
    Splits targets, executes a chronological cross-validation audit,
    trains independent native count:poisson XGBoost regressors,
    and saves the production-ready models to disk.
    """
    print("\n🏋️‍♂️ Initializing Gradient-Boosted Tree Training Matrix...")

    # 1. Isolate Features and Independent Targets
    X = feature_matrix[feature_columns]
    y_home = feature_matrix["home_score"]
    y_away = feature_matrix["away_score"]

    # 2. Chronological Backtest Audit (TimeSeriesSplit prevents data leakage)
    tscv = TimeSeriesSplit(n_splits=3)
    print("📋 Executing Chronological Cross-Validation Audit...")

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_h_train, y_h_test = y_home.iloc[train_idx], y_home.iloc[test_idx]
        y_a_train, y_a_test = y_away.iloc[train_idx], y_away.iloc[test_idx]

        # Quick validation check using baseline models
        h_base = xgb.XGBRegressor(
            objective="count:poisson", n_estimators=50, max_depth=3, learning_rate=0.05
        )
        h_base.fit(X_train, y_h_train)
        preds = h_base.predict(X_test)
        mae = mean_absolute_error(y_h_test, preds)
        print(f"   ↳ Fold {fold} -> Home Goals Validation MAE: {mae:.3f}")

    # 3. Define the Production Hyperparameters for Count Data
    # Using count:poisson forces predictions to be strictly positive log-linear bounds
    xgb_params = {
        "objective": "count:poisson",
        "eval_metric": "poisson-nloglik",
        "n_estimators": 120,
        "max_depth": 4,
        "learning_rate": 0.04,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 1989,  # Dynamic demographical echo anchor
    }

    print("\n🚀 Fitting final production models across full historical horizon...")

    # Train independent Home Model
    model_home = xgb.XGBRegressor(**xgb_params)
    model_home.fit(X, y_home)

    # Train independent Away Model
    model_away = xgb.XGBRegressor(**xgb_params)
    model_away.fit(X, y_away)

    # 4. Save Core Model Artifacts to Disk
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

    return model_home, model_away
