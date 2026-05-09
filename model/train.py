"""
Model Training Script
Trains Linear Regression and Random Forest, selects the best performer.
Run: py model/train.py
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Config ──────────────────────────────────────────────
FEATURES = [
    'distance_km',
    'pickup_delay_min',
    'traffic_score',
    'weather_score',
    'delivery_person_age',
    'delivery_person_ratings',
    'vehicle_condition',
    'multiple_deliveries',
    'order_hour',
    'is_peak_hour',
]
TARGET = 'time_takenmin'
DATA_PATH = Path(__file__).parent.parent / 'cleaned_delivery_data.csv'
MODEL_PATH = Path(__file__).parent.parent / 'delivery_time_model.pkl'
META_PATH = Path(__file__).parent.parent / 'model_metadata.json'


def load_data():
    """Load and prepare data for training."""
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    # Drop rows with NaN in features or target
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]

    return X, y


def evaluate_model(model, X_test, y_test):
    """Compute evaluation metrics."""
    predictions = model.predict(X_test)
    return {
        'r2': round(r2_score(y_test, predictions), 4),
        'mae': round(mean_absolute_error(y_test, predictions), 4),
        'rmse': round(np.sqrt(mean_squared_error(y_test, predictions)), 4),
    }


def cross_validate(model, X, y, cv=5):
    """Run cross-validation and return mean R²."""
    scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    return round(scores.mean(), 4), round(scores.std(), 4)


def train():
    """Train models, compare, and save the best one."""
    print("=" * 60)
    print("🚀 Food Delivery — Model Training Pipeline")
    print("=" * 60)

    # Load data
    X, y = load_data()
    print(f"\n📊 Dataset: {len(X)} samples, {len(FEATURES)} features")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   Train: {len(X_train)} | Test: {len(X_test)}")

    # ── Train Linear Regression ──
    print("\n─── Linear Regression ───")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_metrics = evaluate_model(lr, X_test, y_test)
    lr_cv_mean, lr_cv_std = cross_validate(lr, X, y)
    print(f"   R²: {lr_metrics['r2']}  |  MAE: {lr_metrics['mae']} min  |  RMSE: {lr_metrics['rmse']} min")
    print(f"   CV R² (5-fold): {lr_cv_mean} ± {lr_cv_std}")

    # ── Train Random Forest ──
    print("\n─── Random Forest ───")
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_metrics = evaluate_model(rf, X_test, y_test)
    rf_cv_mean, rf_cv_std = cross_validate(rf, X, y)
    print(f"   R²: {rf_metrics['r2']}  |  MAE: {rf_metrics['mae']} min  |  RMSE: {rf_metrics['rmse']} min")
    print(f"   CV R² (5-fold): {rf_cv_mean} ± {rf_cv_std}")

    # ── Select best model ──
    models = {
        'LinearRegression': (lr, lr_metrics, lr_cv_mean),
        'RandomForest': (rf, rf_metrics, rf_cv_mean),
    }

    best_name = max(models, key=lambda k: models[k][2])  # Best CV R²
    best_model, best_metrics, best_cv = models[best_name]

    print(f"\n✅ Best Model: {best_name} (CV R²: {best_cv})")

    # ── Feature importance ──
    if best_name == 'RandomForest':
        importances = dict(zip(FEATURES, best_model.feature_importances_.round(4).tolist()))
    else:
        importances = dict(zip(FEATURES, best_model.coef_.round(4).tolist()))

    print("\n📈 Feature Importances:")
    for feat, imp in sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"   {feat}: {imp}")

    # ── Save model (compressed for GitHub) ──
    joblib.dump(best_model, MODEL_PATH, compress=3)
    import os
    model_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"\n💾 Model saved: {MODEL_PATH} ({model_size_mb:.1f} MB)")

    # ── Save metadata ──
    metadata = {
        'model_type': best_name,
        'features': FEATURES,
        'target': TARGET,
        'test_metrics': best_metrics,
        'cv_r2_mean': best_cv,
        'feature_importances': importances,
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'all_results': {
            'LinearRegression': {
                'test': lr_metrics,
                'cv_r2': lr_cv_mean,
            },
            'RandomForest': {
                'test': rf_metrics,
                'cv_r2': rf_cv_mean,
            },
        }
    }

    with open(META_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"📄 Metadata saved: {META_PATH}")

    print("\n" + "=" * 60)
    print("✅ Training complete!")
    print("=" * 60)

    return best_model, metadata


if __name__ == '__main__':
    train()
