# src/models/train_model.py

import pandas as pd
import numpy as np
from loguru import logger
import os
import json

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import joblib

INPUT_PATH  = "src/etl/data/processed/dpe_paris_clean.csv"
MODEL_PATH  = "src/etl/data/db/model_xgb.pkl"
SHAP_PATH   = "src/etl/data/processed/shap_values.png"

FEATURES = [
    "age_batiment",
    "surface_habitable_logement",
    "hauteur_sous_plafond",
    "score_dpe",
    "score_isolation",
    "conso_norm",
    "deperditions_enveloppe",
    "deperditions_murs",
    "emission_ges_5_usages",
    "cout_total_5_usages",
    "nombre_niveau_logement",
    "nombre_niveau_immeuble",
]

TARGET = "score_degradation"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")
    logger.info(f"📂 Chargé : {df.shape}")
    return df


def prepare(df: pd.DataFrame):
    # Garde uniquement les features disponibles
    feats = [f for f in FEATURES if f in df.columns]
    X = df[feats].copy()
    y = df[TARGET].copy()

    # Remplace NaN résiduels
    X = X.fillna(X.median())

    logger.info(f"✅ Features utilisées : {feats}")
    logger.info(f"📊 X shape : {X.shape} | y shape : {y.shape}")
    return X, y, feats


def train(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )

    model.fit(X_train, y_train)

    # Évaluation
    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")

    logger.info(f"🎯 MAE  : {mae:.2f} points")
    logger.info(f"🎯 R²   : {r2:.3f}")
    logger.info(f"🎯 CV R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    return model, X_test, y_test, y_pred


def explain(model, X, feats):
    logger.info("🔍 Calcul SHAP...")
    explainer   = shap.Explainer(model)
    shap_values = explainer(X)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, show=False)
    os.makedirs("data/processed", exist_ok=True)
    plt.tight_layout()
    plt.savefig(SHAP_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    logger.success(f"📊 SHAP sauvegardé → {SHAP_PATH}")


def save_model(model):
    os.makedirs("data/db", exist_ok=True)
    joblib.dump(model, MODEL_PATH.replace(".json", ".pkl"))
    logger.success(f"💾 Modèle sauvegardé → {MODEL_PATH.replace('.json', '.pkl')}")


if __name__ == "__main__":
    df          = load_data()
    X, y, feats = prepare(df)
    model, X_test, y_test, y_pred = train(X, y)
    explain(model, X, feats)
    save_model(model)

    # Aperçu des prédictions
    results = X_test.copy()
    results["score_reel"]  = y_test.values
    results["score_predit"] = y_pred.round(1)
    results["ecart"]        = (results["score_predit"] - results["score_reel"]).abs().round(1)
    print("\n── Aperçu prédictions ──")
    print(results[["score_reel", "score_predit", "ecart"]].head(10).to_string())