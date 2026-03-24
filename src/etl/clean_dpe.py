# src/etl/clean_dpe.py

import pandas as pd
import numpy as np
from loguru import logger
import os

INPUT_PATH  = "data/raw/dpe_paris.csv"
OUTPUT_PATH = "data/processed/dpe_paris_clean.csv"


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")
    logger.info(f"📂 Chargé : {df.shape[0]} lignes | {df.shape[1]} colonnes")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:

    # ── 1. Supprime les doublons
    avant = len(df)
    df = df.drop_duplicates(subset="numero_dpe")
    logger.info(f"🧹 Doublons supprimés : {avant - len(df)}")

    # ── 2. Année construction : remplace NaN par médiane
    df["annee_construction"] = pd.to_numeric(df["annee_construction"], errors="coerce")
    mediane = df["annee_construction"].median()
    df["annee_construction"] = df["annee_construction"].fillna(mediane)
    logger.info(f"📅 annee_construction — médiane : {mediane}")

    # ── 3. Surface : supprime les valeurs aberrantes (<5m² ou >1000m²)
    df["surface_habitable_logement"] = pd.to_numeric(
        df["surface_habitable_logement"], errors="coerce"
    )
    avant = len(df)
    df = df[df["surface_habitable_logement"].between(5, 1000, inclusive="both")]
    logger.info(f"📐 Surfaces aberrantes supprimées : {avant - len(df)}")

    # ── 4. Colonnes numériques : remplace NaN par médiane
    cols_num = [
        "hauteur_sous_plafond",
        "deperditions_enveloppe",
        "deperditions_murs",
        "conso_5_usages_par_m2_ep",
        "emission_ges_5_usages",
        "cout_total_5_usages",
        "nombre_niveau_logement",
        "nombre_niveau_immeuble",
    ]
    for col in cols_num:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    # ── 5. Colonnes catégorielles : remplace NaN par "Inconnu"
    cols_cat = [
        "etiquette_dpe",
        "etiquette_ges",
        "type_batiment",
        "qualite_isolation_enveloppe",
        "qualite_isolation_murs",
        "qualite_isolation_menuiseries",
        "qualite_isolation_plancher_bas",
        "type_energie_principale_chauffage",
        "classe_inertie_batiment",
        "zone_climatique",
        "periode_construction",
    ]
    for col in cols_cat:
        if col in df.columns:
            df[col] = df[col].fillna("Inconnu")

    # ── 6. Feature Engineering — Score de dégradation (0-100)
    logger.info("⚙️ Calcul du score de dégradation...")

    # Age du bâtiment (normalisé 0-1)
    annee_actuelle = 2026
    df["age_batiment"] = annee_actuelle - df["annee_construction"]
    df["age_norm"] = (df["age_batiment"] / 150).clip(0, 1)

    # Score isolation (0=très bonne, 1=insuffisante)
    isolation_map = {
        "très bonne": 0.0,
        "bonne": 0.25,
        "moyenne": 0.5,
        "insuffisante": 1.0,
        "Inconnu": 0.5,
    }
    df["score_isolation"] = df["qualite_isolation_enveloppe"].str.lower().map(
        isolation_map
    ).fillna(0.5)

    # Score DPE (A=0, G=1)
    dpe_map = {"A": 0.0, "B": 0.17, "C": 0.33, "D": 0.5,
               "E": 0.67, "F": 0.83, "G": 1.0, "Inconnu": 0.5}
    df["score_dpe"] = df["etiquette_dpe"].map(dpe_map).fillna(0.5)

    # Consommation normalisée (0-1, cap à 500 kWh/m²)
    df["conso_norm"] = (df["conso_5_usages_par_m2_ep"] / 500).clip(0, 1)

    # ── Score final pondéré (0-100)
    df["score_degradation"] = (
        df["age_norm"]       * 0.35 +
        df["score_isolation"] * 0.30 +
        df["score_dpe"]      * 0.25 +
        df["conso_norm"]     * 0.10
    ) * 100

    df["score_degradation"] = df["score_degradation"].round(1)

    # ── Niveau de risque
    df["niveau_risque"] = pd.cut(
        df["score_degradation"],
        bins=[0, 33, 66, 100],
        labels=["Faible", "Modéré", "Élevé"],
        include_lowest=True
    )

    logger.info(f"✅ Nettoyage terminé : {df.shape[0]} lignes | {df.shape[1]} colonnes")
    logger.info(f"\n{df['niveau_risque'].value_counts()}")
    return df


def save_clean(df: pd.DataFrame) -> None:
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    logger.success(f"💾 Sauvegardé → {OUTPUT_PATH}")


if __name__ == "__main__":
    df = load_raw()
    df = clean(df)
    save_clean(df)
    print(df[["adresse_ban", "annee_construction",
              "etiquette_dpe", "score_degradation",
              "niveau_risque"]].head(10).to_string())
    print(f"\nShape final : {df.shape}")