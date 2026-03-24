# src/etl/fetch_dpe.py

import requests
import pandas as pd
from loguru import logger
from urllib.parse import urlparse, parse_qs, unquote
import os

API_URL = "https://data.ademe.fr/data-fair/api/v1/datasets/dpe03existant/lines"
OUTPUT_PATH = "data/raw/dpe_paris.csv"

COLONNES_UTILES = [
    "numero_dpe",
    "date_etablissement_dpe",
    "annee_construction",
    "periode_construction",
    "type_batiment",
    "etiquette_dpe",
    "etiquette_ges",
    "adresse_ban",
    "code_postal_ban",
    "nom_commune_ban",
    "coordonnee_cartographique_x_ban",
    "coordonnee_cartographique_y_ban",
    "surface_habitable_logement",
    "hauteur_sous_plafond",
    "qualite_isolation_enveloppe",
    "qualite_isolation_murs",
    "qualite_isolation_menuiseries",
    "qualite_isolation_plancher_bas",
    "deperditions_enveloppe",
    "deperditions_murs",
    "type_energie_principale_chauffage",
    "conso_5_usages_par_m2_ep",
    "emission_ges_5_usages",
    "cout_total_5_usages",
    "classe_inertie_batiment",
    "zone_climatique",
    "nombre_niveau_logement",
    "nombre_niveau_immeuble",
]


def fetch_dpe(max_records: int = 5000) -> pd.DataFrame:
    all_data = []
    total_fetched = 0
    after = None

    logger.info("🚀 Démarrage ingestion DPE ADEME...")

    while total_fetched < max_records:

        params = {"size": 100}
        if after:
            params["after"] = after

        try:
            response = requests.get(API_URL, params=params, timeout=30)
        except Exception as e:
            logger.error(f"Erreur réseau : {e}")
            break

        if response.status_code != 200:
            logger.error(f"Erreur {response.status_code} : {response.text[:200]}")
            break

        data = response.json()
        results = data.get("results", [])

        if not results:
            logger.info("Fin des données.")
            break

        all_data.extend(results)
        total_fetched += len(results)
        logger.info(f"✅ {total_fetched} lignes récupérées...")

        # ✅ Extraction propre du curseur de pagination
        next_url = data.get("next")
        if next_url and "after=" in next_url:
            parsed = urlparse(next_url)
            qs = parse_qs(parsed.query)
            after = unquote(qs.get("after", [None])[0])
        else:
            after = None
            break

    df = pd.DataFrame(all_data)
    logger.info(f"📊 Brut : {len(df)} lignes | {df.shape[1]} colonnes")

    # Filtre Paris (75xxx)
    if "code_postal_ban" in df.columns:
        df = df[df["code_postal_ban"].astype(str).str.startswith("75")]
        logger.info(f"🗼 Après filtre Paris : {len(df)} lignes")

    # Garde uniquement les colonnes utiles disponibles
    cols = [c for c in COLONNES_UTILES if c in df.columns]
    df = df[cols]

    logger.info(f"📦 Final : {len(df)} lignes | {df.shape[1]} colonnes")
    return df


def save_raw(df: pd.DataFrame) -> None:
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    logger.success(f"💾 Sauvegardé → {OUTPUT_PATH}")


if __name__ == "__main__":
    df = fetch_dpe(max_records=5000)
    if not df.empty:
        save_raw(df)
        print(df.head(3).to_string())
        print(f"\nShape : {df.shape}")
        print(f"\nColonnes : {list(df.columns)}")
    else:
        logger.error("❌ DataFrame vide")