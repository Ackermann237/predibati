# src/etl/store_sql.py

import pandas as pd
import sqlite3
from loguru import logger
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV_PATH = os.path.join(BASE_DIR, "src", "etl", "data", "processed", "dpe_paris_clean.csv")
DB_PATH  = os.path.join(BASE_DIR, "src", "etl", "data", "db", "predibati.db")


def create_db(df: pd.DataFrame) -> None:
    """Stocke les données DPE dans SQLite avec schéma optimisé."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # ── Création table principale
    cursor.executescript("""
        DROP TABLE IF EXISTS batiments;
        CREATE TABLE batiments (
            id                          INTEGER PRIMARY KEY AUTOINCREMENT,
            numero_dpe                  TEXT,
            date_etablissement_dpe      TEXT,
            annee_construction          REAL,
            periode_construction        TEXT,
            type_batiment               TEXT,
            etiquette_dpe               TEXT,
            etiquette_ges               TEXT,
            adresse_ban                 TEXT,
            code_postal_ban             TEXT,
            nom_commune_ban             TEXT,
            surface_habitable_logement  REAL,
            hauteur_sous_plafond        REAL,
            qualite_isolation_enveloppe TEXT,
            qualite_isolation_murs      TEXT,
            type_energie_principale_chauffage TEXT,
            conso_5_usages_par_m2_ep    REAL,
            emission_ges_5_usages       REAL,
            cout_total_5_usages         REAL,
            age_batiment                REAL,
            score_degradation           REAL,
            niveau_risque               TEXT
        );
    """)

    # ── Insertion des données
    cols = [
        "numero_dpe","date_etablissement_dpe","annee_construction",
        "periode_construction","type_batiment","etiquette_dpe","etiquette_ges",
        "adresse_ban","code_postal_ban","nom_commune_ban",
        "surface_habitable_logement","hauteur_sous_plafond",
        "qualite_isolation_enveloppe","qualite_isolation_murs",
        "type_energie_principale_chauffage","conso_5_usages_par_m2_ep",
        "emission_ges_5_usages","cout_total_5_usages",
        "age_batiment","score_degradation","niveau_risque"
    ]
    cols_present = [c for c in cols if c in df.columns]
    df[cols_present].to_sql("batiments", conn, if_exists="append", index=False)

    # ── Index pour performances
    cursor.executescript("""
        CREATE INDEX IF NOT EXISTS idx_code_postal ON batiments(code_postal_ban);
        CREATE INDEX IF NOT EXISTS idx_etiquette   ON batiments(etiquette_dpe);
        CREATE INDEX IF NOT EXISTS idx_risque      ON batiments(niveau_risque);
        CREATE INDEX IF NOT EXISTS idx_score       ON batiments(score_degradation DESC);
    """)

    conn.commit()
    conn.close()
    logger.success(f"✅ Base SQLite créée → {DB_PATH}")


def run_queries(db_path: str) -> None:
    """Démontre des requêtes SQL analytiques sur les données DPE."""
    conn = sqlite3.connect(db_path)

    queries = {
        "📊 Nb bâtiments par niveau de risque": """
            SELECT niveau_risque,
                   COUNT(*)                    AS nb_batiments,
                   ROUND(AVG(score_degradation),1) AS score_moyen
            FROM batiments
            GROUP BY niveau_risque
            ORDER BY score_moyen DESC;
        """,

        "🏆 Top 10 bâtiments les plus dégradés": """
            SELECT adresse_ban,
                   code_postal_ban,
                   etiquette_dpe,
                   ROUND(score_degradation,1) AS score,
                   niveau_risque,
                   annee_construction
            FROM batiments
            ORDER BY score_degradation DESC
            LIMIT 10;
        """,

        "🗺️ Score moyen par arrondissement": """
            SELECT code_postal_ban,
                   COUNT(*)                        AS nb,
                   ROUND(AVG(score_degradation),1) AS score_moyen,
                   ROUND(AVG(conso_5_usages_par_m2_ep),1) AS conso_moyenne
            FROM batiments
            GROUP BY code_postal_ban
            ORDER BY score_moyen DESC;
        """,

        "⚡ Répartition DPE": """
            SELECT etiquette_dpe,
                   COUNT(*)                        AS nb,
                   ROUND(AVG(score_degradation),1) AS score_moyen,
                   ROUND(AVG(cout_total_5_usages),0) AS cout_moyen
            FROM batiments
            GROUP BY etiquette_dpe
            ORDER BY etiquette_dpe;
        """,

        "🏗️ Bâtiments avant 1948 à risque élevé": """
            SELECT adresse_ban, code_postal_ban,
                   annee_construction,
                   etiquette_dpe,
                   ROUND(score_degradation,1) AS score
            FROM batiments
            WHERE annee_construction < 1948
              AND niveau_risque = 'Élevé'
            ORDER BY score_degradation DESC
            LIMIT 15;
        """,

        "💰 Coût moyen par qualité d'isolation": """
            SELECT qualite_isolation_enveloppe,
                   COUNT(*)                         AS nb,
                   ROUND(AVG(cout_total_5_usages),0) AS cout_moyen,
                   ROUND(AVG(score_degradation),1)  AS score_moyen
            FROM batiments
            GROUP BY qualite_isolation_enveloppe
            ORDER BY score_moyen DESC;
        """
    }

    for title, sql in queries.items():
        print(f"\n{'='*60}")
        print(f"  {title}")
        print('='*60)
        df = pd.read_sql_query(sql, conn)
        print(df.to_string(index=False))

    conn.close()


if __name__ == "__main__":
    logger.info("📂 Chargement des données...")
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    logger.info(f"✅ {len(df)} lignes chargées")

    create_db(df)
    run_queries(DB_PATH)
    logger.success("🎯 Pipeline SQL terminé !")