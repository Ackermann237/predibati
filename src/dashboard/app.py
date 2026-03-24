# src/dashboard/app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import folium
from streamlit_folium import st_folium
from pyproj import Transformer
from groq import Groq
from dotenv import load_dotenv
from streamlit_lottie import st_lottie
import joblib
import json
import os

load_dotenv()

st.set_page_config(
    page_title="PrediBâti — Maintenance Prédictive",
    page_icon="🏗️",
    layout="wide"
)

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH  = os.path.join(BASE_DIR, "src", "etl", "data", "processed", "dpe_paris_clean.csv")
MODEL_PATH = os.path.join(BASE_DIR, "src", "etl", "data", "db", "model_xgb.pkl")
LOTTIE_PATH = os.path.join(BASE_DIR, "src", "assets", "animation_2.json")

# ══════════════════════════════════════
# LOTTIE
# ══════════════════════════════════════
@st.cache_data
def load_lottie():
    with open(LOTTIE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

# ══════════════════════════════════════
# DONNÉES
# ══════════════════════════════════════
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df["niveau_risque"] = df["niveau_risque"].astype(str)
    return df

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def convert_coords(df):
    transformer = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)
    dff = df.dropna(subset=["coordonnee_cartographique_x_ban","coordonnee_cartographique_y_ban"]).copy()
    coords = dff.apply(
        lambda r: transformer.transform(
            r["coordonnee_cartographique_x_ban"],
            r["coordonnee_cartographique_y_ban"]
        ), axis=1
    )
    dff["lon"] = coords.apply(lambda c: c[0])
    dff["lat"] = coords.apply(lambda c: c[1])
    return dff[dff["lat"].between(48.7,49.0) & dff["lon"].between(2.2,2.5)]

lottie_anim = load_lottie()
df          = load_data()
model       = load_model()
df_geo      = convert_coords(df)

# ══════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════
with st.sidebar:
    st.title("🏗️ PrediBâti")
    st.caption("Maintenance Prédictive · Paris IDF")
    st.divider()

    st.subheader("🔍 Filtres")

    arrondissements = sorted(df["code_postal_ban"].dropna().astype(str).unique())
    tout_arr = st.checkbox("Tous les arrondissements", value=True)
    if tout_arr:
        arr_select = arrondissements
    else:
        arr_select = st.multiselect("Choisir", arrondissements, default=arrondissements[:3])

    st.markdown("")

    etiquettes = sorted(df["etiquette_dpe"].dropna().unique())
    tout_dpe = st.checkbox("Toutes les étiquettes DPE", value=True)
    if tout_dpe:
        dpe_select = etiquettes
    else:
        dpe_select = st.multiselect("Choisir", etiquettes, default=etiquettes, key="dpe")

    st.markdown("")

    st.markdown("**Niveau de risque**")
    cb_eleve  = st.checkbox("🔴 Élevé",  value=True)
    cb_modere = st.checkbox("🟡 Modéré", value=True)
    cb_faible = st.checkbox("🟢 Faible", value=True)
    risque_select = (
        (["Élevé"]  if cb_eleve  else []) +
        (["Modéré"] if cb_modere else []) +
        (["Faible"] if cb_faible else [])
    )

    st.markdown("")

    st.markdown("**Score de dégradation**")
    score_range = st.slider("", 0, 100, (0, 100), label_visibility="collapsed")

    st.divider()
    st.caption("Données · ADEME Open Data")
    st.caption("Modèle · XGBoost R²=0.966")
    st.caption("Auteur · André Amougou")

# ══════════════════════════════════════
# FILTRAGE
# ══════════════════════════════════════
mask = (
    df["code_postal_ban"].astype(str).isin(arr_select) &
    df["etiquette_dpe"].isin(dpe_select) &
    df["niveau_risque"].isin(risque_select) &
    df["score_degradation"].between(score_range[0], score_range[1])
)
dff = df[mask].copy()

dff_geo = df_geo[
    df_geo["code_postal_ban"].astype(str).isin(arr_select) &
    df_geo["etiquette_dpe"].isin(dpe_select) &
    df_geo["niveau_risque"].isin(risque_select) &
    df_geo["score_degradation"].between(score_range[0], score_range[1])
].copy()

nb_total  = len(dff)
nb_eleve  = len(dff[dff["niveau_risque"] == "Élevé"])
nb_modere = len(dff[dff["niveau_risque"] == "Modéré"])
nb_faible = len(dff[dff["niveau_risque"] == "Faible"])
score_moy = dff["score_degradation"].mean() if nb_total else 0
age_moy   = dff["age_batiment"].mean() if nb_total else 0

# ══════════════════════════════════════
# ONGLETS
# ══════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Vue Globale",
    "🗺️ Carte & Analyse",
    "🤖 Simulateur IA",
    "💬 Agent IA"
])

# ──────────────────────────────────────
# TAB 1 — VUE GLOBALE
# ──────────────────────────────────────
with tab1:
    col_anim, col_title = st.columns([1, 6])
    with col_anim:
        st_lottie(lottie_anim, height=120, width=120, key="lottie_tab1", loop=True, speed=1)
    with col_title:
        st.header("Vue Globale du Parc Immobilier")
        st.caption("Analyse des diagnostics DPE — Bâtiments parisiens")
    st.divider()

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("🏠 Bâtiments",     nb_total)
    k2.metric("🔴 Risque élevé",  nb_eleve,
              delta=f"{nb_eleve/nb_total*100:.0f}% du parc" if nb_total else "0%",
              delta_color="inverse")
    k3.metric("🟡 Risque modéré", nb_modere)
    k4.metric("🟢 Risque faible", nb_faible)
    k5.metric("📊 Score moyen",   f"{score_moy:.1f} / 100")

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Distribution des scores")
        fig_hist = px.histogram(
            dff, x="score_degradation", color="niveau_risque",
            color_discrete_map={"Élevé":"#ef4444","Modéré":"#f59e0b","Faible":"#22c55e"},
            nbins=20, template="plotly_white",
            labels={"score_degradation":"Score","count":"Nb bâtiments"}
        )
        fig_hist.update_layout(height=280, margin=dict(t=10,b=10),
                               showlegend=True, legend=dict(orientation="h",y=-0.2))
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        st.subheader("Répartition DPE")
        dpe_counts = dff["etiquette_dpe"].value_counts().reset_index()
        dpe_counts.columns = ["DPE","count"]
        fig_dpe = px.bar(
            dpe_counts, x="DPE", y="count", color="DPE",
            color_discrete_map={"A":"#16a34a","B":"#65a30d","C":"#ca8a04",
                                "D":"#ea580c","E":"#dc2626","F":"#9f1239","G":"#4c0519"},
            template="plotly_white"
        )
        fig_dpe.update_layout(height=280, margin=dict(t=10,b=10), showlegend=False)
        st.plotly_chart(fig_dpe, use_container_width=True)

    with col3:
        st.subheader("Niveau de risque")
        risque_counts = dff["niveau_risque"].value_counts().reset_index()
        risque_counts.columns = ["Risque","count"]
        fig_pie = px.pie(
            risque_counts, names="Risque", values="count",
            color="Risque",
            color_discrete_map={"Élevé":"#ef4444","Modéré":"#f59e0b","Faible":"#22c55e"},
            hole=0.45, template="plotly_white"
        )
        fig_pie.update_layout(height=280, margin=dict(t=10,b=10))
        st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()

    col4, col5 = st.columns([3, 2])

    with col4:
        st.subheader("Score de dégradation vs Âge")
        fig_sc = px.scatter(
            dff, x="age_batiment", y="score_degradation",
            color="niveau_risque",
            color_discrete_map={"Élevé":"#ef4444","Modéré":"#f59e0b","Faible":"#22c55e"},
            size="surface_habitable_logement",
            hover_data=["adresse_ban","etiquette_dpe","annee_construction"],
            labels={"age_batiment":"Âge (ans)","score_degradation":"Score"},
            template="plotly_white", opacity=0.8
        )
        fig_sc.update_layout(height=320, margin=dict(t=10,b=10),
                             legend=dict(orientation="h",y=-0.2))
        st.plotly_chart(fig_sc, use_container_width=True)

    with col5:
        st.subheader("Score par étiquette DPE")
        fig_box = px.box(
            dff, x="etiquette_dpe", y="score_degradation",
            color="etiquette_dpe",
            color_discrete_map={"A":"#16a34a","B":"#65a30d","C":"#ca8a04",
                                "D":"#ea580c","E":"#dc2626","F":"#9f1239","G":"#4c0519"},
            labels={"etiquette_dpe":"DPE","score_degradation":"Score"},
            template="plotly_white"
        )
        fig_box.update_layout(height=320, margin=dict(t=10,b=10), showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

    st.divider()

    st.subheader("🚨 Top 10 bâtiments prioritaires")
    top10 = dff.nlargest(10, "score_degradation")[[
        "adresse_ban","code_postal_ban","annee_construction",
        "etiquette_dpe","score_degradation","niveau_risque",
        "qualite_isolation_enveloppe","type_energie_principale_chauffage"
    ]].rename(columns={
        "adresse_ban":"Adresse","code_postal_ban":"CP",
        "annee_construction":"Année","etiquette_dpe":"DPE",
        "score_degradation":"Score","niveau_risque":"Risque",
        "qualite_isolation_enveloppe":"Isolation",
        "type_energie_principale_chauffage":"Énergie"
    })
    st.dataframe(top10, use_container_width=True, hide_index=True)


# ──────────────────────────────────────
# TAB 2 — CARTE & ANALYSE
# ──────────────────────────────────────
with tab2:
    st.header("🗺️ Carte des risques — Paris")
    st.caption("Chaque point = un bâtiment. Taille proportionnelle au score. Cliquez pour les détails.")
    st.divider()

    col_map, col_stats = st.columns([3, 1])

    with col_map:
        m = folium.Map(location=[48.8566, 2.3522], zoom_start=12, tiles="CartoDB positron")
        couleurs = {"Élevé":"#ef4444","Modéré":"#f59e0b","Faible":"#22c55e"}

        for _, row in dff_geo.iterrows():
            couleur = couleurs.get(row["niveau_risque"],"gray")
            score   = row["score_degradation"]
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=5 + (score/100)*8,
                color=couleur, fill=True,
                fill_color=couleur, fill_opacity=0.75, weight=1.5,
                popup=folium.Popup(
                    f"<b>{row.get('adresse_ban','N/A')}</b><br>"
                    f"Score : <b>{score}/100</b><br>"
                    f"DPE : <b>{row['etiquette_dpe']}</b> | "
                    f"Risque : <b>{row['niveau_risque']}</b><br>"
                    f"Année : <b>{int(row['annee_construction'])}</b>",
                    max_width=220
                )
            ).add_to(m)

        st_folium(m, width=None, height=520, returned_objects=[])

    with col_stats:
        st.subheader("Légende")
        st.markdown("🔴 **Élevé** — Score > 66")
        st.markdown("🟡 **Modéré** — Score 33–66")
        st.markdown("🟢 **Faible** — Score < 33")
        st.divider()

        st.subheader("Stats carte")
        st.metric("Points affichés", len(dff_geo))
        st.metric("Score max", f"{dff_geo['score_degradation'].max():.1f}" if not dff_geo.empty else "N/A")
        st.metric("Score min", f"{dff_geo['score_degradation'].min():.1f}" if not dff_geo.empty else "N/A")
        st.metric("Score moy", f"{dff_geo['score_degradation'].mean():.1f}" if not dff_geo.empty else "N/A")
        st.divider()

        st.subheader("Par arrondissement")
        if not dff.empty:
            arr_stats = (
                dff.groupby("code_postal_ban")["score_degradation"]
                .mean().round(1).reset_index()
                .rename(columns={"code_postal_ban":"CP","score_degradation":"Score moy"})
                .sort_values("Score moy", ascending=False)
            )
            st.dataframe(arr_stats, use_container_width=True, hide_index=True, height=280)

    st.divider()

    col_p1, col_p2 = st.columns(2)

    with col_p1:
        st.subheader("📈 Score moyen par période de construction")
        if "periode_construction" in dff.columns:
            periode_stats = (
                dff.groupby("periode_construction")["score_degradation"]
                .agg(["mean","count"]).reset_index()
                .rename(columns={"periode_construction":"Période","mean":"Score moyen","count":"Nb"})
                .sort_values("Score moyen", ascending=False)
            )
            fig_per = px.bar(
                periode_stats, x="Période", y="Score moyen",
                color="Score moyen",
                color_continuous_scale=["#22c55e","#f59e0b","#ef4444"],
                template="plotly_white", text="Score moyen"
            )
            fig_per.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            fig_per.update_layout(height=320, margin=dict(t=10,b=80),
                                  xaxis_tickangle=-30, coloraxis_showscale=False)
            st.plotly_chart(fig_per, use_container_width=True)

    with col_p2:
        st.subheader("🧱 Score moyen par qualité d'isolation")
        fig_iso = px.bar(
            dff.groupby("qualite_isolation_enveloppe")["score_degradation"]
            .mean().round(1).reset_index()
            .rename(columns={"qualite_isolation_enveloppe":"Isolation","score_degradation":"Score moyen"}),
            x="Isolation", y="Score moyen", color="Score moyen",
            color_continuous_scale=["#22c55e","#f59e0b","#ef4444"],
            template="plotly_white", text="Score moyen"
        )
        fig_iso.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        fig_iso.update_layout(height=320, margin=dict(t=10,b=10), coloraxis_showscale=False)
        st.plotly_chart(fig_iso, use_container_width=True)


# ──────────────────────────────────────
# TAB 3 — SIMULATEUR IA
# ──────────────────────────────────────
with tab3:
    st.header("🤖 Simulateur — Prédiction en temps réel")
    st.caption("Entrez les caractéristiques d'un bâtiment pour prédire son score de dégradation.")
    st.divider()

    col_form, col_result = st.columns([2, 1])

    with col_form:
        st.subheader("Caractéristiques du bâtiment")
        r1c1, r1c2 = st.columns(2)

        with r1c1:
            annee   = st.number_input("📅 Année de construction", 1800, 2024, 1950)
            surface = st.number_input("📐 Surface habitable (m²)", 10, 500, 65)
            hauteur = st.number_input("📏 Hauteur sous plafond (m)", 2.0, 5.0, 2.7)
            dpe_sim = st.selectbox("🏷️ Étiquette DPE", ["A","B","C","D","E","F","G"], index=4)
            iso_sim = st.selectbox("🧱 Qualité isolation",
                                   ["très bonne","bonne","moyenne","insuffisante"], index=2)
            conso_s = st.number_input("⚡ Consommation énergie (kWh/m²/an)", 0, 800, 250)

        with r1c2:
            dep_env  = st.number_input("🌡️ Déperditions enveloppe (W/K)", 0, 500, 120)
            dep_mur  = st.number_input("🧱 Déperditions murs (W/K)", 0, 300, 80)
            emission = st.number_input("💨 Émissions GES (kgCO2/an)", 0, 2000, 500)
            cout     = st.number_input("💶 Coût total 5 usages (€/an)", 0, 5000, 1500)
            niv_log  = st.number_input("🏢 Niveaux du logement", 1, 10, 1)
            niv_imm  = st.number_input("🏙️ Niveaux de l'immeuble", 1, 20, 6)

        predict_btn = st.button("🔮 Prédire", use_container_width=True, type="primary")

    with col_result:
        st.subheader("Résultat")

        if predict_btn:
            dpe_map = {"A":0.0,"B":0.17,"C":0.33,"D":0.5,"E":0.67,"F":0.83,"G":1.0}
            iso_map = {"très bonne":0.0,"bonne":0.25,"moyenne":0.5,"insuffisante":1.0}

            X_sim = pd.DataFrame([{
                "age_batiment": 2026 - annee,
                "surface_habitable_logement": surface,
                "hauteur_sous_plafond": hauteur,
                "score_dpe": dpe_map[dpe_sim],
                "score_isolation": iso_map[iso_sim],
                "conso_norm": min(conso_s/500, 1),
                "deperditions_enveloppe": dep_env,
                "deperditions_murs": dep_mur,
                "emission_ges_5_usages": emission,
                "cout_total_5_usages": cout,
                "nombre_niveau_logement": niv_log,
                "nombre_niveau_immeuble": niv_imm,
            }])

            score_pred = model.predict(X_sim)[0]
            niveau = "🔴 Élevé"  if score_pred > 66 else \
                     "🟡 Modéré" if score_pred > 33 else "🟢 Faible"

            st.metric("🎯 Score prédit",     f"{score_pred:.1f} / 100")
            st.metric("⚠️ Niveau de risque", niveau)
            st.metric("📅 Âge du bâtiment",  f"{2026-annee} ans")
            st.metric("🏷️ DPE sélectionné", dpe_sim)
            st.divider()

            fig_gauge = px.bar(
                x=[score_pred], orientation="h",
                color_discrete_sequence=["#ef4444" if score_pred>66 else "#f59e0b" if score_pred>33 else "#22c55e"],
                range_x=[0,100], template="plotly_white"
            )
            fig_gauge.update_layout(
                height=80, margin=dict(t=5,b=5,l=5,r=5),
                xaxis_title="Score /100", showlegend=False,
                yaxis=dict(showticklabels=False)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            st.divider()
            st.subheader("💡 Recommandation")
            if score_pred > 66:
                st.error("🚨 **Intervention urgente requise**\n\nInspection et travaux de rénovation recommandés dans les 12 mois.")
            elif score_pred > 33:
                st.warning("⚠️ **Surveillance recommandée**\n\nPlanifier une inspection dans les 24 mois.")
            else:
                st.success("✅ **État satisfaisant**\n\nMaintenir le plan de maintenance préventive standard.")
        else:
            st.info("👈 Renseignez les caractéristiques et cliquez sur **Prédire**.")
            st.subheader("📖 Guide des scores")
            st.markdown("""
| Score | Niveau | Action |
|-------|--------|--------|
| 0–33  | 🟢 Faible  | Maintenance standard |
| 33–66 | 🟡 Modéré  | Surveillance renforcée |
| 66–100| 🔴 Élevé   | Intervention urgente |
            """)


# ──────────────────────────────────────
# TAB 4 — AGENT LLM
# ──────────────────────────────────────
with tab4:
    col_anim4, col_title4 = st.columns([1, 6])
    with col_anim4:
        st_lottie(lottie_anim, height=100, width=100, key="lottie_tab4", loop=True, speed=1)
    with col_title4:
        st.header("💬 Agent IA — Analyse en langage naturel")
        st.caption("Posez vos questions sur le parc immobilier parisien. L'agent analyse les données en temps réel.")
    st.divider()

    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        st.error("❌ Clé GROQ_API_KEY manquante dans le fichier .env")
        st.stop()

    client = Groq(api_key=groq_api_key)

    def build_context(df: pd.DataFrame) -> str:
        nb = len(df)
        if nb == 0:
            return "Aucune donnée disponible avec les filtres actuels."

        nb_eleve = len(df[df["niveau_risque"] == "Élevé"])
        nb_mod   = len(df[df["niveau_risque"] == "Modéré"])
        nb_fai   = len(df[df["niveau_risque"] == "Faible"])
        score_m  = df["score_degradation"].mean()
        age_m    = df["age_batiment"].mean()

        top5 = df.nlargest(5, "score_degradation")[
            ["adresse_ban","score_degradation","etiquette_dpe","niveau_risque","annee_construction"]
        ].to_string(index=False)

        dpe_dist = df["etiquette_dpe"].value_counts().to_string()
        iso_dist = df["qualite_isolation_enveloppe"].value_counts().to_string()
        arr_dist = (
            df.groupby("code_postal_ban")["score_degradation"]
            .mean().round(1).sort_values(ascending=False).head(5).to_string()
        )

        return f"""
Tu es un expert en maintenance prédictive immobilière pour la société Oxand.
Tu analyses le parc immobilier parisien à partir de données DPE (Diagnostic de Performance Énergétique).
Le modèle de prédiction utilisé est XGBoost avec un R²=0.966.

## Données actuelles (filtrées par l'utilisateur)
- Nombre de bâtiments analysés : {nb}
- Risque élevé  : {nb_eleve} bâtiments ({nb_eleve/nb*100:.1f}%)
- Risque modéré : {nb_mod} bâtiments ({nb_mod/nb*100:.1f}%)
- Risque faible : {nb_fai} bâtiments ({nb_fai/nb*100:.1f}%)
- Score de dégradation moyen : {score_m:.1f}/100
- Âge moyen du parc : {age_m:.0f} ans

## Top 5 bâtiments les plus dégradés
{top5}

## Distribution des étiquettes DPE
{dpe_dist}

## Qualité d'isolation
{iso_dist}

## Score moyen par arrondissement (top 5)
{arr_dist}

Réponds en français, de manière concise et professionnelle.
Appuie tes réponses sur les données ci-dessus.
Si une question dépasse les données disponibles, dis-le clairement.
"""

    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.subheader("💡 Questions suggérées")
    col_s1, col_s2, col_s3 = st.columns(3)
    suggestions = [
        "Quels sont les bâtiments les plus urgents à rénover ?",
        "Quel est le lien entre l'étiquette DPE et le score ?",
        "Quels arrondissements ont le parc le plus dégradé ?",
        "Combien de bâtiments nécessitent une intervention urgente ?",
        "Quelle est l'influence de l'isolation sur le score ?",
        "Donne-moi un résumé exécutif du parc analysé.",
    ]

    for i, col in enumerate([col_s1, col_s2, col_s3]):
        with col:
            if st.button(suggestions[i], use_container_width=True, key=f"sug_{i}"):
                st.session_state.messages.append({"role":"user","content":suggestions[i]})
            if st.button(suggestions[i+3], use_container_width=True, key=f"sug_{i+3}"):
                st.session_state.messages.append({"role":"user","content":suggestions[i+3]})

    st.divider()

    st.subheader("🗨️ Conversation")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Posez votre question sur le parc immobilier..."):
        st.session_state.messages.append({"role":"user","content":prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        with st.chat_message("assistant"):
            with st.spinner("Analyse en cours..."):
                context = build_context(dff)
                messages_api = [{"role":"system","content":context}] + [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ]
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages_api,
                    max_tokens=1000,
                    temperature=0.3,
                )
                answer = response.choices[0].message.content
                st.markdown(answer)
                st.session_state.messages.append({"role":"assistant","content":answer})

    if st.session_state.messages:
        if st.button("🗑️ Effacer la conversation"):
            st.session_state.messages = []
            st.rerun()


st.divider()
st.caption("PrediBâti · André Amougou · Données ADEME Open Data · Modèle XGBoost R²=0.966")