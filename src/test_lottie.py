import streamlit as st
import json
from streamlit_lottie import st_lottie
import os

st.set_page_config(layout="wide")
st.title("🎨 Choix de l'animation Lottie")
st.caption("Dis-moi laquelle tu préfères — 1, 2 ou 3")
st.divider()

BASE = os.path.dirname(os.path.abspath(__file__))

col1, col2, col3 = st.columns(3)

for i, col in enumerate([col1, col2, col3], start=1):
    path = os.path.join(BASE, "assets", f"animation_{i}.json")
    with col:
        st.subheader(f"Animation {i}")
        try:
            with open(path, "r", encoding="utf-8") as f:
                anim = json.load(f)
            st_lottie(anim, height=250, key=f"anim_{i}", loop=True, speed=1)
        except Exception as e:
            st.error(f"Erreur : {e}")