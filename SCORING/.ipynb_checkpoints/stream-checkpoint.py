import streamlit as st
import pandas as pd
from exemple import generate_cluster_profiles, assign_cluster_from_rules

st.title("🧬 Analyse et Exploration des Clusters")

uploaded = st.file_uploader("Importer un DataFrame CSV", type="csv")

# Initialisation du state pour garder profils / règles
if "profiles" not in st.session_state:
    st.session_state["profiles"] = None
if "summaries" not in st.session_state:
    st.session_state["summaries"] = None
if "rules" not in st.session_state:
    st.session_state["rules"] = None


if uploaded:
    df = pd.read_csv(uploaded)
    st.subheader("Aperçu des données 📄")
    st.dataframe(df.head())

    st.subheader("Profilage des clusters 🔍")

    # Bouton pour générer les profils
    if st.button("Générer les profils"):
        profiles, summaries, rules = generate_cluster_profiles(df)

        st.session_state["profiles"] = profiles
        st.session_state["summaries"] = summaries
        st.session_state["rules"] = rules

        st.success("Profils générés !")

    # Interface si profils déjà générés
    if st.session_state["profiles"] is not None:

        profiles = st.session_state["profiles"]
        summaries = st.session_state["summaries"]
        rules = st.session_state["rules"]

        cluster_ids = list(profiles.keys())
        selected = st.selectbox("Choisir un cluster", cluster_ids)

        st.subheader(f"📊 Statistiques numériques – Cluster {selected}")
        st.dataframe(profiles[selected]["numeric"])

        st.subheader("📁 Variables catégorielles")
        for var, table in profiles[selected]["categorical"].items():
            st.write(f"### {var}")
            st.dataframe(table)

        st.subheader("📝 Résumé automatique")
        st.info(summaries[selected])

        st.subheader("🧩 Règles de segmentation")
        st.json(rules[selected])

    # --- Prédiction sur nouveau dataset ---
    st.subheader("Prédire le cluster sur un nouveau dataset 📌")
    uploaded2 = st.file_uploader("Importer un dataset à segmenter", type="csv", key="dataset2")

    if uploaded2 and st.session_state["rules"] is not None:
        df_new = pd.read_csv(uploaded2)
        result = assign_cluster_from_rules(df_new, st.session_state["rules"])

        st.dataframe(result)
        st.download_button(
            "Télécharger résultats",
            result.to_csv(index=False),
            "segmented.csv"
        )
