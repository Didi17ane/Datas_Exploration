import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Titre dashboard
st.title("Dashboard d'Analyse de la Population Locataire")

# Chargement des données CSV
uploaded_file = st.file_uploader("Importer votre fichier CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Aperçu rapide
    st.subheader("Aperçu des données")
    st.write(df.head())

    # -- FILTRES --
    st.sidebar.header("Filtres")

    # Filtres géographiques
    region = st.sidebar.selectbox("Région", options=["Tous"] + sorted(df['region_name'].dropna().unique().tolist()))
    milieu_options = ["Tous"] + sorted(df['milieu_resid'].dropna().unique().tolist())
    milieu = st.sidebar.selectbox("Milieu résidentiel", options=milieu_options)

    sexe_options = ["Tous"] + sorted(df['sex'].dropna().unique().tolist())
    sexe = st.sidebar.selectbox("Sexe", options=sexe_options)

    # Filtre âge (numérique)
    age_min, age_max = int(df['age_num'].min()), int(df['age_num'].max())
    age_range = st.sidebar.slider("Tranche d'âge", age_min, age_max, (age_min, age_max))

    marital_options = ["Tous"] + sorted(df['marital_status'].dropna().unique().tolist())
    marital = st.sidebar.selectbox("Statut matrimonial", options=marital_options)

    # Filtre accès électricité (Valeurs Oui/Non strictes)
    elec_options = ["Tous"] + sorted(df['elec_ac'].dropna().unique().tolist())
    elec = st.sidebar.selectbox("Accès à l'électricité", options=elec_options)

    # Appliquer filtres sur dataframe
    df_filt = df[
        ((df['region_name'] == region) | (region == "Tous")) &
        ((df['milieu_resid'] == milieu) | (milieu == "Tous")) &
        ((df['sex'] == sexe) | (sexe == "Tous")) &
        (df['age_num'] >= age_range[0]) &
        (df['age_num'] <= age_range[1]) &
        ((df['marital_status'] == marital) | (marital == "Tous")) &
        ((df['elec_ac'] == elec) | (elec == "Tous"))
    ]

    # -- INDICATEURS CLÉS --
    st.subheader("Indicateurs clés")
    col1, col2, col3 = st.columns(3)
    col1.metric("Population filtrée", len(df_filt))
    col2.metric("Revenu moyen (FCFA)", round(df_filt['salaire_mois'].mean(), 0))
    # Pour le taux de bancarisation on vérifie aussi la présence de "Oui"/"Non" dans la colonne
    bancarise_yes_count = df_filt['bancarise'].value_counts().get("Oui", 0)
    bancarise_total_count = df_filt['bancarise'].count()
    bancarise_rate = (bancarise_yes_count / bancarise_total_count) * 100 if bancarise_total_count > 0 else 0
    col3.metric("Taux de bancarisation (%)", round(bancarise_rate, 2))

    # -- VISUALISATIONS --

    st.subheader("Pyramide des âges (par tranches)")

    # On s'assure que age_grp est bien catégorielle
    df_filt['age_grp'] = df_filt['age_grp'].astype(str)
    
    # Compter par tranche d'âge et sexe
    age_sex = df_filt.groupby(['age_grp', 'sex']).size().unstack(fill_value=0)
    
    # Pour avoir un ordre cohérent, ordonner les tranches selon votre logique métier
    # Exemple d'ordre si vos tranches ressemblent à '0-17', '18-24', '25-34', ...
    ordre_age = sorted(age_sex.index, key=lambda x: int(x.split('-')[0]) if '-' in x else 0)
    age_sex = age_sex.reindex(ordre_age)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(age_sex.index, -age_sex.get('M', pd.Series(0, index=age_sex.index)), color='blue', label='Hommes')
    ax.barh(age_sex.index, age_sex.get('F', pd.Series(0, index=age_sex.index)), color='red', label='Femmes')
    ax.set_xlabel("Population")
    ax.set_ylabel("Tranche d'âge")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig)

    # Revenu moyen par région
    st.subheader("Revenu moyen par région")
    revenu_region = df_filt.groupby('region_name')['salaire_mois'].mean().reset_index()
    fig2 = px.bar(revenu_region, x='region_name', y='salaire_mois',
                  labels={'salaire_mois': 'Revenu moyen (FCFA)', 'region_name': 'Région'},
                  title="Revenu moyen par Région")
    st.plotly_chart(fig2, use_container_width=True)

    # Répartition par statut matrimonial
    st.subheader("Répartition par statut matrimonial")
    stat_marital_count = df_filt['marital_status'].value_counts().reset_index()
    stat_marital_count.columns = ['Statut matrimonial', 'Effectif']
    fig3 = px.pie(stat_marital_count, names='Statut matrimonial', values='Effectif', title="Statut matrimonial")
    st.plotly_chart(fig3, use_container_width=True)

    # Taux d'accès à l'électricité
    st.subheader("Taux d'accès à l'électricité")
    elec_counts = df_filt['elec_ac'].value_counts(normalize=True) * 100
    for val in ["Oui", "Non"]:
        st.write(f"{val} : {elec_counts.get(val, 0):.1f} %")

    # Distribution des salaires
    st.subheader("Distribution des revenus mensuels")
    fig4, ax4 = plt.subplots()
    sns.histplot(df_filt['salaire_mois'], bins=30, kde=True, ax=ax4)
    ax4.set_xlabel("Revenu mensuel (FCFA)")
    st.pyplot(fig4)

else:
    st.info("Veuillez importer un fichier CSV pour commencer l'analyse.")
