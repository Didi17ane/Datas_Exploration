#Importations des bibliothéques

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from PIL import Image

import plotly.express as px
import plotly.graph_objects as go

from datetime import date, time, datetime, timedelta
from numerize.numerize import numerize
from streamlit_elements import elements, mui, html
import altair as alt

import folium
from streamlit_folium import st_folium

import pydeck as pdk

import joblib
from sklearn.metrics import precision_score, recall_score, r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
# ___________________________________________

#### Page Configuration ####
st.set_page_config(
    page_title="Scoring Locataires Tylimmo",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

alt.themes.enable("dark")

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
    }
    .chart-container {
        background-color: white;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


st.markdown('<h1 class="main-header">Analytics Dashboard</h1>', unsafe_allow_html=True)



# ------------------------ Charger les données

df = pd.read_csv("../DATAS/CleanALL_EHCVM.csv")
df_ehcvm = df.copy()

print(f"Dataset de visualisation : {df_ehcvm}")


# ------------------------------------------------------ AFFICHAGE DANS LA PAGE --------------------------------------------------

# ------------------------- Ajout de la Sidebar

with st.sidebar:
    # st.image("", width=200)
    st.title("Dashboard DIATA.ai")
    
    st.divider()

    # Bases de données
    st.subheader("Base de données")
    database = st.multiselect(
        "Database",
        options=["EHCVM2021", ""],
        default=["EHCVM2021"]
    )
    
    # Filters
    st.subheader("Filters")
    
    region = st.sidebar.selectbox("Région", ["Toutes"] + sorted(df_ehcvm["region"].unique().tolist()))
    age_grp = st.sidebar.selectbox("Tranche d’âge", ["Toutes"] + sorted(df_ehcvm["age_grp"].unique().tolist()))
    sexe = st.sidebar.selectbox("Sexe", ["Tous"] + sorted(df_ehcvm["sexe"].unique().tolist()))
    mstat = st.sidebar.selectbox("Statut matrimonial", ["Tous"] + sorted(df_ehcvm["mstat"].unique().tolist()))
    
    data = df_ehcvm.copy()
    # if "Toutes" not in region:
    #    data = data[data["region"].isin(region)]
    if region != "Toutes":
        data = data[data["region"] == region]
    if age_grp != "Toutes":
        data = data[data["age_grp"] == age_grp]
    if sexe != "Tous":
        data = data[data["sexe"] == sexe]
    if mstat != "Tous":
        data = data[data["mstat"] == mstat]
    
    
    # Advanced options
    st.subheader("Advanced Options")
    show_targets = st.checkbox("Show Targets", value=True)
    show_forecasts = st.checkbox("Show Forecasts", value=False)
    
    st.divider()
    st.markdown("© 2025 DIATA AFRICA SAS")

# ------------------------ PAGE PRINCIPALE

    
# ------------------------------ KPIs
# KPIs

st.subheader("Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)


with col1:
    population = len(data)
    st.metric(
        label="Total Population",
        value=f"{population:,.0f} ",   
    )
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    empl = round(data['empl_formel'].mean()*100, 1)
    if np.isnan(empl):
        empl=0
    st.metric("Emploi formel", f"{empl} %")
    # print(f"bak :{empl}")
    # print(type(empl).__name__)
    
with col2:
    df_femme = data[data["sexe"]=="Féminin"]
    femme = len(df_femme)
    
    if femme != 0:
        pourc_fem = round((femme / population)*100,1)
    st.metric(
        label="Total femmes",
        value=f"{femme:,.0f}",
    )
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    if femme == 0:
        pourc_fem = 0
    st.metric("% Femmes", f"{pourc_fem} %")

with col3:
    df_homme = data[data["sexe"]=="Masculin"]
    homme = len(df_homme)
    
    if homme != 0:
        pour_hom = round((homme / population)*100,1)
    st.metric(
        label="Total hommes",
        value=f"{homme:,.0f} ",
    )
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    if homme == 0:
        pour_hom = 0
    st.metric("% Hommes", f"{pour_hom} %")
    
with col4:
    rev = round(data['rev_total_mois'].mean(), 0)
    if np.isnan(rev):
        rev=0    
    st.metric("Revenu moyen (FCFA)", f"{rev:,.0f}")
    
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    bank = round(data['bancarise'].mean()*100, 1)
    if np.isnan(bank):
        bank=0    
    st.metric("Bancarisation", f"{bank} %")
    

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------  NAVIGATIONS PAGES ------------------



st.subheader("PERFORMANCES")
st.markdown("")
st.markdown("Cette vision globale permet d'identifier les zones fortes/faibles, profils majoritaires, contrastes majeurs. Incontournable avant tout projet de scoring.")
tab1, tab2, tab3 = st.tabs(["Vue générale de la population", "KPIs locatifs & financiers", "Exploration Score"])


# --------------------------- 1ere page ---------------------
with tab1:

    # Vue globale de la population
    
    c1, c2= st.columns(2)
    with c1:
        # Pyramide des âges
        st.subheader(":green[**Pyramide des âges**]")
        st.divider()
        age_sex = data.groupby(["age_grp","sexe"])["region"].count().unstack().fillna(0)
        age_sex.plot(kind="barh", stacked=True)
        st.pyplot(plt)
    with c2:
        # Zoom région: répartition des effectifs et indicateurs
        tab_region = data.groupby("region").agg(population=("region", "size"),
                                            revenu_moy=("rev_total_mois","mean"),
                                            bank_moy=("bancarise","mean")).reset_index()
         
        st.subheader(":green[**Comparaison des régions - Population & Revenus**]")
        st.divider()
        fig, ax = plt.subplots()
        tab_region.set_index("region")[["population", "revenu_moy"]].plot(kind="bar", ax=ax, secondary_y="rev_moy")

        st.pyplot(fig)
            
    # Groupes d’âge : population et revenu médian
    st.subheader(":green[**Effectifs & revenu moyen par âge**]")
        
    age_grp = data.groupby("age_grp").agg(population=("region", "size"), revenu_moy=("rev_total_mois","mean")).reset_index()
    st.dataframe(age_grp)

    ca1, ca2 = st.columns(2)
    with ca1:
        #Catégorie socioprofessionnelle
        
        st.subheader(":green[**Catégorie socioprofessionnelle**]")
        st.divider()
        csp_counts = data['csp'].value_counts()
        palette = sns.color_palette("tab20", len(csp_counts))
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(csp_counts.index, csp_counts.values, color=palette)
        
        # Annoter chaque barre avec son effectif
        for bar, value in zip(bars, csp_counts.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f'{value:,}',  # format séparateur milliers
                ha='center', va='bottom',
                fontsize=12, fontweight='bold'
            )
        
        ax.set_ylabel("Effectif", fontsize=15)
        ax.set_xlabel("")
        plt.xticks(rotation=45, ha='right', fontsize=13)
        sns.despine()
        plt.tight_layout()
        
        st.pyplot(fig)
        
    with ca2:
        # Secteur institutionnel
        
        st.subheader(":green[**Secteur institutionnel**]")
        st.divider()
        branch_counts = data['branch'].value_counts()
        palette = sns.color_palette("tab20", len(branch_counts))
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(branch_counts.index, branch_counts.values, color=palette)
        
        # Annoter chaque barre avec son effectif
        for bar, value in zip(bars, branch_counts.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f'{value:,}',  # format séparateur milliers
                ha='center', va='bottom',
                fontsize=12, fontweight='bold'
            )
        
        ax.set_ylabel("Effectif", fontsize=15)
        ax.set_xlabel("")
        plt.xticks(rotation=45, ha='right', fontsize=13)
        sns.despine()
        plt.tight_layout()
        
        st.pyplot(fig)
        
# --------------------------- 2ere page ---------------------

with tab2:
    
    # KPIs locatifs & financiers

    # Répartition par région et indicateurs clés
    st.subheader(":green[**📈 Répartition régionale**]")
    st.divider()
    
    st.dataframe(tab_region)
    
    c1, c2 = st.columns(2)
    with c1:
        # Revenu moyen par tranche d’âge
        st.subheader(":green[**Revenu moyen par tranche d’âge**]")
        st.divider()

        # Revenu moyen par tranche d’âge
        fig, ax = plt.subplots()
        data.groupby("age_grp")["rev_total_mois"].mean().plot(kind="bar", ax=ax, color="skyblue")
        ax.set_ylabel("Revenu moyen (FCFA)")

        st.pyplot(fig)

    with c2:
        
        st.subheader(":green[**📈 Répartition Taux bancarisation**]")
        st.divider()
        
        fig, ax = plt.subplots()
        data.groupby("age_grp")["bancarise"].mean().plot(kind="bar", ax=ax, color=["red", "orange","blue","green","grey", "purple", "skyblue"])
        ax.set_ylabel("Taux de bancarisation (%)")
        st.pyplot(fig) 
    
    # --------------------------------------------------------------------------------

    ca1, ca2 = st.columns(2)
    
    with ca1:
        
        st.subheader(":green[**💼 Assurance et emploi formel**]")
        st.divider()
        
        fig, ax = plt.subplots()
        data[["empl_formel","a_assurance"]].mean().plot(kind="bar", ax=ax, color=["purple","red"])
        ax.set_ylabel("Proportion (%)")
        st.pyplot(fig)

    with ca2: 
        # Statuts logement moyens nationaux
        st.subheader(":green[**🏠 Répartition statuts logement**]")
        st.divider()
        
        # Compter les effectifs par modalité de la variable 'logem'
        statuts_logement = data['logem'].value_counts(dropna=False)
        labels = statuts_logement.index  # ou map les int en labels si besoin
        
        colors = ["green","orange","blue","grey"]  # Personnalise ici
        
        fig, ax = plt.subplots(figsize=(7, 5))
        wedges, texts, autotexts = ax.pie(
            statuts_logement,
            labels=labels,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors[:len(statuts_logement)],
            textprops={'fontsize': 12}
        )
        ax.set_ylabel("")
        
        plt.tight_layout()
        st.pyplot(fig)


    # --------------------------------------------------------
    
# --------------------------- 3e page ---------------------

with tab3:

    # Exploration Score

    # Data Prediction
    def prediction(data):

        label_encoders = {}
        # Variables catégorielles à encoder
        cat_cols = ['mstat', 'age_grp', 'logem']
        
        for col in cat_cols:
            le = LabelEncoder()
            data_ml[col] = le.fit_transform(data_ml[col])
            label_encoders[col] = le
        print(data_ml)
        y_pred = model.predict(data_ml)
        print(f"y_prediction : {y_pred}")

        return y_pred
    
    # Load Model
    model = joblib.load('best_xgb.joblib')
    data_ml = data.drop(['region', 'sexe', 'branch', 'sectins', 'csp', 'age_num'], axis=1)   
    X_val = data_ml.tail(1)
    
    # Prédiction
    Profil_Score = prediction(X_val)
        
    print(f"Score de Solvabilité : {Profil_Score}")

    data_ml["Profil_Score"] = Profil_Score
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Score moyen")
        st.metric("", f"{round(data_ml["Profil_Score"].mean(), 1)}")
        print(f"New Dataset :\n{data_ml}")
    with col2:
        st.subheader("% Profils sécurisé")
        data_Secure = data_ml[data_ml["Profil_Score"] >= 76]
        secure = (data_Secure.shape[0] / data_ml.shape[0]) * 100
        print(f"Securité : {secure}")
        
        st.metric("", f"{secure:,.2f} %")
        
    with col3:
        st.subheader("% Profils très à risque")

        data_Risque = data_ml[data_ml["Profil_Score"] <= 50]
        risque = (data_Risque.shape[0] / data_ml.shape[0]) * 100
        print(f"Risque : {risque}")
        
        st.metric("", f"{risque:,.2f} %")

    c1, c2= st.columns(2)
    with c1:
        # Types de profil
        st.subheader(":green[**Types de Profil**]")
        st.divider()
        
        
    with c2:
        # Score Viz
        st.subheader(":green[**Score Viz**]")
        st.divider()
        














# =====================
# Tableau des données filtrées
# =====================


st.subheader("📋 Données filtrées")
st.dataframe(data, use_container_width=True)

st.markdown("✅ Ce tableau de bord peut être enrichi avec d’autres bases (NSIA, BHCI, RGPH2021) pour élargir la vision et construire un **proxy de scoring locatif**.")