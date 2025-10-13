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
        # Secteur institutionnel
        #st.subheader("Secteur institutionnel")
        #fig, ax = plt.subplots()
        #data.groupby('branch')["region"].count().plot.pie(
        #    autopct='%1.1f%%', ax=ax, colors=["green","orange","gray","blue","red","grey","purple","skyblue","yellow"])
        #ax.set_ylabel("")
        #st.pyplot(fig)
        counts = data.groupby('branch')["region"].count()
        fig, ax = plt.subplots(figsize=(20, 20))
        ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90, pctdistance=0.85)
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')  # cercle blanc au centre
        fig.gca().add_artist(centre_circle)
        ax.set_title("Secteur institutionnel")
        plt.tight_layout()
        st.pyplot(fig)
        
    with ca2:
        st.subheader("Secteur")

# =====================
# Tableau des données filtrées
# =====================


st.subheader("📋 Données filtrées")
st.dataframe(data, use_container_width=True)

st.markdown("✅ Ce tableau de bord peut être enrichi avec d’autres bases (NSIA, BHCI, RGPH2021) pour élargir la vision et construire un **proxy de scoring locatif**.")