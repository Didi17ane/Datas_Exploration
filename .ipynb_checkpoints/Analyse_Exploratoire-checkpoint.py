#Importations des bibliothéques

import streamlit as st
import pandas as pd
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from PIL import Image

import plotly.express as px
import plotly.graph_objects as go

from typing import List, Tuple
from datetime import date, time, datetime, timedelta
from numerize.numerize import numerize
from streamlit_elements import elements, mui, html
import altair as alt

import requests

from streamlit_option_menu import option_menu
from streamlit_navigation_bar import st_navbar
import pages as pg

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


# --- Charger agrégats
df_ehcvm = pd.read_csv("AGG_EHCVM2021_V2.csv") 


# --- Pré-traitements

prem_ligne = df_ehcvm.head()
nbr_value = df_ehcvm.shape
desc = df_ehcvm.describe(include='all')
df_fin = df_ehcvm.dropna()
null_val = df_fin.isna().sum()
df_fin = df_fin.drop_duplicates()


# --- Affichages en console

print(f"\n***** Premiers lignes : \n {prem_ligne}")
print(f"\n***** Nombres de lignes et colonnes : \n {nbr_value}")
print(f"\n***** Description des variables : \n {desc}")
print(f"\n***** Valeurs nulles : \n {null_val}")
print(f"\n***** Dataset de traitement : \n {df_fin}")




# ------------------------------------------------------ AFFICHAGE DANS LA PAGE --------------------------------------------------

# ------------------------- Ajout de la Sidebar

with st.sidebar:
    # st.image("", width=200)
    st.title("SCORING Dashboard")
    
    st.divider()

    # Bases de données
    st.subheader("Base de données")
    database = st.multiselect(
        "Database",
        options=["AGG_EHCVM2021_V2.csv", ""],
        default=["AGG_EHCVM2021_V2.csv"]
    )
    
    # Filters
    st.subheader("Filters")
    listes = sorted(df_fin["region"].unique().tolist())
    options = ["Toutes"] + listes 
    print(listes)
    region = st.multiselect(
        "Région",
        options=options,
        default=["Toutes"]
    )
    # region = st.sidebar.selectbox("Région", ["Toutes"] + sorted(df_fin["region"].unique().tolist()))
    age_grp = st.sidebar.selectbox("Tranche d’âge", ["Toutes"] + sorted(df_fin["age_grp"].unique().tolist()))
    sexe = st.sidebar.selectbox("Sexe", ["Tous"] + sorted(df_fin["sexe"].unique().tolist()))
    mstat = st.sidebar.selectbox("Statut matrimonial", ["Tous"] + sorted(df_fin["mstat"].unique().tolist()))
    
    data = df_fin.copy()
    if "Toutes" not in region:
        data = data[data["region"].isin(region)]
    # if region != "Toutes":
      #  data = data[data["region"] == region]
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
    st.markdown("© 2025 TYLIMMO AFRICA")


if region:
        
    # ------------------------------ KPIs
    # KPIs
    st.subheader("Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    
    
    with col1:
        population = data["nbr_indv"].sum()
        st.metric(
            label="Total Population",
            value=f"{population} ",   
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
        femme = df_femme["nbr_indv"].sum()
        pourc_fem = round((femme / population)*100,1)
        st.metric(
            label="Total femmes",
            value=f"{femme}",
           
        )
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("% Femmes", f"{pourc_fem} %")
    
    with col3:
        df_homme = data[data["sexe"]=="Masculin"]
        homme = df_homme["nbr_indv"].sum()
        pour_hom = round((homme / population)*100,1)
        st.metric(
            label="Total hommes",
            value=f"{homme} ",
        )
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("% Hommes", f"{pour_hom} %")
        
    with col4:
        rev = round(data['mean_rev'].mean(), 0)
        if np.isnan(rev):
            rev=0    
        st.metric("Revenu moyen (FCFA)", f"{rev:,.0f}")
        
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        bank = round(data['mean_banked'].mean()*100, 1)
        if np.isnan(bank):
            bank=0    
        st.metric("Bancarisation", f"{bank} %")
        
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # -----------------------  NAVIGATIONS PAGES ------------------
    
    st.subheader("PERFORMANCES")
    tab1, tab2, tab3 = st.tabs(["Vue générale de la population", "KPIs locatifs & financiers", "Exploration interactive"])
    
    with tab1:
        c1, c2 = st.columns(2)
        st.subheader("Pyramide des âges")
        age_sex = data.groupby(["age_grp","sexe"])["nbr_indv"].sum().unstack().fillna(0)
        age_sex.plot(kind="barh", stacked=True)
        st.pyplot(plt)
    
    
    
    # ------------------------------------------------
    
    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader(":green[**📈 Revenu moyen par tranche d’âge**]")
            st.divider()
            
            fig, ax = plt.subplots()
            data.groupby("age_grp")["mean_rev"].mean().plot(kind="bar", ax=ax, color="skyblue")
            ax.set_ylabel("Revenu moyen (FCFA)")
            st.pyplot(fig)

        # ----------------------------------------------------------------------------------

            st.subheader(":green[**🏠 Répartition statuts logement**]")
            st.divider()
            
            fig, ax = plt.subplots()
            data[["proprio_titre","proprio_sans","locataire","autre_logement"]].mean().plot(
                kind="bar", ax=ax, color=["green","orange","blue","grey"]
            )
            ax.set_ylabel("Proportion (%)")
            st.pyplot(fig)
        with c2:
            st.subheader(":green[**🏠 Répartition statuts logement**]")
            st.divider()
            
            fig, ax = plt.subplots()
            data[["proprio_titre","proprio_sans","locataire","autre_logement"]].mean().plot(
                kind="bar", ax=ax, color=["green","orange","blue","grey"]
            )
            ax.set_ylabel("Proportion (%)")
            st.pyplot(fig)
        # --------------------------------------------------------------------------------
            st.subheader(":green[**💼 Assurance et emploi formel**]")
            st.divider()
            
            fig, ax = plt.subplots()
            data[["empl_formel","assurance"]].mean().plot(kind="bar", ax=ax, color=["purple","red"])
            ax.set_ylabel("Proportion (%)")
            st.pyplot(fig)
            
        # Exemple avec revenu moyen
        df_region = data.groupby("region")["mean_rev"].mean().reset_index()
        
        st.pydeck_chart(pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            initial_view_state=pdk.ViewState(latitude=7.54, longitude=-5.55, zoom=6),
            layers=[
                pdk.Layer(
                    "GeoJsonLayer",
                    data="geoBoundaries-CIV-ADM1.geojson",
                    get_fill_color="[255, (1-mean_rev/200000)*255, 0]",  # gradient selon revenu
                    pickable=True,
                ),
            ],
        ))
    with tab3:
        c1, c2 = st.columns(2)
    
    
    
    # =====================
    # Tableau des données filtrées
    # =====================

    
    st.subheader("📋 Données filtrées")
    st.dataframe(data, use_container_width=True)
    
    st.markdown("✅ Ce tableau de bord peut être enrichi avec d’autres bases (NSIA, BHCI, RGPH2021) pour élargir la vision et construire un **proxy de scoring locatif**.")   