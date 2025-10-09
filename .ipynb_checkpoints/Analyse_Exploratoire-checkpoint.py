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

from Utilities.Sidebar import *
from Utilities.globale import *
from Utilities.fintech import *
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
df_ehcvm = pd.read_csv("./DATAS/dataset_scoring_bancaire_fusion.csv") 


# --- Pré-traitements

prem_ligne = df_ehcvm.head()
nbr_value = df_ehcvm.shape
desc = df_ehcvm.describe(include='all')
df_fin = df_ehcvm.dropna()
null_val = df_fin.isna().sum()
df_fin = df_fin.drop_duplicates()


# --- Affichages en console

print(f"\n***** Premieres lignes : \n {prem_ligne}")
print(f"\n***** Nombres de lignes et colonnes : \n {nbr_value}")
print(f"\n***** Description des variables : \n {desc}")
print(f"\n***** Valeurs nulles : \n {null_val}")
print(f"\n***** Dataset de traitement : \n {df_fin}")




# ------------------------------------------------------ AFFICHAGE DANS LA PAGE --------------------------------------------------

# ------------------------- Ajout de la Sidebar

with st.sidebar:
    region,age_grp,sexe,mstat,risque,reco,data = sidebar(df_fin)

# ------------------------ PAGE PRINCIPALE

    
# ------------------------------ KPIs
# KPIs
st.subheader("Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)


with col1:
    population = data["nbr_indv"].sum()
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
    femme = df_femme["nbr_indv"].sum()
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
    homme = df_homme["nbr_indv"].sum()
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
st.markdown("")
st.markdown("Cette vision globale permet d'identifier les zones fortes/faibles, profils majoritaires, contrastes majeurs. Incontournable avant tout projet de scoring.")
tab1, tab2, tab3 = st.tabs(["Vue générale de la population", "KPIs locatifs & financiers", "Exploration interactive"])

tab_region = tabRegion(data)

# --------------------------- 1ere page ---------------------

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        # Pyramide des âges
        st.subheader(":green[**Pyramide des âges**]")
        pyramide(data)
    with c2:
        # Zoom région: répartition des effectifs et indicateurs
        st.subheader(":green[**Comparaison des régions - Population & Revenus**]")
        fig = repartition(tab_region)
        st.pyplot(fig)
            
    # Groupes d’âge : population et revenu médian
    st.subheader(":green[**Effectifs & revenu moyen par âge**]")
        
    age_grp = data.groupby("age_grp").agg(population=("nbr_indv","sum"), revenu_moy=("mean_rev","mean")).reset_index()
    st.dataframe(age_grp)

# --------------------------- 2e page ---------------------

with tab2:
    
    
   # st.subheader(":green[**📈 Effectifs & Revenu moyen par tranche d’âge**]")
    # Répartition par région et indicateurs clés
    st.subheader(":green[**📈 Répartition régionale**]")
    st.divider()
    
    st.dataframe(tab_region)
    
    c1, c2 = st.columns(2)
    with c1:
        # Revenu moyen par tranche d’âge
        st.subheader(":green[**Revenu moyen par tranche d’âge**]")
        st.divider()

        fig = revenu(data)
        st.pyplot(fig)

    with c2:
        
        st.subheader(":green[**📈 Répartition Taux bancarisation**]")
        st.divider()
        
        fig, ax = plt.subplots()
        data.groupby("age_grp")["mean_banked"].mean().plot(kind="bar", ax=ax, color=["red", "orange","blue","green","grey", "purple", "skyblue"])
        ax.set_ylabel("Taux de bancarisation (%)")
        st.pyplot(fig) 
    
    # --------------------------------------------------------------------------------

    ca1, ca2 = st.columns(2)
    
    with ca1:
        
        st.subheader(":green[**💼 Assurance et emploi formel**]")
        st.divider()
        
        fig, ax = plt.subplots()
        data[["empl_formel","assurance"]].mean().plot(kind="bar", ax=ax, color=["purple","red"])
        ax.set_ylabel("Proportion (%)")
        st.pyplot(fig)

    with ca2: 
        # Statuts logement moyens nationaux
        st.subheader(":green[**🏠 Répartition statuts logement**]")
        st.divider()
        
        fig, ax = plt.subplots()
        statut_cols = ["proprio_titre","proprio_sans","locataire","autre_logement"]
        data[statut_cols].mean().plot(
            kind="bar", ax=ax, color=["green","orange","blue","grey"]
        )
        ax.set_ylabel("Proportion (%)")
        st.pyplot(fig) 

    # --------------------------------------------------------
    
            

# --------------------------- 3e page ---------------------
    
with tab3:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Score moyen bancaire", round(data["score_bancaire_optimise"].mean(),1))
    with col2:
        st.metric("% Accords crédit (recommandé+conditionnel)", 
                  round(100*data[data["recommandation_credit"].str.contains("ACCORD")]["nbr_indv"].sum()/max(1,data["nbr_indv"].sum()),1))
    with col3:
        st.metric("% Profils très à risque", 
                  round(100*data[data["categorie_risque_bancaire"]=="Risque_Tres_Eleve"]["nbr_indv"].sum()/max(1,data["nbr_indv"].sum()),1))

    c1, c2 = st.columns(2)
    with c1:
        # Visualisation de la distribution du score bancaire 
        st.subheader("Distribution du score bancaire par profil")
        fig, ax = plt.subplots()
        data.groupby("categorie_risque_bancaire")["nbr_indv"].sum().plot(kind="bar", color="crimson", ax=ax)
        ax.set_ylabel("Population")
        ax.set_xlabel("Catégorie de risque")
        st.pyplot(fig)

    with c2:
        # Mix & analyse scoring : origine du risque
        st.subheader("Répartition des recommandations de crédit")
        fig, ax = plt.subplots()
        data.groupby('recommandation_credit')["nbr_indv"].sum().plot.pie(
            autopct='%1.1f%%', ax=ax, colors=["green","orange","gray","red"])
        ax.set_ylabel("")
        st.pyplot(fig)

    # ------------------------------------------
    
    # Exploration avancée : score par statut logement/profil emploi
    st.subheader("Scoring bancaire moyen par statut de logement")
    stats = data.groupby("statut_logement")["score_bancaire_optimise"].mean().sort_values()
    st.bar_chart(stats)



# =====================
# Tableau des données filtrées
# =====================


st.subheader("📋 Données filtrées")
st.dataframe(data, use_container_width=True)

st.markdown("✅ Ce tableau de bord peut être enrichi avec d’autres bases (NSIA, BHCI, RGPH2021) pour élargir la vision et construire un **proxy de scoring locatif**.")