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

from sklearn.svm import SVC
from streamlit_option_menu import option_menu
from streamlit_navigation_bar import st_navbar
import pages as pg



#### Page Configuration ####
st.set_page_config(
    page_title="Scoring Locataires Tylimmo",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

alt.themes.enable("dark")

st.title("📊 Tableau de Scoring des Locataires")



# --- Charger agrégats
df_ehcvm = pd.read_csv("AGG_EHCVM2021_V2.csv") 


# --- Pré-traitements

prem_ligne = df_ehcvm.head()
nbr_value = df_ehcvm.shape
desc = df_ehcvm.describe(include='all')
null_val = df_ehcvm.isna().sum()
df_ehcvm = df_ehcvm.drop_duplicates()
df_fin = df_ehcvm.dropna()


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
    
    # Date range selector
    st.subheader("Date Range")
    today = datetime.now().date()
    default_start = today - timedelta(days=30)
    start_date = st.date_input("Start Date", default_start)
    end_date = st.date_input("End Date", today)
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
    variable_filter = st.multiselect(
        "Variable Category",
        options=["region", "sexe", "mstat", "age_grp", "nbr_indv", "mean_rev", "mean_banked", "empl_formel", "proprio_titre", "proprio_sans", "locataire", "autre_logement", "assurance"],
        default=["region", "sexe", "mstat", "age_grp"]
    )
    
    region_filter = st.selectbox(
        "Region",
        options=["Abidjan", "Yamoussoukro", "Jacqueville", "Divo"]
    )
    
    # Advanced options
    st.subheader("Advanced Options")
    show_targets = st.checkbox("Show Targets", value=True)
    show_forecasts = st.checkbox("Show Forecasts", value=False)
    
    st.divider()
    st.markdown("© 2025 TYLIMMO AFRICA")

# ------------------------------ KPIs
# KPIs
st.subheader("Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    population = df_fin["nbr_indv"].sum()
    st.metric(
        label="Total Population",
        value=f"{population} ",   
    )
with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    df_femme = df_fin[df_fin["sexe"]=="Féminin"]
    #print(df_femme)
    femme = df_femme["nbr_indv"].sum()
    st.metric(
        label="Total femmes",
        value=f"{femme} ",
    )

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    df_homme = df_fin[df_fin["sexe"]=="Masculin"]
    #print(df_femme)
    homme = df_homme["nbr_indv"].sum()
    st.metric(
        label="Total hommes",
        value=f"{homme} ",
    )
with col4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    population = df_fin["nbr_indv"].sum()
    st.metric(
        label="",
        value=f"{population} ",
    )

st.markdown('</div>', unsafe_allow_html=True)


   