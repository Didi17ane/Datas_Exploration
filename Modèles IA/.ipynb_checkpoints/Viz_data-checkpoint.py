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


# -------------------------- Authentificate
user = "admin"
passw = "admin@2025"

# Initialisation de la variable de session
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def _show_login_form():
    placeholder = st.empty()
    cols = st.columns([1, 2, 1])
    with cols[1]:
        with placeholder.form("login_form"):
            st.subheader("Login")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_submitted = st.form_submit_button("Login")

    return username, password, login_submitted, placeholder

#username, password, login_submitted, placeholder = _show_login_form()

# Vérifie la connexion
if not st.session_state.logged_in:
    username, password, login_submitted, placeholder = _show_login_form()

    if login_submitted:
        if username == user and password == passw:
            st.session_state.logged_in = True
            placeholder.empty()
            st.success("Connexion réussie !")
            st.rerun()()  # Recharge la page sans repasser par le formulaire
        else:
            st.error("Nom d'utilisateur ou mot de passe incorrect")

else:
    
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
    
        # Bouton de déconnexion
        if st.button("Se déconnecter"):
            st.session_state.logged_in = False
            st.rerun()()
    
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
        def prediction(dat_ml):
    
            label_encoders = {}
            # Variables catégorielles à encoder
            cat_cols = ['mstat', 'age_grp', 'logem']
            
            for col in cat_cols:
                le = LabelEncoder()
                dat_ml[col] = le.fit_transform(dat_ml[col])
                label_encoders[col] = le
            print(dat_ml)
            y_pred = model.predict(dat_ml)
            print(f"y_prediction : {y_pred}")
    
            return y_pred
        
        # Load Model
        model = joblib.load('GradientBoostingRegressor.pkl')
        data_ml = data.drop(['region', 'sexe', 'branch', 'sectins', 'csp', 'age_num'], axis=1)   
        X_val = data_ml
        print(f"X_val : {X_val}")
        
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
            def scoring():
                placeholder = st.empty()
                cols = st.columns([1, 3, 1])
                with cols[1]:
                    with placeholder.form("Score_form"):
                        st.subheader("Scorez-vous ici")
                        situation = st.selectbox("Votre Statut matrimonial", sorted(df_ehcvm["mstat"].unique().tolist()))
                        revenu = st.text_input("Votre Revenu Mensuel")
                        gr_age = st.selectbox("Votre Tranche d’âge", sorted(df_ehcvm["age_grp"].unique().tolist()))
                        emploi_form = st.selectbox("Votre Emploi Formel", ["Oui", "Non"])
                        bank = st.selectbox("Êtes-vous bancarisé?", ["Oui", "Non"])
                        assure = st.selectbox("Êtes-vous assuré?", ["Oui", "Non"])
                        statut_log = st.selectbox("Votre Statut de Logement", sorted(df_ehcvm["logem"].unique().tolist()))
                        score_submitted = st.form_submit_button(" Mon Score")
            
                return situation, revenu, gr_age, emploi_form, bank, assure, statut_log, score_submitted, placeholder

            situation, revenu, gr_age, emploi_form, bank, assure, statut_log, score_submitted, placeholder = scoring()

            if score_submitted:
                if emploi_form == "Oui":
                    emploi_form = 1
                else:
                    emploi_form = 0

                if bank == "Oui":
                    bank = 1
                else:
                    bank = 0

                if assure == "Oui":
                    assure = 1
                else:
                    assure = 0

                if not revenu:
                    revenu = 0
                
                data_scoring = pd.DataFrame([
                    {'mstat':situation, 'rev_total_mois':revenu, 'age_grp':gr_age, 'empl_formel':emploi_form, 'bancarise':bank, 'a_assurance':assure, 'logem':statut_log}
                ])
                print(f"data Scoring : {data_scoring}")
                
                
                Mon_score = prediction(data_scoring)
                print(f"Mon Score : {Mon_score}")
                st.success(f"Votre score est de : {Mon_score[0]:,.2f}")

                if (Mon_score >= 0) and (Mon_score < 20):
                    Mon_Profil = "Profil très vulnérable"
                elif (Mon_score >= 21) and (Mon_score < 50):
                    Mon_Profil = "Profil vulnérable"
                elif (Mon_score >= 51) and (Mon_score < 75):
                    Mon_Profil = " Profil intermédiaire"
                elif (Mon_score >= 76) and (Mon_score < 90):
                    Mon_Profil = " Profil sécurisé"
                elif (Mon_score > 90):
                    Mon_Profil = " Profil très sécurisé"
                    
                st.success(f"Vous faites partir de la catégorie : {Mon_Profil}")

    
    
    
    
    
    
    
    
    
    
    
    
    # =====================
    # Tableau des données filtrées
    # =====================
    
    
    st.subheader("📋 Données filtrées")
    st.dataframe(data, use_container_width=True)
    
    st.markdown("✅ Ce tableau de bord peut être enrichi avec d’autres bases (NSIA, BHCI, RGPH2021) pour élargir la vision et construire un **proxy de scoring locatif**.")