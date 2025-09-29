import streamlit as st
import matplotlib.pyplot as plt

def tabRegion(data):

    # Répartition par région et indicateurs clés
    tab_region = data.groupby("region").agg(population=("nbr_indv","sum"),
                                            revenu_moy=("mean_rev","mean"),
                                            bank_moy=("mean_banked","mean")).reset_index()
    return tab_region

def pyramide(data):
    
    # Pyramide des âges
    age_sex = data.groupby(["age_grp","sexe"])["nbr_indv"].sum().unstack().fillna(0)
    age_sex.plot(kind="barh", stacked=True)
    st.pyplot(plt)

def repartition(tab_region):
    
    # Zoom région: répartition des effectifs et indicateurs
    fig, ax = plt.subplots()
    tab_region.set_index("region")[["population", "revenu_moy"]].plot(kind="bar", ax=ax, secondary_y="rev_moy")

    return fig