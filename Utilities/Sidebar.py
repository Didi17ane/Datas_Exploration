import streamlit as st

def sidebar(df_fin):
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
        # listes = sorted(df_fin["region"].unique().tolist())
        # options = ["Toutes"] + listes 
        # print(listes)
        # region = st.multiselect(
        #    "Région",
        #    options=options,
        #    default=["Toutes"]
        #)
        region = st.sidebar.selectbox("Région", ["Toutes"] + sorted(df_fin["region"].unique().tolist()))
        age_grp = st.sidebar.selectbox("Tranche d’âge", ["Toutes"] + sorted(df_fin["age_grp"].unique().tolist()))
        sexe = st.sidebar.selectbox("Sexe", ["Tous"] + sorted(df_fin["sexe"].unique().tolist()))
        mstat = st.sidebar.selectbox("Statut matrimonial", ["Tous"] + sorted(df_fin["mstat"].unique().tolist()))
        risque = st.selectbox("Risque bancaire", ['Tous'] + sorted(df_fin['categorie_risque_bancaire'].unique().tolist()))
        reco = st.selectbox("Recommandation crédit", ['Toutes'] + sorted(df_fin['recommandation_credit'].unique().tolist()))
        
        data = df_fin.copy()
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
        if risque != "Tous":
            data = data[data['categorie_risque_bancaire'] == risque]
        if reco != "Toutes":
            data = data[data['recommandation_credit'] == reco]

        
        # Advanced options
        st.subheader("Advanced Options")
        show_targets = st.checkbox("Show Targets", value=True)
        show_forecasts = st.checkbox("Show Forecasts", value=False)
        
        st.divider()
        st.markdown("© 2025 DIATA AFRICA SAS")

    return region,age_grp,sexe,mstat,risque,reco,data