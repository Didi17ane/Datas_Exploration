import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("AGG_EHCVM2021_V2.csv")

st.title("📊 Dashboard Scoring Locatif - Côte d’Ivoire")

# =====================
# SECTION 1 - Vue générale
# =====================
st.header("1️⃣ Vue générale de la population")

col1, col2, col3 = st.columns(3)
col1.metric("Population totale", int(df["nbr_indv"].sum()))
col2.metric("% Hommes", f"{round((df[df['sexe']=='Homme']['nbr_indv'].sum() / df['nbr_indv'].sum())*100,1)} %")
col3.metric("% Femmes", f"{round((df[df['sexe']=='Femme']['nbr_indv'].sum() / df['nbr_indv'].sum())*100,1)} %")

# Pyramide des âges
st.subheader("Pyramide des âges")
age_sex = df.groupby(["age_grp","sexe"])["nbr_indv"].sum().unstack().fillna(0)
age_sex.plot(kind="barh", stacked=True)
st.pyplot(plt)

# =====================
# SECTION 2 - Indicateurs socio-éco
# =====================
st.header("2️⃣ Indicateurs socio-économiques")

col1, col2 = st.columns(2)
col1.metric("Revenu moyen", f"{round(df['mean_rev'].mean(),0)} FCFA")
col2.metric("Taux de bancarisation", f"{round(df['mean_banked'].mean()*100,1)} %")

st.subheader("Distribution des revenus")
sns.boxplot(x="sexe", y="mean_rev", data=df)
st.pyplot(plt)

# =====================
# SECTION 3 - Logement
# =====================
st.header("3️⃣ Logement")

logement = df[["proprio_titre","proprio_sans","locataire","autre_logement"]].mean()*100
st.bar_chart(logement)

# =====================
# SECTION 4 - Synthèse
# =====================
st.header("4️⃣ Synthèse et proxy scoring")

# Proxy scoring simple
df["proxy_score"] = df["mean_rev"] * df["mean_banked"] * df["empl_formel"] * (df["proprio_titre"]+0.5*df["proprio_sans"])

top_regions = df.groupby("region")["proxy_score"].mean().sort_values(ascending=False)
st.subheader("Classement des régions (proxy scoring)")
st.table(top_regions)

# Corrélations
st.subheader("Corrélations entre variables clés")
corr = df[["mean_rev","mean_banked","empl_formel","proprio_titre","assurance"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
st.pyplot(plt)
