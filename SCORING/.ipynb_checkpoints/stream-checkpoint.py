import streamlit as st
import pandas as pd
import numpy as np
import json
from Regle import generate_cluster_profiles, assign_cluster_from_rules

# Variables utilisées pour la segmentation
SEGMENTATION_VARS = [
    "bancarise", "sex", "marital_status",
    "milieu_resid", "region_name", "city", "age_group"
]

def safe_sort_clusters(cluster_list):
    """Trie les clusters de manière robuste (gère int et str)"""
    try:
        # Essayer de convertir tous en int
        return sorted(cluster_list, key=lambda x: int(x) if str(x).isdigit() else float('inf'))
    except (ValueError, TypeError):
        # Sinon, trier comme strings
        return sorted(cluster_list, key=str)

st.set_page_config(page_title="Analyse des Clusters", layout="wide")
st.title("🧬 Plateforme d'Analyse et Segmentation par Clusters")

# ==========================================================
# SECTION 1 : UPLOAD ET GÉNÉRATION DES PROFILS
# ==========================================================
st.header("1️⃣ Importer le dataset avec clusters")

uploaded = st.file_uploader("📌 Choisir un fichier CSV contenant une colonne 'cluster'", type="csv")

if uploaded:
    df = pd.read_csv(uploaded)
    
    st.subheader("📄 Aperçu du dataset chargé")
    st.dataframe(df.head(10))
    
    st.info(f"**Dimensions :** {df.shape[0]} lignes × {df.shape[1]} colonnes")
    
    # Vérifier la présence de la colonne cluster
    if "cluster" not in df.columns:
        st.error("❌ Le dataset doit contenir une colonne 'cluster'")
        st.stop()
    
    st.success(f"✅ {df['cluster'].nunique()} clusters détectés : {sorted(df['cluster'].unique())}")

    # ==========================================================
    # GÉNÉRATION DES PROFILS
    # ==========================================================
    st.header("2️⃣ Génération des profils et règles")
    
    threshold = st.slider(
        "Seuil de définition des modalités (%)", 
        min_value=50, 
        max_value=95, 
        value=80, 
        step=5,
        help="Pourcentage cumulé pour identifier les modalités clés d'un cluster"
    ) / 100
    
    if st.button("🔍 Générer les profils et règles de segmentation"):
        with st.spinner("Génération en cours..."):
            profiles, summaries, rules, df_modalities = generate_cluster_profiles(df, threshold=threshold)
        
        st.success("🎉 Profils et règles générés avec succès !")
        
        # Sauvegarder les règles dans la session
        st.session_state["profiles"] = profiles
        st.session_state["summaries"] = summaries
        st.session_state["rules"] = rules
        st.session_state["df_modalities"] = df_modalities
        
        # Sauvegarder en JSON avec types cohérents
        rules_jsonable = {}
        for k, v in rules.items():
            cluster_key = int(k) if isinstance(k, (int, float, np.integer)) else str(k)
            rules_jsonable[cluster_key] = {
                str(var): [str(val) for val in vals] 
                for var, vals in v.items()
            }
        
        with open("cluster_rules.json", "w", encoding="utf-8") as f:
            json.dump(rules_jsonable, f, ensure_ascii=False, indent=4)
        
        st.info("💾 Règles sauvegardées dans `cluster_rules.json`")

    # ==========================================================
    # EXPLORATION DES CLUSTERS
    # ==========================================================
    if "profiles" in st.session_state:
        st.header("3️⃣ Explorer les clusters")
        
        profiles = st.session_state["profiles"]
        summaries = st.session_state["summaries"]
        rules = st.session_state["rules"]
        
        # Trier les clusters de manière robuste
        cluster_ids = safe_sort_clusters(list(profiles.keys()))
        
        selected = st.selectbox("📊 Sélectionner un cluster à explorer", cluster_ids)
        
        # Affichage du résumé
        st.subheader(f"📝 Résumé du Cluster {selected}")
        st.info(summaries[selected])
        
        # Colonnes pour organisation
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Variables numériques")
            if not profiles[selected]["numeric"].empty:
                st.dataframe(
                    profiles[selected]["numeric"][["mean", "50%", "std", "min", "max"]]
                    .rename(columns={"50%": "median"}),
                    use_container_width=True
                )
            else:
                st.write("Aucune variable numérique")
        
        with col2:
            st.subheader("🧩 Règles de segmentation")
            st.json(rules[selected])
        
        # Variables catégorielles détaillées
        st.subheader("📁 Variables catégorielles (utilisées pour la segmentation)")
        
        for var in SEGMENTATION_VARS:
            if var in profiles[selected]["categorical"]:
                with st.expander(f"🔹 {var}"):
                    tab = profiles[selected]["categorical"][var]
                    
                    # Afficher uniquement les modalités définissantes en priorité
                    defining = tab[tab["is_defining"]]
                    other = tab[~tab["is_defining"]]
                    
                    st.write("**Modalités définissantes :**")
                    st.dataframe(
                        defining[["cluster_pct", "global_pct", "cum_pct"]].round(2),
                        use_container_width=True
                    )
                    
                    if not other.empty:
                        st.write("**Autres modalités :**")
                        st.dataframe(
                            other[["cluster_pct", "global_pct"]].round(2),
                            use_container_width=True
                        )

# ==========================================================
# SECTION 2 : SEGMENTATION D'UN NOUVEAU DATASET
# ==========================================================
st.header("4️⃣ Appliquer la segmentation à un nouveau dataset")

uploaded2 = st.file_uploader(
    "📌 Importer un dataset à segmenter (doit contenir les mêmes variables)", 
    type="csv",
    key="upload_new"
)

if uploaded2:
    df_new = pd.read_csv(uploaded2)
    
    st.subheader("📄 Aperçu du nouveau dataset")
    st.dataframe(df_new.head(10))
    
    st.info(f"**Dimensions :** {df_new.shape[0]} lignes × {df_new.shape[1]} colonnes")
    
    # Charger les règles
    try:
        with open("cluster_rules.json", "r", encoding="utf-8") as f:
            rules_loaded = json.load(f)
        
        # Reconvertir au format attendu (forcer tous les clusters en int si possible)
        formatted_rules = {}
        for cluster_key, rule_dict in rules_loaded.items():
            # Essayer de convertir en int, sinon garder comme string
            try:
                cluster_id = int(float(cluster_key))
            except (ValueError, TypeError):
                cluster_id = str(cluster_key)
            formatted_rules[cluster_id] = {var: vals for var, vals in rule_dict.items()}
        
        st.info(f"✅ Règles chargées pour {len(formatted_rules)} clusters")
        
        if st.button("🔮 Prédire les clusters"):
            with st.spinner("Segmentation en cours..."):
                result = assign_cluster_from_rules(df_new, formatted_rules)
            
            st.success("✅ Segmentation terminée !")
            
            # Statistiques de segmentation
            st.subheader("📊 Résultats de la segmentation")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total d'individus", len(result))
                st.metric("Individus non assignés", (result["cluster_assigned"] == "Aucun").sum())
            
            with col2:
                assigned = result[result["cluster_assigned"] != "Aucun"]
                if len(assigned) > 0:
                    st.metric("Taux d'assignation", f"{len(assigned)/len(result)*100:.1f}%")
                    st.metric("Score moyen de correspondance", f"{assigned['match_score'].mean():.2f}")
            
            # Distribution par cluster
            st.subheader("📈 Distribution par cluster")
            dist = result["cluster_assigned"].value_counts()
            
            # Réorganiser pour avoir un ordre logique
            dist_sorted = dist.reindex(safe_sort_clusters(dist.index.tolist()), fill_value=0)
            
            st.bar_chart(dist_sorted)
            
            # Affichage du résultat
            st.subheader("🗂️ Dataset segmenté")
            st.dataframe(result)
            
            # Téléchargement
            csv = result.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Télécharger les résultats (CSV)",
                data=csv,
                file_name="segmentation_results.csv",
                mime="text/csv"
            )
    
    except FileNotFoundError:
        st.error("⚠️ Aucune règle trouvée. Veuillez d'abord générer les profils dans la section précédente.")
    except Exception as e:
        st.error(f"❌ Erreur lors de la segmentation : {str(e)}")