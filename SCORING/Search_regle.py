import pandas as pd
import numpy as np

#INPUT_C_100 = "../DATAS/ANSTAT2021_clusters_PC.csv"
#df_100 = pd.read_csv(INPUT_C_100)




##########################################
# 1. Génération des tranches d'âge
##########################################

def create_age_bins():
    bins = [0, 14, 24, 34, 44, 54, 64, 74, 84, 94, np.inf]
    labels = [
        "0-14","15-24","25-34","35-44","45-54",
        "55-64","65-74","75-84","85-94","95+"
    ]
    return bins, labels


##########################################
# 2. Fonction de profilage complet des clusters
##########################################

def generate_cluster_profiles(df_100, threshold=0.8):

    # --- Préparer les tranches d’âge ---
    bins, labels = create_age_bins()
    df_100["age_group"] = pd.cut(df_100["age_num"], bins=bins, labels=labels, right=True)

    # --- Détection variables catégorielles ---
    categorical_vars = df_100.select_dtypes(include="object").columns.tolist()
    forced_categorical = ["bancarise", "sex", "marital_status", "milieu_resid", "region_name", "city"]

    for col in forced_categorical:
        if col in df_100.columns and col not in categorical_vars:
            categorical_vars.append(col)

    categorical_vars += ["age_group"]
    
    # --------------------------------
    print(df_100)
    print(df_100.dtypes)
   
    print(categorical_vars)
    # --------------------------------
    categorical_vars = [c for c in categorical_vars if c not in ["cluster"]]

    numeric_vars = df_100.select_dtypes(include=["int64","float64"]).columns.tolist()
    numeric_vars = [n for n in numeric_vars if n not in ["cluster"]]

    profiles = {}
    summaries = {}
    rules = {}

    for c in df_100["cluster"].unique():

        cluster_data = df_100[df_100["cluster"] == c]
        cluster_size = len(cluster_data)

        # -----------------------------
        # STATISTIQUES NUMÉRIQUES
        # -----------------------------
        num_stats = cluster_data[numeric_vars].describe().T

        # -----------------------------
        # CATÉGORIELLES + SURREPRÉSENTATIONS
        # -----------------------------
        cat_results = {}

        for var in categorical_vars:
            freq = cluster_data[var].value_counts(normalize=True) * 100
            df_full_freq = df_100[var].value_counts(normalize=True) * 100

            merged = pd.DataFrame({
                "cluster_pct": freq,
                "global_pct": df_full_freq
            }).fillna(0)

            merged["surrepresented"] = merged["cluster_pct"] >= (threshold * 100)

            cat_results[var] = merged

        # -----------------------------
        # RÉSUMÉ AUTOMATIQUE
        # -----------------------------
        summary_parts = []

        for var, table in cat_results.items():
            dominant = table[table["cluster_pct"] >= table["cluster_pct"].max()]
            dominant_modalities = ", ".join(dominant.index.astype(str))

            summary_parts.append(f"{var}: {dominant_modalities}")

        summary_text = f"Cluster {c} → " + " | ".join(summary_parts)

        profiles[c] = {
            "numeric": num_stats,
            "categorical": cat_results
        }

        summaries[c] = summary_text

        # -----------------------------
        # RÈGLES DE SEGMENTATION
        # (conditions automatiques pour retrouver le cluster)
        # -----------------------------
        conditions = []

        for var, table in cat_results.items():
            dom = table[table["surrepresented"]].index.tolist()
            if len(dom) > 0:
                conditions.append({var: dom})

        rules[c] = conditions

    return profiles, summaries, rules


##########################################
# 3. Fonction pour retrouver un cluster dans un nouveau dataset
##########################################

def assign_cluster_from_rules(df_new, rules):
    df_new = df_new.copy()

    if "age_num" in df_new.columns:
        bins, labels = create_age_bins()
        df_new["age_group"] = pd.cut(df_new["age_num"], bins=bins, labels=labels, right=True)
    else:
        print("⚠️ Le dataset ne contient pas la colonne age_num → age_group ne peut pas être créée.")

    
    df_new["predicted_cluster"] = None

    for idx, row in df_new.iterrows():
        best_cluster = None
        best_match = 0  

        for cluster_id, cluster_rules in rules.items():
            score = 0

            for rule in cluster_rules:
                var = list(rule.keys())[0]
                values = rule[var]

                if row[var] in values:
                    score += 1

            if score > best_match:
                best_match = score
                best_cluster = cluster_id

        df_new.at[idx, "predicted_cluster"] = best_cluster

    return df_new


##########################################
# 4. Exemple d'utilisation
##########################################

if __name__ == "__main__":
    df = pd.read_csv("../DATAS/ANSTAT2021_clusters_PC.csv")

    profiles, summaries, rules = generate_cluster_profiles(df, threshold=0.8)

    # Sauvegarde
    pd.Series(summaries).to_csv("cluster_summaries.csv")
    
    pd.DataFrame(rules.items(), columns=["cluster","rules"]).to_csv("cluster_rules.csv", index=False)
    
    df_rules = pd.DataFrame([(cluster, content) for cluster, content in rules.items()],columns=["cluster", "content"])

    df_rules.to_json("cluster_rules.json", orient="records", indent=4)

    
    print("Profilage terminé.")
    print("Règles et résumés exportés.")