import pandas as pd
import json
from Regle_Segmentation import assign_cluster_from_rules

# Charger le dataset original avec clusters du modèle
df_model = pd.read_csv("../DATAS/ANSTAT2021_clusters_PC.csv")

# Charger les règles manuelles
with open("./Rules Clusters/cluster_rules_manual_example.json", "r", encoding="utf-8") as f:
    rules_loaded = json.load(f)

print(f"📋 Exemple de règle chargée:")
print(f"{json.dumps(list(rules_loaded.items())[0], indent=2, ensure_ascii=False)}\n")

# Convertir les clés en int si possible
formatted_rules = {}
for cluster_key, rule_dict in rules_loaded.items():
    try:
        cluster_id = int(float(cluster_key))
    except (ValueError, TypeError):
        cluster_id = str(cluster_key)
    
    # DEBUG : afficher la structure
    print(f"Cluster {cluster_id}:")
    print(f"  Type de rule_dict: {type(rule_dict)}")
    print(f"  Contenu: {rule_dict}")
    
    if isinstance(rule_dict, list):
        # C'est une liste de dictionnaires
        cond_dict = {}
        for rule in rule_dict:
            for var, vals in rule.items():
                cond_dict[var] = vals
    else:
        # C'est un dictionnaire simple
        cond_dict = rule_dict
    
    formatted_rules[cluster_id] = cond_dict

print(f"\n✅ Règles formatées pour {len(formatted_rules)} clusters")

# Appliquer les règles au dataset
df_rules = assign_cluster_from_rules(df_model.copy(), formatted_rules)

# Vérifier les colonnes
print(f"\n🔍 Colonnes dans df_rules: {df_rules.columns.tolist()}")

# Adapter selon les colonnes réelles
cluster_col = "cluster_assigned" if "cluster_assigned" in df_rules.columns else "cluster"
match_col = "match_score" if "match_score" in df_rules.columns else None

print(f"📌 Colonne cluster trouvée: {cluster_col}")
print(f"📌 Colonne match trouvée: {match_col}")

# Renommer les colonnes pour clarté
df_model_renamed = df_model.rename(columns={"cluster": "cluster_model"})
df_rules_renamed = df_rules.rename(columns={cluster_col: "cluster_rules"})

# Fusionner les résultats
cols_to_select = ["cluster_rules"]
if match_col:
    cols_to_select.append(match_col)

comparison = pd.concat([
    df_model_renamed[["cluster_model"]],
    df_rules_renamed[cols_to_select]
], axis=1)

# Ajouter colonne concordance
comparison["concordance"] = comparison["cluster_model"].astype(str) == comparison["cluster_rules"].astype(str)

# Identifier les cas problématiques
problem_cases = comparison[
    ((comparison["cluster_rules"].isna()) | (comparison["cluster_rules"] == "Aucun")) & 
    (comparison["cluster_model"].notna() & (comparison["cluster_model"] != "Aucun"))
]

print(f"\n📊 STATISTIQUES")
print(f"Total individus: {len(comparison)}")
print(f"Concordants: {comparison['concordance'].sum()} ({comparison['concordance'].sum()/len(comparison)*100:.1f}%)")
print(f"Non-concordants: {(~comparison['concordance']).sum()} ({(~comparison['concordance']).sum()/len(comparison)*100:.1f}%)")
print(f"Non-assignés: {(comparison['cluster_rules'] == 'Aucun').sum()}")
print(f"\n⚠️ Cas problématiques (cluster_rules vide vs cluster_model non vide): {len(problem_cases)}")

# Sauvegarder les résultats
comparison.to_csv("./Rules Clusters/comparison_rules_vs_model.csv", index=False)
problem_cases.to_csv("./Rules Clusters/problem_cases_empty_rules.csv", index=False)

print("\n✅ Fichiers sauvegardés:")
print("  - comparison_rules_vs_model.csv")
print("  - problem_cases_empty_rules.csv")

# Afficher quelques exemples des cas problématiques
print("\n🔍 Exemples de cas problématiques:")
if len(problem_cases) > 0:
    print(problem_cases.head(10))
else:
    print("Aucun problématique détecté")

# ===== DIAGNOSTIC =====
print("\n" + "="*60)
print("🔍 DIAGNOSTIC - Vérifier les non-assignés")
print("="*60)

non_assigned = comparison[comparison["cluster_rules"] == "Aucun"]
if len(non_assigned) > 0:
    print(f"\n⚠️ {len(non_assigned)} individus non assignés")
    
    # Récupérer les indices
    indices = non_assigned.index
    
    # Afficher quelques exemples du dataset original
    print("\nExemples d'individus non assignés:")
    print(df_model.iloc[indices[:5]])
    
    print("\nVérifier les clusters attendus:")
    print(non_assigned[["cluster_model", "cluster_rules", "match_score"]].head(10))