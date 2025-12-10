import pandas as pd
import json

# Charger les données et règles
df = pd.read_csv("../DATAS/ANSTAT2021_clusters_PC.csv")

with open("./Rules Clusters/cluster_rules_manual_example.json", "r", encoding="utf-8") as f:
    rules_loaded = json.load(f)

# Sélectionner un cluster à tester
test_cluster = list(rules_loaded.keys())[0]
print(f"🧪 Test sur le cluster: {test_cluster}")

# Charger sa règle
rule = rules_loaded[test_cluster]
print(f"\nRègle chargée:")
print(json.dumps(rule, indent=2, ensure_ascii=False))

# Récupérer les individus de ce cluster du dataset original
df_cluster = df[df["cluster"].astype(str) == str(test_cluster)]
print(f"\n📊 {len(df_cluster)} individus dans le cluster {test_cluster}")

# Vérifier manuellement si la règle s'applique
print(f"\n✅ Vérification manuelle des conditions:")

if isinstance(rule, list):
    conditions = {}
    for rule_item in rule:
        for var, vals in rule_item.items():
            conditions[var] = vals
else:
    conditions = rule

for var, expected_vals in conditions.items():
    if var in df_cluster.columns:
        actual_vals = df_cluster[var].unique()
        print(f"\n  {var}:")
        print(f"    Attendu: {expected_vals}")
        print(f"    Réel: {list(actual_vals)}")
        print(f"    Correspond: {'✅' if set(actual_vals).issubset(set(expected_vals)) else '❌'}")
    else:
        print(f"\n  ❌ {var} - COLONNE NON TROUVÉE")

print(f"\n💡 Si certaines conditions ne correspondent pas:")
print(f"   - Vérifier la casse (Masculin vs masculin)")
print(f"   - Vérifier les espaces")
print(f"   - Vérifier les valeurs manquantes (NaN)")