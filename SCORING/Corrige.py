import pandas as pd
import numpy as np
import pickle
from sklearn.tree import _tree
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

# ============================================================================
# 1. CHARGEMENT ET ENCODAGE DES DONNÉES
# ============================================================================

df = pd.read_csv("../DATAS/ANSTAT2021_clusters_PC.csv")

# Sauvegarde des données originales pour référence
df_original = df.copy()

# Variables catégorielles à encoder
cat_vars = ['sex', 'marital_status', 'city', 'milieu_resid', 'region_name']

# Dictionnaires pour stocker les encoders et les mappings inversés
label_encoders = {}
inverse_mappings = {}  # Pour reconvertir code → label original

for col in cat_vars:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    
    # Créer le mapping inverse : {code_numérique: label_original}
    inverse_mappings[col] = dict(enumerate(le.classes_))
    
print("✓ Encodage terminé")
print(f"Mappings inversés disponibles pour: {list(inverse_mappings.keys())}")

# ============================================================================
# 2. PRÉPARATION DES DONNÉES POUR L'ENTRAÎNEMENT
# ============================================================================

y = df['cluster']
X = df.drop(columns=['cluster'])
features = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\n✓ Données préparées:")
print(f"  - Features: {features}")
print(f"  - Train: {len(X_train)}, Test: {len(X_test)}")

# ============================================================================
# 3. ENTRAÎNEMENT DU MODÈLE AVEC OPTIMISATION
# ============================================================================

param_grid = {
    "max_depth": [3, 4, 5, 6, 8, None],
    "min_samples_leaf": [1, 5, 10, 20, 30],
    "criterion": ["gini", "entropy"]
}

dt = DecisionTreeClassifier(random_state=42)

print("\n⏳ Optimisation des hyperparamètres en cours...")
grid = GridSearchCV(dt, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid.fit(X_train, y_train)

best_dt = grid.best_estimator_
print(f"✓ Meilleurs paramètres: {grid.best_params_}")

# Évaluation
y_pred = best_dt.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✓ Accuracy sur test: {accuracy:.4f}")

# ============================================================================
# 4. EXTRACTION DES RÈGLES LISIBLES (VALEURS ORIGINALES)
# ============================================================================

def decode_threshold_to_labels(feature, threshold, operator):
    """
    Convertit un threshold numérique en labels originaux
    """
    if feature not in inverse_mappings:
        # Pour age_num ou bancarise (pas de décodage nécessaire)
        if feature == "age_num":
            return f"{int(threshold)} ans"
        return f"{threshold:.2f}"
    
    mapping = inverse_mappings[feature]
    
    # Récupérer tous les codes possibles
    all_codes = sorted(mapping.keys())
    
    if operator == "<=":
        # Codes <= threshold
        selected_codes = [c for c in all_codes if c <= threshold]
    else:  # ">"
        # Codes > threshold
        selected_codes = [c for c in all_codes if c > threshold]
    
    # Convertir en labels originaux
    labels = [mapping[c] for c in selected_codes]
    
    return labels

def format_condition(feature, operator, threshold):
    """
    Formate une condition en langage naturel avec valeurs originales
    Optimise l'affichage en utilisant 'PAS DANS' quand c'est plus court
    """
    decoded = decode_threshold_to_labels(feature, threshold, operator)
    
    if isinstance(decoded, list):
        # Nettoyer les types numpy
        decoded_clean = [str(v).replace('np.int64(', '').replace(')', '') for v in decoded]
        
        if len(decoded_clean) == 0:
            return f"{feature} (aucune valeur)"
        elif len(decoded_clean) == 1:
            return f"{feature} = '{decoded_clean[0]}'"
        else:
            # Calculer si "PAS DANS" serait plus court
            if feature in inverse_mappings:
                all_values = list(inverse_mappings[feature].values())
                all_values_clean = [str(v).replace('np.int64(', '').replace(')', '') for v in all_values]
                excluded = [v for v in all_values_clean if v not in decoded_clean]
                
                # Si moins de valeurs exclues et que ça représente < 50% du total
                if len(excluded) > 0 and len(excluded) < len(decoded_clean) and len(excluded) <= 10:
                    values_str = "', '".join(excluded)
                    return f"{feature} PAS DANS ['{values_str}']"
            
            # Sinon, afficher toutes les valeurs incluses
            values_str = "', '".join(decoded_clean)
            return f"{feature} ∈ ['{values_str}']"
    else:
        return f"{feature} {operator} {decoded}"

def extract_rules_with_original_values(tree_model, feature_names):
    """
    Extrait toutes les règles de l'arbre avec les valeurs originales
    """
    tree = tree_model.tree_
    
    def recurse(node_id=0, current_conditions=None):
        if current_conditions is None:
            current_conditions = []
        
        rules = []
        
        # Feuille : on a trouvé une règle complète
        if tree.feature[node_id] == _tree.TREE_UNDEFINED:
            cluster = int(np.argmax(tree.value[node_id][0]))
            n_samples = int(tree.n_node_samples[node_id])
            rules.append({
                "conditions": current_conditions.copy(),
                "cluster": cluster,
                "n_samples": n_samples
            })
            return rules
        
        # Nœud interne : on continue la récursion
        feature = feature_names[tree.feature[node_id]]
        threshold = tree.threshold[node_id]
        
        # Branche gauche (<=)
        left_conditions = current_conditions + [(feature, "<=", threshold)]
        rules.extend(recurse(tree.children_left[node_id], left_conditions))
        
        # Branche droite (>)
        right_conditions = current_conditions + [(feature, ">", threshold)]
        rules.extend(recurse(tree.children_right[node_id], right_conditions))
        
        return rules
    
    return recurse()

# Extraction des règles
print("\n⏳ Extraction des règles avec valeurs originales...")
rules_raw = extract_rules_with_original_values(best_dt, features)

# Formatage lisible
rules_readable = []
for rule in rules_raw:
    conditions_text = []
    for feature, op, threshold in rule["conditions"]:
        cond_str = format_condition(feature, op, threshold)
        conditions_text.append(cond_str)
    
    rule_text = " ET ".join(conditions_text)
    
    rules_readable.append({
        "Règle_ID": len(rules_readable) + 1,
        "Conditions": rule_text,
        "Cluster_Assigné": rule["cluster"],
        "Nombre_Individus": rule["n_samples"]
    })

rules_df = pd.DataFrame(rules_readable)
print(f"✓ {len(rules_df)} règles extraites")

# ============================================================================
# 5. SAUVEGARDE DES RÈGLES ET DU MODÈLE
# ============================================================================

# Version 1: CSV avec toutes les valeurs (peut être long)
rules_df.to_csv("regles_segmentation_lisibles.csv", index=False, encoding="utf-8")
print("✓ Règles sauvegardées: regles_segmentation_lisibles.csv")

# Version 2: Format Excel avec colonnes séparées pour faciliter la lecture
rules_excel = []
for idx, rule in enumerate(rules_raw):
    row = {"Règle_ID": idx + 1, "Cluster_Assigné": rule["cluster"], "Nombre_Individus": rule["n_samples"]}
    
    for i, (feature, op, threshold) in enumerate(rule["conditions"]):
        decoded = decode_threshold_to_labels(feature, op, threshold)
        if isinstance(decoded, list):
            # Nettoyer numpy types
            decoded = [str(v).replace('np.int64(', '').replace(')', '') for v in decoded]
            row[f"Condition_{i+1}_Variable"] = feature
            row[f"Condition_{i+1}_Valeurs"] = ", ".join(decoded)
        else:
            row[f"Condition_{i+1}_Variable"] = feature
            row[f"Condition_{i+1}_Valeurs"] = f"{op} {decoded}"
    
    rules_excel.append(row)

rules_excel_df = pd.DataFrame(rules_excel)
rules_excel_df.to_excel("regles_segmentation_excel.xlsx", index=False)
print("✓ Règles format Excel: regles_segmentation_excel.xlsx")

# Sauvegarder le modèle et les encoders
with open("modele_clustering.pkl", "wb") as f:
    pickle.dump({
        "model": best_dt,
        "label_encoders": label_encoders,
        "inverse_mappings": inverse_mappings,
        "features": features
    }, f)
print("✓ Modèle sauvegardé: modele_clustering.pkl")

# Afficher un échantillon de règles
print("\n" + "="*80)
print("EXEMPLE DE RÈGLES EXTRAITES (5 premières):")
print("="*80)
print(rules_df.head(10).to_string(index=False))

# ============================================================================
# 6. FONCTION D'ASSIGNATION POUR NOUVELLES DONNÉES
# ============================================================================

def assigner_nouveaux_individus(csv_path, model_path="modele_clustering.pkl"):
    """
    Assigne des clusters à de nouveaux individus
    
    Args:
        csv_path: Chemin vers le CSV avec les nouvelles données
        model_path: Chemin vers le modèle sauvegardé
    
    Returns:
        DataFrame avec les clusters assignés
    """
    # Charger le modèle et les encoders
    with open(model_path, "rb") as f:
        saved_data = pickle.load(f)
    
    model = saved_data["model"]
    label_encoders = saved_data["label_encoders"]
    features = saved_data["features"]
    
    # Charger les nouvelles données
    new_data = pd.read_csv(csv_path)
    print(f"✓ {len(new_data)} nouveaux individus chargés")
    
    # Encoder les variables catégorielles avec les MÊMES encoders
    new_data_encoded = new_data.copy()
    for col in label_encoders.keys():
        if col in new_data_encoded.columns:
            le = label_encoders[col]
            # Gérer les valeurs inconnues
            new_data_encoded[col] = new_data_encoded[col].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
    
    # Prédiction
    X_new = new_data_encoded[features]
    clusters = model.predict(X_new)
    
    # Ajouter les résultats
    result = new_data.copy()
    result['Cluster_Assigné'] = clusters
    
    return result

# Exemple d'utilisation commenté:
# nouvelles_assignations = assigner_nouveaux_individus("nouvelles_donnees.csv")
# nouvelles_assignations.to_csv("resultats_assignation.csv", index=False)

print("\n" + "="*80)
print("✓ PROCESSUS TERMINÉ")
print("="*80)
print("\nFichiers générés:")
print("  1. regles_segmentation_lisibles.csv - Règles en langage naturel")
print("  2. modele_clustering.pkl - Modèle + encoders pour nouvelles données")
print("\nPour assigner de nouveaux individus:")
print("  nouvelles = assigner_nouveaux_individus('votre_fichier.csv')")