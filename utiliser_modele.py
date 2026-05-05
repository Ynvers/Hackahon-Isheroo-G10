import pandas as pd
import joblib
import json
import os

def main():
    print("--- LECTURE DES MODÈLES .PKL ---")
    
    # 1. Charger les modèles binaires (.pkl)
    # Les fichiers .pkl ne sont pas lisibles dans un éditeur de texte, 
    # ils doivent être chargés par Python via joblib.
    try:
        kmeans = joblib.load('models/kmeans_geopolitics.pkl')
        scaler = joblib.load('models/scaler_geopolitics.pkl')
        print("[OK] Modèles chargés avec succès.")
    except FileNotFoundError:
        print("[ERREUR] Les fichiers .pkl sont introuvables dans le dossier 'models/'.")
        return

    # 2. Rendre les règles du modèle "lisibles" pour les humains
    # Nous allons extraire les centroïdes (les centres de chaque groupe)
    # et les déséchelonner (inverse transform) pour retrouver les vraies valeurs.
    centroids_scaled = kmeans.cluster_centers_
    centroids_real = scaler.inverse_transform(centroids_scaled)
    
    features = ['GoldsteinScale', 'AvgTone', 'NumMentions', 'NumSources', 'NumArticles']
    
    # Création d'un dictionnaire lisible
    model_info = {}
    for i, centroid in enumerate(centroids_real):
        # Application de notre logique métier pour nommer le cluster
        goldstein, tone, mentions, sources, articles = centroid
        
        # Logique de nommage (la même que lors de l'entraînement)
        if goldstein < -2 and tone < -2:
            name = "Crise/Conflit Négatif"
        elif goldstein > 2 and tone > 2:
            name = "Coopération/Diplomatie Positive"
        elif mentions > np_median(centroids_real[:, 2]) * 1.5:  # Approximation pour la médiatisation
            name = "Événement Fortement Médiatisé"
        elif goldstein < 0:
            name = "Tension Modérée"
        else:
            name = "Événement Neutre/Routinier"
            
        model_info[f"Cluster_{i}"] = {
            "Nom_Contexte": name,
            "Valeurs_Moyennes": {
                "GoldsteinScale": round(goldstein, 2),
                "AvgTone": round(tone, 2),
                "NumMentions": round(mentions, 2),
                "NumSources": round(sources, 2),
                "NumArticles": round(articles, 2)
            }
        }
        
    # Sauvegarde des règles dans un fichier lisible (JSON)
    with open('models/regles_du_modele.json', 'w', encoding='utf-8') as f:
        json.dump(model_info, f, indent=4, ensure_ascii=False)
    print("[OK] Les règles du modèle ont été extraites en format lisible dans 'models/regles_du_modele.json'.")


    # 3. Exemple : Comment exécuter le modèle sur de nouvelles données !
    print("\n--- EXÉCUTION DU MODÈLE SUR DE NOUVELLES DONNÉES ---")
    
    # Imaginons un nouvel événement géopolitique fictif
    nouvel_evenement = pd.DataFrame([{
        'GoldsteinScale': -5.0,  # Très instable
        'AvgTone': -8.5,         # Très négatif
        'NumMentions': 150,      # Très mentionné
        'NumSources': 20,
        'NumArticles': 140
    }])
    
    print("Nouvel événement à analyser :")
    print(nouvel_evenement.to_string(index=False))
    
    # Étape A: Mettre à l'échelle avec le Scaler
    donnees_scaled = scaler.transform(nouvel_evenement)
    
    # Étape B: Prédire le numéro du cluster avec le KMeans
    prediction_cluster = kmeans.predict(donnees_scaled)[0]
    
    # Étape C: Récupérer le nom humain depuis nos infos extraites
    contexte_predit = model_info[f"Cluster_{prediction_cluster}"]["Nom_Contexte"]
    
    print(f"\n--> RÉSULTAT : Le modèle a identifié cet événement comme : **{contexte_predit}** (Cluster {prediction_cluster})")

def np_median(array):
    # Petite fonction utilitaire pour éviter d'importer numpy juste pour ça
    sorted_array = sorted(array)
    n = len(array)
    if n % 2 == 1:
        return sorted_array[n // 2]
    else:
        return (sorted_array[n // 2 - 1] + sorted_array[n // 2]) / 2

if __name__ == "__main__":
    main()
