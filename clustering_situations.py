import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

def main():
    print("Chargement des données...")
    # Path to the data
    data_path = "data/events_clean.csv"
    
    if not os.path.exists(data_path):
        print(f"Erreur : Le fichier {data_path} n'existe pas.")
        return

    # Load data
    df = pd.read_csv(data_path)
    
    # Selection of features relevant for geopolitical context clustering
    # GoldsteinScale: Potentiel d'impact sur la stabilité du pays
    # AvgTone: Tonalité moyenne des articles (sentiment)
    # NumMentions, NumSources, NumArticles: Niveau de médiatisation
    features = ['GoldsteinScale', 'AvgTone', 'NumMentions', 'NumSources', 'NumArticles']
    
    # Check if features exist in the dataframe
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        print(f"Erreur : Colonnes manquantes dans les données : {missing_features}")
        return

    print("Préparation des données...")
    # Fill missing values with median
    X = df[features].copy()
    X.fillna(X.median(), inplace=True)
    
    # Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("Entraînement du modèle de clustering (K-Means)...")
    # Determine the number of clusters (let's use 4 as a reasonable default for context types)
    k = 4
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Calculate Silhouette Score on a sample if data is large
    sample_size = min(10000, len(X_scaled))
    indices = np.random.choice(len(X_scaled), sample_size, replace=False)
    sil_score = silhouette_score(X_scaled[indices], df['Cluster'].iloc[indices])
    print(f"Silhouette Score (sur un échantillon de {sample_size}): {sil_score:.4f}")
    
    print("Analyse des clusters...")
    # Add clusters to original dataframe
    
    # Profiling the clusters
    cluster_profiles = df.groupby('Cluster')[features].mean()
    print("\nProfils des clusters (moyennes des caractéristiques) :")
    print(cluster_profiles)
    
    # Naming the clusters based on logic (Heuristics based on GoldsteinScale and AvgTone)
    cluster_names = {}
    for i in range(k):
        goldstein = cluster_profiles.loc[i, 'GoldsteinScale']
        tone = cluster_profiles.loc[i, 'AvgTone']
        mentions = cluster_profiles.loc[i, 'NumMentions']
        
        if goldstein < -2 and tone < -2:
            name = "Crise/Conflit Négatif"
        elif goldstein > 2 and tone > 2:
            name = "Coopération/Diplomatie Positive"
        elif mentions > cluster_profiles['NumMentions'].median() * 1.5:
            name = "Événement Fortement Médiatisé"
        elif goldstein < 0:
            name = "Tension Modérée"
        elif goldstein >= 0:
            name = "Événement Neutre/Routinier"
        else:
            name = f"Contexte Type {i}"
        
        cluster_names[i] = name
        
    df['Context_Type'] = df['Cluster'].map(cluster_names)
    
    print("\nRépartition des contextes géopolitiques :")
    print(df['Context_Type'].value_counts())

    print("Réduction de dimensionnalité pour la visualisation (PCA)...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df['PCA1'] = X_pca[:, 0]
    df['PCA2'] = X_pca[:, 1]
    
    print("Sauvegarde des résultats et des modèles...")
    # Save the models
    os.makedirs('models', exist_ok=True)
    joblib.dump(kmeans, 'models/kmeans_geopolitics.pkl')
    joblib.dump(scaler, 'models/scaler_geopolitics.pkl')
    joblib.dump(pca, 'models/pca_geopolitics.pkl')
    
    # Save the clustered data
    output_path = "data/events_clustered.csv"
    df.to_csv(output_path, index=False)
    print(f"Modèles sauvegardés dans le dossier 'models/'.")
    print(f"Données avec clusters sauvegardées dans {output_path}.")
    print("Terminé avec succès !")

if __name__ == "__main__":
    main()
