import joblib
import pandas as pd
import numpy as np
import os

def test():
    print("=== TEST DE SANTE DES MODELES ===")
    
    # Test Clustering
    print("\n1. Test du Modele de Clustering...")
    try:
        kmeans = joblib.load('models/kmeans_geopolitics.pkl')
        scaler = joblib.load('models/scaler_geopolitics.pkl')
        
        # Test avec un evenement neutre fictif
        features = ['GoldsteinScale', 'AvgTone', 'NumMentions', 'NumSources', 'NumArticles']
        dummy_event = pd.DataFrame([[0.0, 0.0, 10, 1, 10]], columns=features)
        
        cluster = kmeans.predict(scaler.transform(dummy_event))[0]
        print(f"[OK] Modele Clustering : Fonctionnel (Cluster predit : {cluster})")
    except Exception as e:
        print(f"[ERREUR] Modele Clustering : {e}")

    # Test Time Series
    print("\n2. Test du Modele de Prediction Temporelle...")
    try:
        forecaster = joblib.load('models/forecasting_stability.pkl')
        
        # Test avec des lags fictifs (8 features : 7 lags + 1 rolling mean)
        dummy_lags = np.random.rand(1, 8) 
        prediction = forecaster.predict(dummy_lags)[0]
        print(f"[OK] Modele Prediction Temporelle : Fonctionnel (Prevision : {prediction:.4f})")
    except Exception as e:
        print(f"[ERREUR] Modele Prediction Temporelle : {e}")

    print("\n=== FIN DES TESTS ===")

if __name__ == "__main__":
    test()
