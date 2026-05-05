import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os
import matplotlib.pyplot as plt

def main():
    print("--- MODÈLE DE PRÉDICTION TEMPORELLE (STABILITÉ) ---")
    
    # 1. Chargement des données
    data_path = "data/daily_score.csv"
    if not os.path.exists(data_path):
        print(f"[ERREUR] Le fichier {data_path} n'existe pas.")
        return
        
    df = pd.read_csv(data_path)
    df['SQLDATE'] = pd.to_datetime(df['SQLDATE'])
    df = df.sort_values('SQLDATE')
    
    # 2. Ingénierie des caractéristiques (Features Engineering)
    # On va prédire la 'stability_score' (basée sur Goldstein)
    target = 'stability_score'
    
    # Création de lags (valeurs passées)
    for i in range(1, 8): # Lags de 1 à 7 jours
        df[f'lag_{i}'] = df[target].shift(i)
        
    # Moyenne mobile sur 7 jours
    df['rolling_mean_7'] = df[target].shift(1).rolling(window=7).mean()
    
    # On supprime les lignes avec des NaN créés par les décalages
    df_ml = df.dropna().copy()
    
    # 3. Préparation des jeux d'entraînement et de test
    # On ne mélange pas les données temporelles ! On coupe à la fin.
    features = [f'lag_{i}' for i in range(1, 8)] + ['rolling_mean_7']
    
    # On prend les derniers 20% pour le test
    split_idx = int(len(df_ml) * 0.8)
    train = df_ml.iloc[:split_idx]
    test = df_ml.iloc[split_idx:]
    
    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]
    
    # 4. Entraînement du modèle (Random Forest Regressor)
    print("Entraînement du modèle Random Forest...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 5. Évaluation
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    print(f"Évaluation du modèle :")
    print(f"- MAE (Erreur Absolue Moyenne) : {mae:.4f}")
    print(f"- RMSE (Erreur Quadratique Moyenne) : {rmse:.4f}")
    
    # 6. Prédiction pour les 7 prochains jours (Forecasting)
    print("\nPrédiction pour les 7 prochains jours...")
    last_data = df_ml.iloc[-1:].copy()
    forecasts = []
    current_features = last_data[features].values
    
    # On fait une boucle pour prédire jour après jour (autoregressif)
    for i in range(7):
        pred = model.predict(current_features)[0]
        forecasts.append(pred)
        
        # Mise à jour des features pour le jour suivant :
        # On décale les lags et on insère la nouvelle prédiction
        new_row = np.zeros(len(features))
        new_row[0] = pred # lag_1 devient la prédiction
        new_row[1:7] = current_features[0][0:6] # on décale les anciens lags
        new_row[7] = np.mean(np.append(current_features[0][0:6], pred)) # approximation de la rolling mean
        current_features = new_row.reshape(1, -1)

    # 7. Sauvegarde des résultats
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/forecasting_stability.pkl')
    
    # Création d'un DataFrame pour le forecast
    last_date = df['SQLDATE'].max()
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7)
    df_forecast = pd.DataFrame({
        'Date': forecast_dates,
        'Predicted_Stability_Score': forecasts
    })
    
    df_forecast.to_csv('data/stability_forecast.csv', index=False)
    print(f"[OK] Modèle sauvegardé dans 'models/forecasting_stability.pkl'")
    print(f"[OK] Prévisions sauvegardées dans 'data/stability_forecast.csv'")
    
    print("\nAperçu des prévisions :")
    print(df_forecast)

if __name__ == "__main__":
    main()
