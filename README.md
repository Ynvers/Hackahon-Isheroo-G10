# Bénin Geo-Watch — Radar Géopolitique et Sécuritaire

**Hackathon iSHEERO × DataCamp Donates 2026 — Bénin Insights Challenge | Équipe 10**

Bénin Geo-Watch est un système d'analyse et d'alerte précoce conçu pour surveiller la stabilité géopolitique du Bénin à partir des données de la base mondiale **GDELT** (Global Database of Events, Language and Tone).

---

## Problématique

> *Comment transformer le bruit médiatique mondial en signaux d'alerte précoces pour anticiper les tensions et mesurer la stabilité du Bénin ?*

Face à la complexité des enjeux sécuritaires (notamment au Nord-Bénin) et diplomatiques, le projet construit un pipeline de données capable d'analyser en continu les événements couverts par la presse internationale. Au-delà de la visualisation, trois couches d'intelligence artificielle permettent de classifier, prédire et interroger l'état géopolitique du pays.

---

## Dashboard

Le dashboard interactif est accessible ici :

**[Bénin Geo-Watch — Dashboard en ligne](https://isheeroo.streamlit.app/)**

Il propose :

- **4 KPIs en temps réel** : total d'événements, Goldstein moyen, Tone médiatique, score de stabilité
- **Carte des tensions régionales** : géolocalisation des événements par département avec code couleur (rouge = négatif, vert = coopération)
- **4 graphiques analytiques** : volume médiatique, intensité des conflits, distribution par type, évolution de la stabilité
- **Insights clés** : 5 observations majeures issues de l'analyse des données 2025/2026
- **Simulateur What-If** : classification en temps réel d'un scénario personnalisé via le modèle KMeans
- **Assistant RAG** : chatbot dans la sidebar, alimenté par Mistral AI et une base vectorielle FAISS de 6 353 articles

---

## Architecture du projet

```
Hackahon-Isheroo-G10/
│
├── dashboard/
│   ├── app.py                     # Application Streamlit principale
│   └── pyproject.toml             # Dépendances (uv)
│
├── data/
│   ├── events_clean.csv           # Données GDELT nettoyées (événements)
│   ├── daily_score.csv            # Score de stabilité quotidien
│   ├── stability_forecast.csv     # Prévisions J+1 à J+7
│   └── VectorBase_db/
│       ├── benin_events.index     # Index FAISS (6 353 vecteurs)
│       └── events_metadata.pkl    # Métadonnées associées
│
├── models/
│   ├── kmeans_geopolitics.pkl     # Modèle de clustering (4 clusters)
│   ├── scaler_geopolitics.pkl     # StandardScaler associé
│   ├── pca_geopolitics.pkl        # PCA pour visualisation
│   ├── forecasting_stability.pkl  # RandomForest de prédiction temporelle
│   └── regles_du_modele.json      # Labels humains des clusters
│
├── notesbooks/
│   ├── Data_Engineering_Hackathon_Isheero.ipynb
│   ├── ML.ipynb
│   └── ML_Engineering_Hackathon__.ipynb
│
├── clustering_situations.py       # Entraînement du modèle de clustering
├── prediction_temporelle.py       # Entraînement + forecasting
├── utiliser_modele.py             # Démo d'utilisation du clustering
├── tester_tout.py                 # Test de santé des modèles
├── .example.env                   # Template de configuration
└── README.md
```

---

## Modèles de Machine Learning

### 1. Clustering géopolitique (KMeans, non-supervisé)

Classifie automatiquement chaque événement en 4 catégories :

| Cluster | Nom | Goldstein moyen | Tone moyen |
|---------|-----|----------------|------------|
| 0 | Crise / Conflit Négatif | -3.22 | -5.54 |
| 1 | Événement Fortement Médiatisé | +1.35 | -0.19 |
| 2 | Événement Neutre / Routinier | +2.78 | +1.31 |
| 3 | Événement Fortement Médiatisé | +0.12 | -1.49 |

**Utilisation :**
```python
import joblib, pandas as pd

kmeans = joblib.load('models/kmeans_geopolitics.pkl')
scaler = joblib.load('models/scaler_geopolitics.pkl')

event = pd.DataFrame([{
    'GoldsteinScale': -4.0, 'AvgTone': -6.0,
    'NumMentions': 80, 'NumSources': 5, 'NumArticles': 75
}])
cluster_id = kmeans.predict(scaler.transform(event))[0]
```

### 2. Prédiction temporelle de la stabilité (RandomForest)

Prédit le score de stabilité des 7 prochains jours en mode autorégressif à partir des 7 lags précédents et d'une moyenne mobile.

```python
import joblib, numpy as np

forecaster = joblib.load('models/forecasting_stability.pkl')
# 8 features : lag_1..lag_7 + rolling_mean_7
lags = np.array([[53.2, 51.8, 55.0, 49.3, 52.1, 50.7, 54.0, 52.1]])
prediction = forecaster.predict(lags)[0]
```

### 3. RAG — Assistant géopolitique (Mistral AI + FAISS)

Un pipeline RAG (Retrieval-Augmented Generation) permet d'interroger en langage naturel la base de 6 353 articles géopolitiques :

1. Embedding de la question via `mistral-embed`
2. Recherche des 2 articles les plus proches dans l'index FAISS
3. Génération de la réponse via `mistral-small-latest` avec le contexte récupéré

---

## Installation locale

### Prérequis

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (gestionnaire de paquets recommandé)

### Lancement

```bash
# Cloner le dépôt
git clone https://github.com/Ynvers/Hackahon-Isheroo-G10.git
cd Hackahon-Isheroo-G10

# Configurer la clé API
cp .example.env .env
# Éditer .env et renseigner MISTRAL_API_KEY

# Lancer le dashboard
cd dashboard
uv run streamlit run app.py
```

### Scripts utilitaires

```bash
# Tester que tous les modèles sont fonctionnels
python tester_tout.py

# Réentraîner le modèle de clustering
python clustering_situations.py

# Réentraîner le forecasting + générer les prévisions J+7
python prediction_temporelle.py
```

---

## Déploiement (Streamlit Cloud)

La clé API Mistral ne doit pas être poussée sur GitHub. Sur Streamlit Cloud, ajoutez-la dans **Settings → Secrets** :

```toml
MISTRAL_API_KEY = "votre_clé_ici"
```

L'application charge automatiquement cette valeur au démarrage.

---

## Indicateurs GDELT — Glossaire

| Indicateur | Description |
|---|---|
| **GoldsteinScale** | Potentiel de déstabilisation (-10 = conflit maximal, +10 = coopération maximale) |
| **AvgTone** | Sentiment émotionnel moyen des articles couvrant l'événement |
| **NumMentions** | Nombre de citations de l'événement dans les flux GDELT |
| **NumSources** | Nombre de sites d'information distincts ayant couvert l'événement |
| **NumArticles** | Nombre total d'articles publiés |
| **Score de Stabilité** | Score composite quotidien dérivé de Goldstein et Tone (50 = neutre) |

---

## 5 Insights Clés

1. **Alerte de Décembre 2025** : 1 097 événements recensés — le double de la moyenne mensuelle — signal d'instabilité majeur.
2. **Résilience diplomatique** : malgré le contexte, **75 %** des événements sont de type "Coopération" ou "Diplomatie".
3. **Le mois le plus tendu** : novembre affiche le Goldstein le plus bas (**+0.13**), frôlant la zone négative.
4. **Dramatisation médiatique** : le Tone moyen des articles de crise est de -1.72 contre +1.13 pour les articles normaux.
5. **Poids du Nigeria** : Plus de **3 300** événements conjoints, le Nigeria s'affirme de très loin comme le premier interlocuteur du Bénin.

---

*Fait avec 💙 par votre équipe préférée : G10*