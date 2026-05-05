# Bénin Geo-Watch : Radar Géopolitique et Sécuritaire 🇧🇯

**Hackathon iSHEERO × DataCamp Donates 2026 - Bénin Insights Challenge**

Bienvenue sur le dépôt officiel de l'Équipe 10. Notre projet, **Bénin Geo-Watch**, est un système d'analyse et d'alerte précoce conçu pour surveiller la stabilité géopolitique du Bénin en s'appuyant sur les données de la base mondiale GDELT.

---

## 🎯 Notre Problématique

*Comment transformer le bruit médiatique mondial en signaux d'alerte précoces pour anticiper les tensions et mesurer la stabilité du Bénin ?*

Face à la complexité des enjeux sécuritaires (notamment au Nord-Bénin) et diplomatiques, notre équipe a construit un pipeline de données capable d'écouter la base GDELT. Au-delà de la simple visualisation, nous avons développé un modèle de Machine Learning capable de classifier l'état du pays (Tension vs Crise) en fonction de signaux faibles : volume médiatique, chute du sentiment (AvgTone) et échelle de Goldstein.

## 👥 L'Équipe 10

Notre approche est pluridisciplinaire, couvrant l'ensemble du cycle de la donnée :

- **Data Engineer :** Extraction, nettoyage et fusion des données brutes GDELT et GKG.
- **Data Analyst :** Création du dashboard interactif PowerBI pour rendre les données lisibles par les décideurs.
- **ML Engineer / Data Scientist :** Ingénierie des caractéristiques temporelles et entraînement des modèles `RandomForest` pour la détection de crises.

---

## 📊 Le Dashboard Interactif (Livrable)

Notre dashboard a été entièrement développé avec **Streamlit** pour offrir une expérience fluide, rapide et interactive.

👉 **[Lien vers le Dashboard Streamlit en ligne]**(Remplacer ce texte par votre lien Streamlit Cloud)

Il permet de visualiser en un coup d'œil :
- Le score de stabilité global et l'évolution du "Tone" médiatique.
- La répartition des événements (Violence vs Coopération).
- Les partenaires diplomatiques majeurs (Le Nigeria en tête).

---

## 🚀 Reproductibilité et Installation

Ce dépôt contient tout le code nécessaire pour reproduire notre pipeline de données et nos modèles prédictifs.

### Structure du projet

```
├── datasets/            # Données sources nettoyées (events, GKG, scores)
├── notesbooks/          # Notebooks Jupyter (Data Engineering et ML)
├── Equipe10_Plateforme...pbix # Fichier source du Dashboard PowerBI
├── requirements.txt     # Dépendances Python
└── README.md            # Ce fichier
```

### Comment lancer l'analyse en local ?

1. Clonez ce dépôt GitHub :
   ```bash
   git clone https://github.com/Ynvers/Hackahon-Isheroo-G10.git
   cd Hackahon-Isheroo-G10
   ```
2. Installez les dépendances requises :
   ```bash
   pip install -r requirements.txt
   ```
3. Lancez les notebooks dans le dossier `notesbooks/` pour rejouer l'extraction de données et l'entraînement du modèle Random Forest.

---

## 💡 5 Insights Clés (Extrait)

1. **L'alerte de Décembre 2025 :** Avec 1097 événements (le double de la moyenne mensuelle), décembre a représenté un signal d'instabilité majeur.
2. **Résilience Diplomatique :** Malgré le contexte, 51% des événements globaux restent de type "Coopération".
3. **Le mois le plus tendu :** L'échelle de Goldstein montre que le mois de novembre a été le plus critique (score descendant à +0.13).
4. **Dramatisation médiatique :** Les articles couvrant une "crise" affichent un ton moyen de -1.72 contre +1.13 pour les articles normaux (écart massif de 2.85 points).
5. **Le poids du Nigeria :** Avec plus de 609 événements conjoints, le Nigeria s'affirme de très loin comme le premier interlocuteur du Bénin.

---
*Fait avec ❤️ par l'Équipe 10 pour le Bénin Insights Challenge 2026.*
