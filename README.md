# Analyse de 20 000 parties d'échecs Lichess

Projet Data Science : explorer environ 20 000 parties jouées sur Lichess et prédire, avant le début de la partie, si les Blancs vont gagner (classement des joueurs, cadence, famille d'ouverture).

Données : `chess.csv` du jeu de données [TidyTuesday du 1er octobre 2024](https://rfordatascience.github.io/tidytuesday/data/2024/2024-10-01/readme.html), issu du jeu de données Kaggle [datasnaek/chess](https://www.kaggle.com/datasets/datasnaek/chess/data) (parties collectées sur Lichess.org).

## Lancer le projet

1. Installe les bibliothèques : `pip install -r requirements.txt`
2. Lance : `python analyse_echecs.py` (le script télécharge le CSV tout seul), ou `python analyse_echecs.py data/chess.csv` avec un fichier local.

Le script crée le dossier `figures/` (4 graphiques d'exploration et 1 d'importance des variables), le fichier `resultats.txt` et un tableau de bord interactif `dashboard.html` à ouvrir dans ton navigateur (il se construit à partir de tes vrais résultats).

## Questions posées

- L'avantage des Blancs existe-t-il, et de quelle taille ?
- Dans quelle mesure l'écart de classement entre les joueurs prédit-il le gagnant ?
- Quelles ouvertures réussissent le mieux aux Blancs ou aux Noirs ?
- Un modèle fait-il mieux que de toujours prédire le résultat le plus fréquent ?

## Méthode

- Nettoyage : doublons, valeurs manquantes, parties nulles retirées de la prédiction.
- Variables construites : écart de classement, temps de base et incrément (cadence), famille d'ouverture (codes ECO A à E).
- Le nombre de coups et le type de fin de partie sont volontairement exclus du modèle : ils ne sont connus qu'à la fin de la partie (fuite d'information).
- Modèles : régression logistique et forêt aléatoire, comparés à un modèle de base, évalués par précision et AUC sur un jeu de test (20 %) et par validation croisée à 5 plis.

## Résultats

(à compléter avec tes chiffres de `resultats.txt`)

## Pistes d'amélioration

- Utiliser les coups de l'ouverture (premiers coups) comme variables.
- Prédire aussi les parties nulles (3 classes).
- Régler les hyperparamètres avec GridSearchCV.
