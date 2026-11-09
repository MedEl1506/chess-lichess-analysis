# Analyse de 20 000 parties d'échecs Lichess

Projet Data Science : explorer environ 20 000 parties jouées sur Lichess et prédire, avant le début de la partie, si les Blancs vont gagner (classement des joueurs, cadence, famille d'ouverture).

Données : `chess.csv` du jeu de données [TidyTuesday du 1er octobre 2024](https://rfordatascience.github.io/tidytuesday/data/2024/2024-10-01/readme.html), issu du jeu de données Kaggle [datasnaek/chess](https://www.kaggle.com/datasets/datasnaek/chess/data) (parties collectées sur Lichess.org).

## Lancer le projet

1. Installer les bibliothèques : `pip install -r requirements.txt`
2. Lancer : `python analyse_echecs.py` (le script télécharge le CSV tout seul), ou `python analyse_echecs.py data/chess.csv` avec un fichier local.

Le script crée le dossier `figures/` (4 graphiques d'exploration et 1 d'importance des variables), le fichier `resultats.txt` et un tableau de bord interactif `dashboard.html` à ouvrir dans un navigateur (il est construit à partir des résultats de l'exécution).

## Questions posées

- L'avantage des Blancs existe-t-il, et de quelle taille ?
- Dans quelle mesure l'écart de classement entre les joueurs prédit-il le gagnant ?
- Quelles ouvertures réussissent le mieux aux Blancs ou aux Noirs ?
- Un modèle fait-il mieux que de toujours prédire le résultat le plus fréquent ?

## Méthode

- Nettoyage : 945 parties en double supprimées (19 113 parties conservées), aucune valeur manquante. Les parties nulles (4,6 %) sont retirées de la prédiction.
- Variables construites : écart de classement, temps de base et incrément (cadence), famille d'ouverture (codes ECO A à E).
- Le nombre de coups et le type de fin de partie sont volontairement exclus du modèle : ils ne sont connus qu'à la fin de la partie (fuite d'information).
- Modèles : régression logistique et forêt aléatoire, comparés à un modèle de base, évalués par précision et AUC sur un jeu de test (20 %) et par validation croisée à 5 plis.

## Résultats

- **Avantage des Blancs :** ils gagnent 52,4 % des parties décisives (hors nulles).
- **L'écart de classement domine :** les Blancs gagnent 17,7 % des parties quand ils ont au moins 300 points de moins que les Noirs, et 85,1 % quand ils en ont au moins 300 de plus.
- **Modèles :**

| Modèle | Précision (test) | AUC | Précision (validation croisée) |
| --- | --- | --- | --- |
| Classe majoritaire (référence) | 52,4 % | 0,500 | - |
| Régression logistique | 65,2 % | 0,717 | 65,1 % |
| Forêt aléatoire | 63,8 % | 0,707 | 63,3 % |

  La régression logistique gagne environ 13 points de précision sur la référence et fait mieux que la forêt aléatoire.
- **Variables les plus importantes (forêt aléatoire) :** l'écart de classement (0,41), puis le classement des Noirs (0,22) et des Blancs (0,21). La cadence pèse peu (environ 0,06 chacune).
- **Ouvertures :** les taux varient de 35,8 % (Van't Kruijs Opening, 327 parties) à 68,5 % (Queen's Pawn, 124 parties) en faveur des Blancs.
