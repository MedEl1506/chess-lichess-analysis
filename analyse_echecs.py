"""
Projet Data : analyser 20 000 parties d'échecs Lichess et prédire si les Blancs gagnent.

Usage :
    python analyse_echecs.py                  # télécharge le CSV depuis GitHub
    python analyse_echecs.py data/chess.csv   # ou utilise un fichier local

Le script :
  1. charge et inspecte les données
  2. nettoie et prépare les variables
  3. fait une analyse exploratoire (graphiques enregistrés dans figures/)
  4. compare un modèle de base, une régression logistique et une forêt aléatoire
  5. écrit les résultats chiffrés dans resultats.txt
  6. génère un tableau de bord interactif : dashboard.html (à ouvrir dans le navigateur)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # pas de fenêtre, on enregistre les images
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

URL = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2024/2024-10-01/chess.csv"
SEED = 42
BLEU = "#1f3a5f"
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)
lignes_resultats = []


def noter(texte=""):
    """Affiche une ligne et la garde pour resultats.txt."""
    print(texte)
    lignes_resultats.append(texte)


# ---------------------------------------------------------------- 1. Chargement
source = sys.argv[1] if len(sys.argv) > 1 else URL
df = pd.read_csv(source)
noter(f"Données chargées : {df.shape[0]} parties, {df.shape[1]} colonnes")
noter(f"Colonnes : {', '.join(df.columns)}")

# ---------------------------------------------------------------- 2. Nettoyage
noter(f"\nValeurs manquantes au total : {int(df.isna().sum().sum())}")
nb_doublons = int(df.duplicated(subset="game_id").sum())
df = df.drop_duplicates(subset="game_id").dropna(
    subset=["winner", "white_rating", "black_rating", "time_increment", "opening_eco"]
).reset_index(drop=True)
noter(f"Parties en double supprimées : {nb_doublons} -> {len(df)} parties conservées")

noter("\nRésultat des parties :")
noter(df["winner"].value_counts().to_string())
part_nulles = (df["winner"] == "draw").mean()
noter(f"Part de parties nulles : {part_nulles:.1%} (retirées pour la prédiction)")

# Variables construites
df["rating_diff"] = df["white_rating"] - df["black_rating"]
base_inc = df["time_increment"].str.split("+", expand=True).astype(float)
df["base_minutes"] = base_inc[0]
df["increment_s"] = base_inc[1]
df["eco_famille"] = df["opening_eco"].str[0]  # A à E

# ---------------------------------------------------------------- 3. Exploration
parties = df[df["winner"] != "draw"].copy()
parties["blancs_gagnent"] = (parties["winner"] == "white").astype(int)
taux_blancs = parties["blancs_gagnent"].mean()
noter(f"\nVictoires des Blancs (hors nulles) : {taux_blancs:.1%}")

# 3a. victoires des Blancs selon l'écart de niveau
tranches = pd.cut(parties["rating_diff"], bins=[-2000, -300, -100, -25, 25, 100, 300, 2000])
par_tranche = parties.groupby(tranches, observed=True)["blancs_gagnent"].agg(["mean", "size"])
noter("\nTaux de victoire des Blancs selon (Elo Blancs - Elo Noirs) :")
noter(par_tranche.rename(columns={"mean": "taux", "size": "parties"}).round(3).to_string())

plt.figure(figsize=(7, 4))
plt.bar(range(len(par_tranche)), par_tranche["mean"], color=BLEU)
plt.xticks(range(len(par_tranche)), [str(i) for i in par_tranche.index], rotation=30, ha="right")
plt.axhline(0.5, color="red", linestyle="--", linewidth=1)
plt.ylabel("Part de victoires des Blancs")
plt.title("Victoires des Blancs selon l'écart de classement")
plt.tight_layout()
plt.savefig(FIG_DIR / "01_victoire_vs_ecart_elo.png", dpi=150)
plt.close()

# 3b. ouvertures les plus jouées et leur taux de victoire
ouvertures = (
    parties.groupby("opening_name")["blancs_gagnent"]
    .agg(taux="mean", parties="size")
    .query("parties >= 100")
    .sort_values("taux")
)
noter("\nOuvertures jouées au moins 100 fois (taux de victoire des Blancs) :")
noter("Les 5 plus favorables aux Noirs :")
noter(ouvertures.head(5).round(3).to_string())
noter("Les 5 plus favorables aux Blancs :")
noter(ouvertures.tail(5).round(3).to_string())

extremes = pd.concat([ouvertures.head(8), ouvertures.tail(8)])
plt.figure(figsize=(8, 6))
plt.barh(extremes.index, extremes["taux"], color=BLEU)
plt.axvline(0.5, color="red", linestyle="--", linewidth=1)
plt.xlabel("Part de victoires des Blancs")
plt.title("Ouvertures (100 parties minimum) : extrêmes")
plt.tight_layout()
plt.savefig(FIG_DIR / "02_ouvertures.png", dpi=150)
plt.close()

# 3c. comment les parties se terminent
plt.figure(figsize=(6, 4))
df["victory_status"].value_counts().plot(kind="bar", color=BLEU)
plt.title("Façon dont les parties se terminent")
plt.ylabel("Nombre de parties")
plt.tight_layout()
plt.savefig(FIG_DIR / "03_fin_de_partie.png", dpi=150)
plt.close()

# 3d. cadence de jeu
plt.figure(figsize=(6, 4))
parties["base_minutes"].clip(upper=30).plot(kind="hist", bins=30, color=BLEU)
plt.xlabel("Temps de base (minutes, limité à 30)")
plt.title("Cadences de jeu")
plt.tight_layout()
plt.savefig(FIG_DIR / "04_cadences.png", dpi=150)
plt.close()

# ---------------------------------------------------------------- 4. Modélisation
# On n'utilise PAS le nombre de coups (turns) ni le type de fin de partie :
# on ne les connaît qu'une fois la partie terminée (fuite d'information).
colonnes = ["white_rating", "black_rating", "rating_diff", "base_minutes", "increment_s"]
X = parties[colonnes].copy()
X["rated"] = parties["rated"].astype(int)
X = X.join(pd.get_dummies(parties["eco_famille"], prefix="eco").astype(int))
y = parties["blancs_gagnent"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

modeles = {
    "Régression logistique": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Forêt aléatoire": RandomForestClassifier(
        n_estimators=300, min_samples_leaf=5, random_state=SEED, n_jobs=-1
    ),
}

# Référence : toujours prédire la classe la plus fréquente
classe_majoritaire = int(y_train.mean() >= 0.5)
acc_base = accuracy_score(y_test, np.full(len(y_test), classe_majoritaire))

noter("\nRésultats (jeu de test = 20 % des parties) :")
noter(f"{'Modèle':<24}{'Précision':>10}{'AUC':>8}{'Précision CV':>14}")
noter(f"{'Classe majoritaire':<24}{acc_base:>10.3f}{0.5:>8.3f}{'-':>14}")
resultats_modeles = []  # gardé pour le tableau de bord
for nom, modele in modeles.items():
    modele.fit(X_train, y_train)
    pred = modele.predict(X_test)
    proba = modele.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, proba)
    acc_cv = cross_val_score(modele, X, y, cv=5, scoring="accuracy").mean()
    noter(f"{nom:<24}{acc:>10.3f}{auc:>8.3f}{acc_cv:>14.3f}")
    resultats_modeles.append(
        {"nom": nom, "acc": float(acc), "auc": float(auc), "acc_cv": float(acc_cv)}
    )

# Importance des variables (forêt aléatoire)
foret = modeles["Forêt aléatoire"]
importances = pd.Series(foret.feature_importances_, index=X.columns).sort_values()
noter("\nVariables les plus importantes (forêt aléatoire) :")
noter(importances.sort_values(ascending=False).round(3).head(5).to_string())

plt.figure(figsize=(7, 5))
importances.plot(kind="barh", color=BLEU)
plt.title("Importance des variables (forêt aléatoire)")
plt.tight_layout()
plt.savefig(FIG_DIR / "05_importance_variables.png", dpi=150)
plt.close()

# ---------------------------------------------------------------- 5. Tableau de bord
# Page HTML interactive générée à partir des VRAIS résultats calculés ci-dessus.
# Elle s'ouvre dans le navigateur (Chart.js est chargé depuis Internet).
top_importances = importances.sort_values(ascending=False).head(8)
donnees = {
    "n_parties": int(len(df)),
    "part_nulles": float(part_nulles),
    "taux_blancs": float(taux_blancs),
    "acc_base": float(acc_base),
    "modeles": resultats_modeles,
    "tranches": {
        "labels": [str(i) for i in par_tranche.index],
        "taux": [round(float(v) * 100, 1) for v in par_tranche["mean"]],
        "parties": [int(v) for v in par_tranche["size"]],
    },
    "ouvertures": {
        "labels": list(extremes.index),
        "taux": [round(float(v) * 100, 1) for v in extremes["taux"]],
        "parties": [int(v) for v in extremes["parties"]],
    },
    "importances": {
        "labels": list(top_importances.index),
        "valeurs": [round(float(v), 3) for v in top_importances.values],
    },
    "fins": {
        "labels": list(df["victory_status"].value_counts().index),
        "valeurs": [int(v) for v in df["victory_status"].value_counts().values],
    },
}

MODELE_HTML = """<!DOCTYPE html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Analyse de parties d'échecs Lichess</title>
<style>
:root{--bg:#fcfcfb;--card:#f1efe8;--ink:#0b0b0b;--ink2:#52514e;--grid:#e1e0d9;--blue:#2a78d6;--grey:#c3c2b7}
@media (prefers-color-scheme: dark){:root{--bg:#1a1a19;--card:#262624;--ink:#f0efec;--ink2:#c3c2b7;--grid:#2c2c2a;--blue:#3987e5;--grey:#898781}}
body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;background:var(--bg);color:var(--ink);margin:0;padding:24px;line-height:1.5}
main{max-width:980px;margin:0 auto}
h1{font-size:22px;font-weight:500;margin:0 0 4px}
h2{font-size:14px;font-weight:500;color:var(--ink2);margin:0 0 8px}
p.sub{color:var(--ink2);font-size:14px;margin:0 0 20px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-bottom:24px}
.card{background:var(--card);border-radius:8px;padding:14px 16px}
.card .l{font-size:13px;color:var(--ink2)}.card .v{font-size:24px;font-weight:500}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:28px}
.box{position:relative;height:280px}
section{margin-bottom:8px}
@media (max-width:520px){.grid{grid-template-columns:1fr}}
</style></head><body><main>
<h1>Analyse de parties d'échecs Lichess</h1>
<p class="sub">Prédire si les Blancs gagnent à partir du classement, de la cadence et de la famille d'ouverture.</p>
<div class="cards" id="cards"></div>
<div class="grid">
<section><h2>Victoires des Blancs selon l'écart de classement</h2><div class="box"><canvas id="c1"></canvas></div></section>
<section><h2>Précision des modèles (jeu de test)</h2><div class="box"><canvas id="c2"></canvas></div></section>
<section><h2>Importance des variables (forêt aléatoire)</h2><div class="box"><canvas id="c3"></canvas></div></section>
<section><h2>Ouvertures : extrêmes (100 parties minimum)</h2><div class="box"><canvas id="c4"></canvas></div></section>
<section><h2>Façon dont les parties se terminent</h2><div class="box"><canvas id="c5"></canvas></div></section>
</div></main>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script>
const D = __DONNEES__;
const css = getComputedStyle(document.documentElement);
const blue = css.getPropertyValue('--blue').trim(), grey = css.getPropertyValue('--grey').trim();
const ink = css.getPropertyValue('--ink2').trim(), grid = css.getPropertyValue('--grid').trim();
Chart.defaults.color = ink; Chart.defaults.font.size = 12;
const pct = v => v + ' %';
const best = D.modeles.reduce((a, b) => a.acc > b.acc ? a : b);
const fmt = (x, n) => x.toLocaleString('fr-FR', {maximumFractionDigits: n});
document.getElementById('cards').innerHTML = [
  ['Parties analysées', fmt(D.n_parties, 0)],
  ['Parties nulles', fmt(D.part_nulles * 100, 1) + ' %'],
  ['Victoires des Blancs (hors nulles)', fmt(D.taux_blancs * 100, 1) + ' %'],
  ['Meilleur modèle', best.nom + ' : ' + fmt(best.acc * 100, 1) + ' %'],
  ['Gain sur le modèle de base', '+' + fmt((best.acc - D.acc_base) * 100, 1) + ' pts']
].map(c => '<div class="card"><div class="l">' + c[0] + '</div><div class="v">' + c[1] + '</div></div>').join('');
const ax = (o) => Object.assign({grid: {color: grid}, border: {display: false}}, o || {});
const bar = (id, labels, data, colors, opts) => new Chart(document.getElementById(id), {
  type: 'bar',
  data: {labels: labels, datasets: [{data: data, backgroundColor: colors, borderRadius: 4, maxBarThickness: 24}]},
  options: Object.assign({responsive: true, maintainAspectRatio: false, plugins: {legend: {display: false}}}, opts)
});
bar('c1', D.tranches.labels, D.tranches.taux, blue, {
  plugins: {legend: {display: false}, tooltip: {callbacks: {label: c => pct(c.parsed.y) + ' (' + D.tranches.parties[c.dataIndex] + ' parties)'}}},
  scales: {x: ax({grid: {display: false}, title: {display: true, text: 'Elo Blancs - Elo Noirs'}}), y: ax({min: 0, max: 100, ticks: {callback: pct}})}
});
const noms = ['Classe majoritaire'].concat(D.modeles.map(m => m.nom));
const accs = [D.acc_base].concat(D.modeles.map(m => m.acc)).map(v => Math.round(v * 1000) / 10);
bar('c2', noms, accs, [grey].concat(D.modeles.map(() => blue)), {
  plugins: {legend: {display: false}, tooltip: {callbacks: {label: c => pct(c.parsed.y)}}},
  scales: {x: ax({grid: {display: false}}), y: ax({min: 0, max: 100, ticks: {callback: pct}})}
});
bar('c3', D.importances.labels, D.importances.valeurs, blue, {
  indexAxis: 'y',
  scales: {x: ax({min: 0}), y: ax({grid: {display: false}})}
});
bar('c4', D.ouvertures.labels, D.ouvertures.taux, blue, {
  indexAxis: 'y',
  plugins: {legend: {display: false}, tooltip: {callbacks: {label: c => pct(c.parsed.x) + ' (' + D.ouvertures.parties[c.dataIndex] + ' parties)'}}},
  scales: {x: ax({min: 0, max: 100, ticks: {callback: pct}}), y: ax({grid: {display: false}, ticks: {font: {size: 10}}})}
});
bar('c5', D.fins.labels, D.fins.valeurs, blue, {
  scales: {x: ax({grid: {display: false}}), y: ax({min: 0})}
});
</script></body></html>
"""

html = MODELE_HTML.replace("__DONNEES__", json.dumps(donnees, ensure_ascii=False))
Path("dashboard.html").write_text(html, encoding="utf-8")

Path("resultats.txt").write_text("\n".join(lignes_resultats), encoding="utf-8")
print("\nTerminé : figures dans figures/, résultats dans resultats.txt, tableau de bord dans dashboard.html")
