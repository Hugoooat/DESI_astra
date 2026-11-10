# Reprise du projet ASTRA vs density split (hors NERSC)

Ce document permet de reprendre le projet (seul ou avec Claude Code) à partir de
la sauvegarde `DESI_sav_complet.tar`, sans accès à NERSC. Sauvegarde faite le
2026-11-10, à la fin du stage (encadrant : Jaime E. Forero-Romero, Uniandes ;
introduit au projet par Valeria Torres-Gomez).

## 1. Contenu de la sauvegarde

```
DESI_sav_complet/
├── REPRISE.md                 ce document (copie de astra-ds/docs/REPRISE.md)
├── astra-ds/                  LE PROJET (dépôt git complet, historique inclus)
│   ├── astra_ds.py            mesure d'un catalogue : ASTRA, density split, quartiles, classes, clustering
│   ├── analysis.py            chargement des mesures, vecteurs de données, covariances, outils Fisher
│   ├── emulator.py            émulateur (réseaux, entraînement avec perte sur les différences, dérivées)
│   ├── scripts/               toutes les analyses et figures (voir annexe A de l'article)
│   ├── slurm/                 scripts de lancement sur Perlmutter
│   ├── config/                listes de catalogues mesurés, cosmologies, sélection des HOD
│   ├── paper/                 article/ (EN), article_fr/, vulgarisation/, poster/, figures/, archive/
│   ├── plots/                 toutes les figures produites par les scripts
│   ├── logs/                  logs des calculs (résultats intermédiaires imprimés)
│   ├── docs/                  NEXT_STEPS.md (questions pour Jaime), REPRISE.md, env_versions.txt
│   ├── data  -> ../astra-ds-store/data       (lien à recréer, voir §4)
│   └── models_pairs, models_* -> ../astra-ds-store/...
├── astra-ds-store/            DONNÉES DÉRIVÉES (3.4 Go)
│   ├── data/                  mesures : catalogues AbacusSummit (cXXX_hodYYY_*.npz), mocks officiels
│   │                          (official/), Quijote (quijote/, quijote_classes/), résultats Fisher (*.json),
│   │                          chaînes MCMC (chains/)
│   └── models_*/              émulateurs entraînés (models_pairs = émulateur de référence)
├── astra-ds-github/           copie de la branche GitHub astra_hugo_v2 (remote Hugoooat/DESI_astra)
├── article_hugon/             première version (V1) de l'article et ses figures
└── claude/                    pour reprendre avec Claude Code
    ├── memory/                notes de mémoire de Claude (MEMORY.md + fichiers)
    └── session_b9dd0fc2....jsonl   transcription complète de la conversation de travail
```

Données **non incluses** (trop lourdes ou non transportables) :
- catalogues Quijote bruts (814 Go) : publics, à retélécharger avec Globus en suivant
  https://quijote-simulations.readthedocs.io (halos Rockstar, snapshot 3, `out_3_pid.list`) ;
- catalogues HOD AbacusSummit de la suite d'émulation et mocks officiels DESI : uniquement sur NERSC
  (chemins dans l'annexe A de l'article). Toutes les mesures qui en dérivent sont dans `astra-ds-store/data`.

## 2. État du projet

- **Analyse terminée.** Référence : émulateur `models_pairs` (perte sur les différences),
  covariance des 1786 petites boîtes officielles, point fiduciel = HOD officielle,
  6 paramètres cosmologiques (ωb, ωcdm, ns, σ8, w0, wa). Quijote : sous-ensembles (Ωm, σ8)
  et (Ωm, ns, σ8), estimateur combiné de Coulton & Wandelt. Classes ASTRA sur Quijote complet.
- **Article** (`paper/article/main.tex`, version française `paper/article_fr/main.tex`) :
  - sections 1 à 10 relues par Hugo et révisées (tous les chiffres vérifiés dans les données) ;
  - **annexes A, B, C pas encore relues** par Hugo ;
  - auteurs : Hugo Nguyen, Jaime E. Forero-Romero (Uniandes) ; remerciements ; déclaration
    d'usage de l'IA (sous-section « Software » + « AI disclosure statement ») ;
  - en suspens : phrase sur le MCMC dans le résumé (version courte de Hugo ou version avec
    « which recovers unbiased, calibrated posteriors ») ; affiliation propre de Hugo à confirmer ;
    remerciements officiels DESI à demander à Jaime ; volume/page de Salcedo et al. à vérifier.
- **Vulgarisation** (`paper/vulgarisation/`) : explications en français, y compris les réponses
  aux relectures des sections 3 à 10.
- **Publication** : en discussion avec Jaime (autorisation, co-auteurs, relecteurs DESI, revue :
  JCAP / MNRAS / PRD envisagées). Une grande partie du travail repose sur des mocks DESI ;
  la partie Quijote repose sur des données publiques.

## 3. Reprendre avec Claude Code

1. Installer Claude Code et ouvrir un terminal dans le dossier `DESI_sav_complet/`.
2. Lui demander de lire, dans cet ordre : `REPRISE.md`, `claude/memory/MEMORY.md` et les fichiers
   qu'il référence, `astra-ds/README.md`, `astra-ds/paper/README.md`, `astra-ds/docs/NEXT_STEPS.md`.
3. La conversation complète est dans `claude/session_*.jsonl` (format JSON lignes, très long) :
   y chercher un sujet précis plutôt que la lire entièrement.
4. Pour compiler l'article : `cd astra-ds/paper && make` (pdflatex), ou importer
   `astra-ds/paper/` sur Overleaf (document principal `article/main.tex`).

## 4. Refaire tourner le code hors de NERSC

- Versions des paquets utilisées : `astra-ds/docs/env_versions.txt` (Python 3.12, numpy 2.4,
  scipy 1.17, torch 2.10, emcee 3.1.6, getdist 1.7.6 ; pycorr et Corrfunc version DESI,
  installables depuis https://github.com/cosmodesi/pycorr).
- Recréer les liens vers les données, depuis `astra-ds/` :
  ```bash
  rm -f data models_* && ln -s ../astra-ds-store/data data && for m in ../astra-ds-store/models_*; do ln -s $m $(basename $m); done
  ```
- Chemins NERSC codés en dur, à adapter si l'on remesure : `HOD_DIR` dans `astra_ds.py`,
  `QUIJOTE` dans `scripts/run_quijote.py`, `MOCKS` dans `scripts/run_official.py`.
- Ce qui tourne sur un ordinateur portable à partir des mesures sauvegardées :
  tous les Fisher Quijote (`scripts/quijote_fisher.py`, `quijote_classes_fisher.py`,
  `summarize_quijote.py`), les Fisher avec l'émulateur (`compare_methods.py`,
  `quartile_breakdown.py`, `smin_numbers.py`), les figures (`paper_figures*.py`,
  `classes_figures.py --only response fisher abacus_multipoles`, `readable_figures.py`),
  le MCMC (`mcmc.py`, plus lent). Ce qui nécessite les catalogues bruts : les mesures
  (`run_list.sh`, `run_official.py`, `run_quijote.py`), `env_maps.py`, `robustness.py`.
