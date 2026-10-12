# Suite du projet ASTRA split vs density split

État au 12/10/2026. L'analyse de référence est figée (voir `README.md` et
l'annexe C de l'article) :

* **DESI-like (AbacusSummit)** : émulateur entraîné avec la perte sur les
  différences de catalogues (`models_pairs/`), covariance des 1786 petites
  boîtes officielles, point fiducial = HOD officielle.
* **Quijote** : sous-ensembles (Ωm, σ8) et (Ωm, ns, σ8), estimateur combiné
  de Coulton & Wandelt, critère de fiabilité (standard/compressé < 1.5).

Résultat robuste : ASTRA et DS contiennent une information comparable
(σ_DS/σ_ASTRA entre 0.75 et 1.1, DS légèrement devant) et sont
complémentaires (gain de 25 % à ×2.4 en les combinant). Le « gagnant »
paramètre par paramètre dépend des choix d'analyse à 10–40 %.

---

## 1. À faire pour finir l'article

- [ ] Références manquantes : article ASTRA (Forero-Romero et al.), prior
      HOD « yuan23 » des catalogues émulateur, vérifier le numéro arXiv de
      Coulton & Wandelt (2023).
- [ ] Relire la note « RELECTURE » sur la densité des halos Quijote
      (10× plus faible que les LRG, échelle effective d'ASTRA ≈ 27 Mpc/h).
- [ ] Relecture complète (cohérence des chiffres entre texte, tableaux et
      figures ; les figures viennent de `scripts/paper_figures.py`,
      `compare_methods.py`, `summarize_quijote.py`, `summarize_mcmc.py`).
- [ ] Pousser la branche `astra_hugo_v2` (voir plus bas).
- [ ] Archiver les données lourdes hors de `$PSCRATCH` (purge à 8 semaines
      sans accès) : `/pscratch/sd/h/hugon/astra-ds-store/` → CFS ou HPSS
      avant la version finale.
- [ ] Poster (à faire une fois l'article relu).

## 2. Améliorations possibles (par coût croissant)

| Amélioration | Coût | Ce que ça apporte | Statut |
|---|---|---|---|
| Courbe d'apprentissage de l'émulateur (25/50/75/100 % des catalogues) | ~1 h de nœud, aucune simulation | dit si plus de catalogues réduirait l'erreur de l'émulateur | **fait : plateau à ≈0.5 σ_box dès 50–75 %, doubler le jeu gagnerait ≲10 % → pas d'agrandissement** |
| Ne mesurer que les statistiques utiles (2PCF + Q1/Q4, sans ℓ=4) | code, ~3× plus rapide par catalogue | rend l'agrandissement du jeu d'entraînement ~3× moins cher | inutile pour l'instant (plateau) ; utile si de nouveaux catalogues (graines) arrivent |
| Agrandir le jeu d'entraînement **par paliers** | ~15–45 h-nœud par palier | — | **abandonné** : la courbe d'apprentissage montre un plateau fixé par le bruit de tirage d'un seul catalogue par HOD ; il faut plusieurs graines par HOD (question 1 à Jaime) |
| Fisher et MCMC sur un volume type DESI avec l'émulateur amélioré | minutes | contraintes réalistes, pas seulement une comparaison d'information | après l'amélioration de l'émulateur |
| Catalogues HOD générés avec **la même HOD dans toutes les cosmologies** (AbacusHOD) | dépend de l'accès aux sous-échantillons | dérivées propres et réalistes, plusieurs graines par point | bloqué (voir questions à Jaime) |
| Quijote : M_ν, Ω_b, h | plus de simulations de dérivées ou traceurs plus denses | contraintes sur M_ν | non faisable avec les données actuelles |

Remarque sur la vitesse : les calculs coûteux (comptage de paires avec
Corrfunc, triangulation de Delaunay avec scipy) sont CPU ; les nœuds GPU
n'apporteraient rien sans réécrire ces parties. Le levier réaliste est de
calculer moins de statistiques par catalogue.

## 3. Questions pour Jaime

1. **Sous-échantillons AbacusHOD.** Pour générer des catalogues avec la même
   HOD dans les cosmologies c100–c113 (et plusieurs graines par HOD), il faut
   les sous-échantillons de particules préparés pour AbacusHOD. Ceux de
   `/pscratch/sd/s/sihany/summit_subsamples_cleaned_desi` ne sont pas
   lisibles et ne contiennent pas c100–c113. Où sont ceux utilisés pour les
   catalogues de `ntbfin`, ou à qui demander l'accès ?
2. **Portée de l'étude.** La comparaison ASTRA vs density split suffit-elle
   pour l'article, ou faut-il aller plus loin en exploitant la nature
   **locale** d'ASTRA ? Pistes possibles :
   - classification par objet (vides, feuillets, filaments, nœuds) et
     clustering par type d'environnement ;
   - r comme marque continue (corrélations pondérées par r) ;
   - traitement naturel du masque du relevé par les points aléatoires
     (application aux données DESI réelles, comme ASTRA-DESI) ;
   - les autocorrélations des quartiles ASTRA semblent casser des
     dégénérescences (σ8 dans Quijote) : est-ce la direction à creuser ?
3. **Choix de l'analyse de référence.** Est-il d'accord avec : covariance des
   petites boîtes officielles, fiducial = HOD officielle, Quijote limité à
   (Ωm, σ8) et (Ωm, ns, σ8) ?
4. **Références** : article ASTRA à citer, et référence du prior HOD
   « yuan23 ».

## 4. GitHub

La branche est prête dans `~/home/astra-ds-github` (branche `astra_hugo_v2`
du dépôt `Hugoooat/DESI_astra`, sans push forcé ; l'ancien code reste sur
`astra_hugo`). Mises à jour : depuis `~/home/astra-ds-github`,
`git fetch local && git merge local/master` puis `git push origin astra_hugo_v2`.
