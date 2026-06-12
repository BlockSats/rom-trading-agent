# Walk-Forward Analysis — Design Document

Version : v0.40  
Statut : conception uniquement — aucune implémentation dans cette phase

---

## 1. Objectif de la Walk-Forward Analysis

La Walk-Forward Analysis (WFA) est une méthode de validation hors-échantillon des stratégies de trading.

Elle repose sur une **séparation stricte** entre :
- la période **In-Sample (IS)** ou *train* : les paramètres sont observés et peuvent être ajustés ;
- la période **Out-of-Sample (OOS)** ou *test* : la stratégie est évaluée sans modification.

L'objectif est de **réduire le risque d'overfitting** : une stratégie qui produit de bons résultats IS peut
simplement avoir été ajustée pour s'adapter au bruit passé. La WFA mesure si ces résultats se maintiennent
sur des données que la stratégie n'a jamais "vues".

**Avertissement explicite** : des résultats OOS positifs sur données historiques ne sont pas une preuve de
rentabilité future. Ils constituent un signal de robustesse supplémentaire, pas une garantie.

---

## 2. Principes méthodologiques

### Mode rolling (défaut)

La fenêtre IS a une taille fixe. Elle avance avec le temps. Le modèle "oublie" les données les plus anciennes.

```
Fenêtre 1 : IS [J0   → J365]  OOS [J365 → J455]
Fenêtre 2 : IS [J90  → J455]  OOS [J455 → J545]
Fenêtre 3 : IS [J180 → J545]  OOS [J545 → J635]
```

### Mode anchored (comparaison secondaire)

La fenêtre IS commence toujours au même point et grandit. Utile pour évaluer si plus de données améliorent
la stabilité.

```
Fenêtre 1 : IS [J0   → J365]  OOS [J365 → J455]
Fenêtre 2 : IS [J0   → J455]  OOS [J455 → J545]
Fenêtre 3 : IS [J0   → J545]  OOS [J545 → J635]
```

### Règles non négociables

- Pas de fenêtres OOS chevauchantes par défaut (évite la fuite de données).
- Aucune auto-sélection de stratégie dans la première implémentation.
- Les paramètres utilisés sur OOS sont identiques aux paramètres IS — pas d'optimisation OOS.
- Aucun live trading, aucun ordre réel.

---

## 3. Paramètres initiaux proposés pour crypto daily

Ces valeurs sont des **defaults de recherche**. Elles ne sont pas garanties pertinentes pour tous les assets
ou régimes de marché. Elles doivent être validées empiriquement au moment de l'implémentation.

| Paramètre | Valeur par défaut | Notes |
|-----------|-------------------|-------|
| `train_window` | 365 jours | ~1 an de données daily |
| `test_window` | 90 jours | ~1 trimestre OOS |
| `step` | 90 jours | Pas entre fenêtres successives |
| `min_trades_oos` | 20 | En dessous : fenêtre marquée comme insuffisante |
| `mode` | `rolling` | Défaut ; `anchored` disponible en comparaison |

**Données minimales requises** : au moins `train_window + test_window` jours de données pour qu'une seule
fenêtre soit possible. Pour 3 fenêtres rolling avec les defaults : environ 2 ans de données daily.

**Cas des assets à historique court** (< 1.5 an) : impossible de constituer les fenêtres. Le rapport doit
retourner une erreur explicite, pas un résultat silencieusement vide.

---

## 4. Architecture future proposée

Cette section décrit les **composants probables**, sans code. Les détails seront fixés lors de l'implémentation.

### 4.1 Découpage IS/OOS

Nouveau helper à créer :

```
split_walk_forward_windows(df, train_window_days, test_window_days, step_days, mode)
    → list[tuple[DataFrame, DataFrame]]  # (IS, OOS) par fenêtre
```

Similaire à `split_ohlcv_windows()` dans `research_robustness.py`, mais avec des tailles de fenêtres
en jours plutôt qu'en nombre de fenêtres.

### 4.2 Exécution backtest IS

Réutilise `run_backtest(df, strategy, initial_balance)` de `backtest.py` sans modification.
Les paramètres IS peuvent être observés mais, dans la première implémentation, ils sont fixes.

### 4.3 Exécution backtest OOS

Réutilise `run_backtest(df, strategy, initial_balance)` avec les **mêmes paramètres** que IS.
Pas d'optimisation, pas d'ajustement.

### 4.4 Agrégation des résultats OOS

Calcul des métriques agrégées sur l'ensemble des fenêtres OOS :
- OOS consistency rate
- Walk-forward efficiency (WFE)
- Statistiques descriptives des métriques OOS

### 4.5 Rapport JSON

Sérialisé dans `state/wfa_report.json`, cohérent avec les autres rapports existants.
Voir section 5 pour la structure indicative.

### 4.6 Affichage read-only CLI

Commande future `tradebot show-wfa-report`, cohérente avec les commandes `show-assets-report` et
`show-backtest-report` existantes. Lecture seule du fichier JSON — pas de recalcul.

---

## 5. Rapport JSON futur

Structure indicative. **Non figée** — à raffiner lors de l'implémentation.

```json
{
  "command": "walk-forward",
  "version": "0.1",
  "asset": "BTCUSDT",
  "strategy_id": "rsi_baseline",
  "parameters": {
    "rsi_period": 14,
    "rsi_oversold": 30
  },
  "mode": "rolling",
  "train_window_days": 365,
  "test_window_days": 90,
  "step_days": 90,
  "windows": [
    {
      "window": 1,
      "train_period": {
        "start": "2022-01-01",
        "end": "2022-12-31",
        "rows": 365
      },
      "test_period": {
        "start": "2023-01-01",
        "end": "2023-03-31",
        "rows": 90
      },
      "train_metrics": {
        "closed_trades": 42,
        "total_return": 0.12,
        "max_drawdown": -0.08,
        "sharpe_like": 1.1,
        "profit_factor": 1.4,
        "expectancy": 45.2
      },
      "oos_metrics": {
        "closed_trades": 8,
        "total_return": 0.03,
        "max_drawdown": -0.04,
        "sharpe_like": 0.8,
        "profit_factor": 1.1,
        "expectancy": 18.5
      },
      "trades_oos": 8,
      "warnings": []
    }
  ],
  "aggregated_oos": {
    "total_windows": 4,
    "windows_with_min_trades": 3,
    "average_oos_return": 0.025,
    "average_oos_drawdown": -0.045,
    "average_oos_expectancy": 20.1,
    "oos_consistency_rate": 0.75,
    "walk_forward_efficiency": 0.21
  },
  "warnings": [
    "1 fenêtre OOS avec moins de 20 trades"
  ]
}
```

Les champs `train_metrics` et `oos_metrics` reprennent les métriques déjà calculées par
`score_trades()` dans `scoring.py`.

---

## 6. Métriques futures

### Métriques par fenêtre OOS

| Métrique | Description |
|----------|-------------|
| `total_return` | Rendement total de la fenêtre OOS |
| `max_drawdown` | Drawdown maximum sur la fenêtre OOS |
| `winrate` | Proportion de trades gagnants |
| `profit_factor` | Ratio gains bruts / pertes brutes |
| `expectancy` | Espérance par trade en devise de cotation |
| `trades_count` | Nombre de trades fermés OOS |

### Métriques agrégées

**OOS consistency rate** : proportion de fenêtres OOS où `profit_factor >= 1`.  
Indique si la stratégie est en profit plus souvent qu'en perte sur les fenêtres OOS, pas si elle est
globalement rentable.

**Walk-forward efficiency (WFE)** : ratio du rendement OOS moyen sur le rendement IS moyen.  
Un WFE proche de 1 indique que les performances IS se transfèrent bien hors-échantillon.  
Un WFE faible ne signifie pas que la stratégie perd de l'argent — il signifie que les performances
IS ne se reproduisent pas à la même échelle OOS.

---

## 7. Place future de DSR et PBO

Le **Deflated Sharpe Ratio (DSR)** et la **Probability of Backtest Overfitting (PBO)** sont des outils
statistiques de correction pour les biais de sélection multiple.

### Prérequis techniques

- Une matrice de returns **T × N_configs** : T périodes, N_configs configurations testées.
- Le Sharpe doit être calculé **par période**, sans annualisation artificielle.
- `n_trials` doit refléter le nombre réel d'essais effectués — pas une estimation.
- Des tests synthétiques (données simulées) sont nécessaires pour valider l'implémentation.

### Séquence requise

DSR et PBO ne peuvent pas être implémentés avant :
1. Une WFA de base fonctionnelle (returns par fenêtre OOS)
2. Une returns matrix T × N_configs (multi-configs ou multi-assets)

**DSR et PBO viennent après la returns matrix, pas avant.**

---

## 8. Roadmap d'implémentation future

Les numéros de version seront assignés lors de chaque phase.

| Phase future | Description |
|--------------|-------------|
| v0.xx | Helpers de découpage walk-forward (`split_walk_forward_windows`) |
| v0.xx | Rapport WFA read-only sur une stratégie, un asset |
| v0.xx | WFA multi-assets (cohérent avec `compare-strategies-windows-csv`) |
| v0.xx | Returns matrix T × N_configs |
| v0.xx | DSR (Deflated Sharpe Ratio) |
| v0.xx | PBO (Probability of Backtest Overfitting) |

Chaque phase est indépendante et testable séparément.

---

## 9. Garde-fous

Ces règles s'appliquent à toutes les phases d'implémentation futures.

- **Pas de `best_strategy`** : aucune sélection automatique de la meilleure stratégie.
- **Pas de ranking** : aucun classement automatique des stratégies ou assets.
- **Pas de `global_score`** : pas de score agrégé cross-stratégies ou cross-assets.
- **Pas de `selected_strategy`** : aucune stratégie n'est automatiquement "choisie".
- **Pas d'auto-promotion** : aucune promotion automatique de candidat vers production.
- **Pas de live trading** : toutes les exécutions sont en simulation ou paper mode.
- **Pas d'ordres réels** : aucune connexion à un exchange en mode trading.
- **Pas de promesse de rentabilité** : les résultats OOS historiques ne prédisent pas les gains futurs.

---

## 10. Questions ouvertes

Ces questions doivent être tranchées lors des phases d'implémentation futures.

| Question | État |
|----------|------|
| **Fréquence daily vs 1h** | Daily proposé en premier pour simplicité et stabilité. 1h implique un coût de calcul et des historiques plus longs. |
| **Nombre minimum de trades OOS** | 20 proposé comme défaut. À valider : en dessous, les métriques OOS sont peu fiables. |
| **Assets à historique court** | Fenêtres impossibles si < `train_window + test_window` jours. Le rapport doit échouer explicitement avec un message clair. |
| **Coût de calcul** | Acceptable pour crypto daily sur laptop. À évaluer pour 1h avec plusieurs assets et configs. |
| **Stabilité des paramètres IS** | La première implémentation utilise des paramètres fixes. L'optimisation IS est une phase séparée et ultérieure. |
| **Intégration aux rapports existants** | La WFA devra s'intégrer au format `assets_comparison_report.json` dans une phase future. |

---

*Ce document est une conception uniquement. Aucune implémentation n'est incluse dans v0.40.*  
*Voir la roadmap (section 8) pour les phases d'implémentation.*
