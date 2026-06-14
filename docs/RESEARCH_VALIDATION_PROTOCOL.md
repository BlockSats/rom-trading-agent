# Research Validation Protocol

**Version :** 0.1-draft
**Statut :** draft
**Phase :** v0.45
**Gel numérique :** reporté (voir section 2)

---

## 1. Objet du protocole

Ce protocole sert à pré-déclarer les règles d'évaluation avant toute observation des résultats.

Son but est de :

- empêcher le déplacement rétroactif des critères après observation des résultats (*outcome-dependent threshold shifting*) ;
- séparer explicitement la phase d'exploration de la phase de confirmation ;
- limiter le biais de sélection humain et algorithmique (*researcher degrees of freedom*) ;
- encadrer les futures phases de validation quantitative : Walk-Forward Analysis, block-bootstrap, Deflated Sharpe Ratio (DSR), Probability of Backtest Overfitting (PBO).

Ce protocole ne prouve pas qu'une stratégie est rentable. Il établit les conditions préalables à toute affirmation de performance.

---

## 2. Pré-enregistrement en deux étapes

### 2.1 Structure méthodologique (définie en v0.45)

Les éléments suivants sont établis dès cette phase :

- rôles des données (section 3) ;
- statuts de décision autorisés (section 5) ;
- règles du budget de holdout (section 4) ;
- règles du paper-forward (section 7) ;
- structure des diagnostics statistiques futurs et leur applicabilité (section 6) ;
- règles de versionnement et d'interdiction de réécriture silencieuse (section 9).

### 2.2 Gel numérique (reporté à une phase future)

Les valeurs suivantes ne sont pas définies en v0.45. Elles seront fixées après création des séries de rendements périodiques et analyse de puissance préliminaire :

- seuils numériques de performance (expectancy, profit factor, drawdown, etc.) ;
- taille minimale d'effet économiquement pertinente ;
- niveau de significativité cible ;
- puissance statistique cible ;
- durée minimale du paper-forward ;
- nombre minimal d'observations et de trades ;
- seuils WFE, DSR ou PBO lorsqu'ils seront applicables.

**Aucune valeur numérique de validation n'est inventée pendant v0.45.**

---

## 3. Rôles des données

### `exploratory_data`

Données utilisées pour construire le moteur, les signaux, les stratégies et les outils de backtest. Elles peuvent servir à l'exploration itérative mais ne fournissent pas de confirmation indépendante. Toute stratégie construite sur ces données est considérée comme ayant vu ces données.

### `confirmatory_holdout`

Données réservées à la confirmation. Un holdout ouvert est irréversiblement considéré comme vu. Une période consommée ne peut pas confirmer une nouvelle version substantielle du protocole ou de la stratégie.

### `prospective_holdout_candidate`

Période identifiée comme candidate à un holdout. Les données correspondantes peuvent être collectées et accumulées de façon append-only. Elles ne sont ni ouvertes pour évaluation ni considérées comme un véritable test paper-forward. Elles constituent une réserve prospective dont l'ouverture future reste irréversible.

### `paper_forward`

Données produites en temps réel après gel complet de la stratégie, des paramètres, des coûts, du moteur de backtest, des métriques d'évaluation et du calendrier d'évaluation. Le paper-forward ne commence qu'après ce gel. Une donnée procéduralement non inspectée ne constitue pas un aveuglement parfait au comportement général du marché.

### Distinction walk-forward OOS / holdouts

**Walk-forward out-of-sample (`walk_forward_oos`)** : fenêtre test hors échantillon par rapport à la fenêtre train correspondante de la même passe. Elle reste exploratoire si la stratégie ou ses paramètres ont déjà été observés ou ajustés sur l'ensemble des données. Aucune indépendance statistique confirmatoire n'est revendiquée. Statut des rapports WFA actuels : `exploratory_data`, `mode: exploratory_walk_forward`.

**Confirmatory holdout** : données réservées avant toute exploration, utilisées en lecture unique après gel numérique du protocole. Non utilisé dans les rapports WFA actuels (`confirmatory_holdout_used: false`).

**Prospective holdout candidate** : données futures accumulées dans le registre `state/prospective_reserve.jsonl`, non encore ouvertes. L'ouverture est irréversible.

**Paper-forward** : simulation sur données futures en temps réel, selon calendrier pré-enregistré. Non activé (`paper_forward.enabled: false` dans `validation_policy.yaml`, `paper_forward_used: false` dans les rapports WFA).

Les rapports walk-forward générés par `build_walk_forward_report` portent le champ `validation_context` qui documente explicitement ces garanties pour chaque rapport produit.

---

## 4. Budget de holdout

Le protocole devra tenir à jour un registre qui compte :

- le nombre de configurations testées ;
- le nombre de versions du protocole ;
- les ouvertures de données confirmatoires (irréversibles) ;
- les périodes consommées ;
- les périodes contaminées par une modification substantielle ;
- les réserves encore disponibles.

### Statuts opérationnels futurs

| Statut | Signification |
|---|---|
| `reserved` | Période identifiée et réservée, non encore collectée |
| `accumulating` | Données en cours d'accumulation, non encore ouvertes |
| `opened` | Données ouvertes pour évaluation (irréversible) |
| `consumed` | Évaluation terminée, période épuisée |
| `contaminated` | Période exposée à une modification substantielle post-réservation |
| `retired` | Période retirée pour raison structurelle (données manquantes, exchange arrêté, etc.) |

Le registre opérationnel append-only est introduit en phase v0.46. Il enregistre des événements opérationnels de type `prospective_batch_archived` au format JSONL. La clé `datasets: []` présente dans `validation_policy.yaml` reste une configuration statique distincte du registre dynamique. L'archivage d'un lot dans la réserve prospective ne constitue ni l'ouverture d'un holdout ni le début d'un paper-forward. Le champ `independence_claimed: false` figurant dans chaque enregistrement empêche toute affirmation abusive d'indépendance statistique. Aucune date de holdout n'est choisie.

---

## 5. Statuts de décision autorisés

### `not_evaluable`

La stratégie ne peut pas être évaluée dans les conditions actuelles : données insuffisantes, manque de puissance statistique (*underpowered_for_declared_effect*), fenêtres OOS trop courtes, ou conditions structurelles non remplies. Ce statut n'est ni un succès ni un échec.

### `exploratory_only`

La stratégie a été étudiée sur des données exploratoires uniquement. Elle n'a pas encore été soumise à un holdout indépendant. Aucune affirmation de performance n'est possible.

### `fails_current_protocol`

La stratégie a été soumise à une évaluation dans les conditions du protocole et n'a pas satisfait les conditions requises. Ce statut est un résultat valide du laboratoire.

### `eligible_for_extended_paper_validation`

La stratégie satisfait les conditions nécessaires pour entrer dans une phase de paper-forward. Ce statut ne signifie pas que la stratégie est rentable. Il n'autorise pas le live trading. Il n'implique aucune promotion automatique.

**Précision :** une statistique non applicable (DSR ou PBO sur trop peu de splits, par exemple) ne vaut ni réussite ni échec. `underpowered_for_declared_effect` mène à `not_evaluable` et non à un succès.

---

## 6. Gate conjonctif futur

Le gate d'évaluation futur sera **conjonctif** : toutes les conditions requises doivent être satisfaites simultanément. Il ne s'agit pas d'une moyenne, d'un score agrégé ou d'un seuil pondéré.

La structure future pourra inclure, sans s'y limiter :

- qualité et complétude des données ;
- fenêtres OOS non chevauchantes en nombre suffisant ;
- nombre suffisant d'observations et de trades ;
- expectancy nette après coûts de transaction réalistes ;
- drawdown maximal ;
- dégradation IS vers OOS dans des bornes acceptables ;
- stabilité des paramètres sur les fenêtres walk-forward ;
- couverture multi-actifs cohérente ;
- block-bootstrap sur les rendements périodiques ;
- Walk-Forward Efficiency (WFE) lorsqu'applicable ;
- DSR lorsqu'applicable ;
- PBO lorsqu'applicable ;
- comptage honnête du nombre d'essais (*trial counting*).

**Aucun seuil numérique n'est fixé en v0.45.**

---

## 7. Contrat paper-forward

Le paper-forward est soumis aux règles suivantes :

- le calendrier d'évaluation est pré-enregistré avant le début du run ;
- les résultats intermédiaires sont descriptifs uniquement, sans conclusion sur le PASS ou le FAIL ;
- aucun PASS anticipé n'est autorisé (*optional stopping interdit*) ;
- aucune modification de la stratégie, des paramètres, des coûts ou du moteur n'est autorisée pendant le run ;
- toute modification substantielle interrompt l'expérience et invalide la période en cours comme confirmation ;
- la durée minimale, le nombre d'observations et les seuils numériques restent non définis en v0.45 ;
- ils seront fixés avant le début du premier run paper-forward.

---

## 8. Analyse de puissance

Avant tout test confirmatoire, une analyse de puissance devra définir :

- la taille minimale d'effet économiquement pertinente (pas statistiquement arbitraire) ;
- la puissance statistique cible ;
- le risque de faux positif accepté ;
- l'unité d'observation retenue (rendement périodique, trade, fenêtre WFA) ;
- la méthode de gestion de la dépendance temporelle (block-bootstrap ou simulation adaptée) ;
- le scénario de coûts retenu.

Ces valeurs sont inconnues en v0.45. Elles ne seront définies qu'après création des séries de rendements périodiques.

`underpowered_for_declared_effect` est une conclusion acceptable et honnête. Le manque de puissance ne doit jamais être converti en succès ou en échec artificiel.

---

## 9. Versionnement et gel

- Le protocole est initialement au statut `draft`.
- Le passage à `frozen` sera explicite, daté et associé à un hash du fichier de configuration.
- Toute modification substantielle après gel impose une nouvelle version avec un nouveau numéro.
- Aucune réécriture silencieuse d'un protocole déjà utilisé pour une évaluation n'est autorisée.
- Le gel numérique (seuils) est distinct du gel structurel (règles) et peut intervenir plus tard.

---

## 10. Résultat probable et philosophie

> Le résultat probable d'une stratégie candidate peut être `fails_current_protocol` ou `not_evaluable` — notamment avec la raison `underpowered_for_declared_effect`. Ce n'est pas un échec du projet. Cela signifie que le laboratoire fonctionne et refuse de transformer une absence de preuve en promesse de rentabilité.

`rom-trading-agent` est un laboratoire de recherche trading :

- research-first ;
- local-first ;
- paper-only par défaut ;
- sans promesse de rentabilité ;
- sans exécution d'ordre réel ;
- sans notion de `best_strategy`, classement, promotion ou sélection automatique.

Une absence de preuve est un résultat. Un protocole qui rejette honnêtement une stratégie faible est un protocole qui fonctionne.
