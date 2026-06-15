# Experiment Artifacts Architecture

**Version :** v0.51
**Statut :** contrat architectural — conception uniquement, aucune implémentation dans cette phase
**Portée :** définit les concepts, identités, responsabilités et relations reliant manifeste,
exécution, configuration, equity/rendements périodiques, registre et future matrice `T × N_configs`.

> Ce document est un **contrat de conception**. Il ne contient aucun code, ne déclenche aucun
> backtest, ne sélectionne ni ne promeut aucune stratégie. Les implémentations correspondantes
> sont planifiées par phases (§18) et chacune fera l'objet de tests dédiés.

---

## 1. Objectifs et limites

### 1.1 Objectif

Établir un vocabulaire et un contrat stables **avant** d'implémenter les futurs artefacts
d'expérience, afin que chaque artefact produit ultérieurement (equity périodique, rendements,
registre, matrice) soit traçable, reproductible et méthodologiquement honnête.

### 1.2 Confusions que ce contrat empêche

- exécution technique ≠ essai statistique ;
- même configuration relancée ≠ nouvelle configuration ;
- Walk-Forward OOS ≠ holdout confirmatoire ;
- rendement par trade ≠ rendement périodique ;
- reproductibilité logicielle ≠ validité statistique ;
- nombre de runs ≠ nombre d'essais pour le DSR ;
- plusieurs actifs corrélés ≠ plusieurs preuves indépendantes.

### 1.3 Limites explicites

Ce document **ne** prouve **pas** qu'une stratégie est rentable. Il **n'introduit pas** :
`best_strategy`, `best_asset`, `winner`, `rank`, `ranking`, `global_score`, `selected_strategy`.
Aucune sélection, promotion ou classement automatique. Le scoring reste un outil de **lecture et
de diagnostic**. Le projet reste research-first, paper-only, local-first, sans live trading, sans
ordre réel, sans clé API réelle, sans secret.

---

## 2. État de référence (constaté dans le dépôt)

- `experiment_manifest.build_experiment_manifest` produit un `experiment_id` déterministe
  (SHA-256 d'un *identity payload* ; `created_at` exclu). Builder pur ; **non câblé au CLI**.
- `fingerprint.py` fournit `canonicalize_for_fingerprint`, `fingerprint_payload`,
  `fingerprint_dataframe`, `build_input_fingerprints` (canonicalisation robuste, `NaN`/infini →
  sentinelle `__special__`, tz → UTC, clés triées, `set` refusé).
- `walk_forward.split_walk_forward_windows` produit des fenêtres `train/test` indexées ;
  `step == test_window` ⇒ OOS non chevauchants. Aucun identifiant persistant.
- `walk_forward_report.build_walk_forward_report` backteste **la fenêtre test seulement**, attache
  des labels méthodologiques par fenêtre et un `validation_context` global.
- `backtest`/`scoring` ne produisent que des **rendements par trade** ; `sharpe_like` n'est **pas**
  un Sharpe périodique. **Aucune equity périodique** n'est capturée bien que
  `broker_paper.PaperBroker.equity()` existe.
- `storage.append_jsonl` / `read_jsonl` sont disponibles ; `prospective_reserve.jsonl` constitue un
  **précédent de registre append-only**.
- `config/validation_policy.yaml` ancre déjà : `observation_unit: periodic_return`,
  `dependence_method: block_resampling`, `dsr/pbo` en `not_computed`.

---

## 3. Taxonomie des identifiants

Quatre catégories distinctes, à ne jamais confondre :

| Catégorie | Définition | Propriété clé | Exemples |
|---|---|---|---|
| **1. Identité de définition** | dérivée du contenu, déterministe, **stable entre reruns** | reproductible | `study_id`, `configuration_id`, `experiment_id`, `window_definition_id` |
| **2. Identité d'occurrence** | **unique par exécution**, *non* dérivée du contenu | distingue les exécutions | `run_id`, `run_window_id` |
| **3. Fingerprint de contenu** | SHA-256 du contenu effectif (valeurs comprises) | détecte l'identité de contenu | `run_input_fingerprint`, `equity_content_fingerprint`, `returns_content_fingerprint`, `dataframe_sha256`, `file_sha256` |
| **4. Identité d'artefact** | un artefact produit, décomposé en 3 éléments distincts | identité indépendante de l'emplacement | `artifact_id`, `artifact_content_fingerprint`, `artifact_path` |

Règle générale : une **identité de définition** ne change que si le contenu défini change ; une
**identité d'occurrence** est neuve à chaque exécution même si les entrées sont identiques ; un
**fingerprint de contenu** inclut les valeurs et pas seulement les métadonnées.

**Décomposition de l'identité d'artefact (catégorie 4).** Un artefact produit (manifeste JSON,
equity curve, série de rendements, rapport WF) ne doit **jamais** être identifié par son chemin
de fichier. On distingue :

- **`artifact_id`** : identité **logique** de l'artefact produit (ce qu'il est dans le graphe :
  par ex. l'equity du `run_window_id` X). Stable.
- **`artifact_content_fingerprint`** : empreinte **déterministe de son contenu** (catégorie 3).
- **`artifact_path`** : **emplacement local modifiable** sur le disque.

Un **déplacement ou un renommage** de fichier change uniquement `artifact_path` ; il **ne modifie
ni `artifact_id` ni `artifact_content_fingerprint`** — donc ni l'identité méthodologique ni le
fingerprint de contenu de l'artefact.

---

## 4. `study_id` — périmètre d'étude (catégorie 1)

`study_id` représente **un ensemble pré-déclaré de configurations comparées dans un même protocole
de recherche**. C'est le périmètre obligatoire de tout comptage d'essais.

**Nom retenu.** « study » est le terme standard en statistique et en pré-enregistrement (une
*study* déclare à l'avance ses hypothèses et ses comparaisons multiples — exactement le cadre de
`n_trials`). Plus neutre que `research_campaign_id` (connotation opérationnelle) et plus complet
que `candidate_universe_id` (ne capture pas la notion de protocole pré-déclaré).

**Rôle.** Rattache : configurations candidates, expériences, runs, matrice `T × N_configs`, règle
de sélection in-sample, nombre d'essais déclaré, futurs DSR/PBO.

**Immuabilité et pré-déclaration.** Un `study_id` désigne un **périmètre immuable**. Il **change**
notamment si change :

- l'**univers des configurations candidates** ;
- le **protocole de comparaison** ;
- la **règle de sélection in-sample** ;
- le **rôle ou le périmètre des données** ;
- une **politique méthodologique déterminante**.

**Interdit :** ajouter rétroactivement de nouvelles configurations sous un même `study_id` —
cela rendrait le périmètre de `n_trials` ambigu. Toute extension de l'univers crée une **nouvelle
étude**.

**Règle dure.** `n_trials` n'est **jamais** calculé sur l'historique indifférencié du projet ; il
est toujours interprété **dans le périmètre d'un `study_id`**. Le registre enregistre
l'**appartenance** à l'étude **sans calculer** `n_trials`.

Champs d'identité (catégorie 1) : nom/version de l'étude, ensemble pré-déclaré des
`configuration_id`, protocole/règle de sélection pré-déclarée, `validation_policy` de référence.

---

## 5. `configuration_id` — configuration comportementale (catégorie 1)

Représente une **configuration comportementale de stratégie**, réutilisable sur plusieurs marchés.

**Inclut** tous les paramètres pouvant modifier signaux / risque / résultats :

- `strategy_id` ;
- **`timeframe`** ;
- paramètres d'entrée ;
- paramètres de sortie ;
- paramètres de risque ;
- hypothèses de **frais** ;
- hypothèses de **slippage** (lorsqu'elles existent) ;
- tout autre paramètre comportemental.

**Exclut** l'**actif** : la même logique doit pouvoir être évaluée sur plusieurs marchés sans
changer d'identité de configuration.

**Décision documentée.** Le **timeframe fait partie** de `configuration_id` (c'est un paramètre
comportemental qui modifie signaux et résultats). Seul l'**actif** en est exclu. L'actif, les
données, l'objectif, la politique de validation, les paramètres Walk-Forward et le contexte
méthodologique sont rattachés au niveau de l'`experiment_id` (§6).

---

## 6. `experiment_id` — entrées méthodologiques (catégorie 1)

Identité **stable des entrées méthodologiques** d'une expérience particulière : configuration +
**actif** + **timeframe** + **données** + **objectif** + **politique de validation** + **paramètres
Walk-Forward** + **contexte méthodologique**.

**Calcul inchangé en v0.51.** L'algorithme existant
(`experiment_manifest.build_experiment_manifest`) n'est pas modifié pendant cette phase :
`experiment_id = SHA-256(strategy_id, asset, timeframe, data_role, walk_forward_parameters,
validation_context, input_fingerprints)`, `created_at` exclu. Le `configuration_id` (§5) est une
**vue conceptuelle complémentaire** à introduire en v0.52 ; il ne remplace pas l'`experiment_id`.

> Remarque : le `timeframe` apparaît à la fois dans `configuration_id` (paramètre comportemental)
> et dans `experiment_id` (contexte de l'expérience). Ce n'est pas une contradiction : ce sont deux
> identités de niveaux différents.

---

## 7. `run_id` et empreintes associées

Un **run** est une **occurrence concrète d'exécution**.

- **`run_id`** *(catégorie 2)* : identifiant **unique d'occurrence**. **Formule non figée.** Il
  n'est *pas nécessairement* un hash de contenu ; il ne dépend *pas* de `code_version` (dont la
  collecte n'est pas encore définie) ni uniquement de `created_at` (deux exécutions pourraient
  partager l'horodatage). Un compteur local croissant ou un ULID est acceptable.
- **`run_input_fingerprint`** *(catégorie 3)* : empreinte **déterministe des entrées effectivement
  exécutées**. Permet de détecter qu'un run **réexécute les mêmes entrées déclarées** qu'un autre.
- **Provenance code** (git sha / `code_fingerprint`) : métadonnée **optionnelle future**,
  **hors périmètre v0.51**. La collecte Git n'est pas définie à ce stade.

**Conséquence méthodologique (formulation prudente).** Il faut distinguer trois degrés :

- **même `experiment_id`** : même **définition méthodologique déclarée** ;
- **même `run_input_fingerprint`** : **mêmes entrées déclarées** (on parle alors de *réexécution
  des mêmes entrées déclarées* / *répétition probable des entrées*) ;
- **répétition technique pleinement confirmée** : mêmes entrées **et** provenance d'exécution
  compatible — possible **seulement lorsque** `code_fingerprint` (ou une provenance équivalente)
  existera.

Tant que cette provenance n'existe pas, ces runs **ne sont pas** comptés automatiquement comme de
nouveaux essais statistiques, **ni** affirmés comme strictement équivalents (§13).

---

## 8. Fenêtres Walk-Forward

Deux identités distinctes :

- **`window_definition_id`** *(catégorie 1)* : identité **stable** d'une fenêtre — bornes
  `train_start/end`, `test_start/end`, `window_index`, paramètres WF. Inchangée d'un run à l'autre.
- **`run_window_id`** *(catégorie 2)* : **occurrence** de cette fenêtre dans un run donné.

Objectif : une fenêtre méthodologiquement identique **ne change pas d'identité** simplement parce
qu'elle est recalculée dans un autre run.

**Rattachement.** Clé unique `(run_id, window_index)` → `run_window_id`. Avant toute concaténation
de rendements OOS, asserter `step ≥ test_window` (pas de chevauchement implicite). Les rendements
ne proviennent **jamais** du train. Un OOS ne devient **jamais** un holdout confirmatoire par
recalcul. Les labels existants de `walk_forward_report.py` sont réutilisés.

---

## 9. Equity périodique (artefact primaire)

**Décision.** La **courbe d'equity périodique** (mark-to-market à la **clôture de barre**) est
l'**artefact primaire / source de vérité**. Les **rendements périodiques** en sont un **artefact
dérivé déterministe** (§10).

**Justification.** L'equity curve permet de vérifier le capital, recalculer les rendements,
contrôler les drawdowns, observer les périodes sans position et détecter les incohérences. Les
rendements seuls ne le permettent pas.

Contrat (à implémenter en v0.53, **non implémenté ici**) :

- **Source** : mark-to-market à la clôture de chaque barre via `PaperBroker.equity()`.
- **Premier point** : `equity_0 = capital_initial`.
- **Timestamp** : clôture de barre, **UTC** tz-aware, ISO 8601 (`…Z`), ordre chronologique strict,
  index unique et monotone.
- **Périodes sans position** : equity = cash constant.
- **Frais / slippage** : intégrés dans l'equity (net) ; le **brut** est conservé séparément.
- **Coûts de financement** : non modélisés (spot, sans levier) ⇒ `0` ; documenté comme hors
  périmètre / futur.

### 9.1 Valeurs non finies (`NaN` / infini)

`fingerprint.py` **sait canonicaliser** `NaN` et les infinis via la sentinelle `__special__` ;
**cela ne signifie pas qu'il les interdit**. Pour l'equity et les rendements, les valeurs non
finies doivent être **rejetées explicitement par une validation dédiée** — jamais simplement
canonicalisées puis acceptées.

### 9.2 Barres manquantes

Le contrat impose :

- **détection explicite des gaps** ;
- **aucune interpolation ni création silencieuse** de rendements ;
- une **politique de rejet ou de traitement déclarée explicitement et fingerprintée**.

La décision exacte (rejet strict, segmentation, etc.) est **différée à v0.53**. Ne pas figer
« trou = erreur » comme unique politique universelle.

### 9.3 Identité de contenu — `equity_content_fingerprint` (catégorie 3)

L'equity curve étant l'artefact **primaire**, elle possède son **propre fingerprint de contenu**.
Le `equity_content_fingerprint` doit inclure **au minimum** :

- version du schéma ;
- timestamps ;
- valeurs d'equity ;
- capital initial ;
- fréquence ;
- rôle méthodologique ;
- `net` ou `gross` ;
- hypothèses de frais et de slippage ;
- `capital_mode` (§11) ;
- `boundary_position_policy` (§11) ;
- politique de traitement des gaps (§9.2) ;
- autres métadonnées déterminantes.

Les rendements dérivés conservent leur propre `returns_content_fingerprint` (§10.1). Deux equity
curves différentes ne peuvent jamais partager le même `equity_content_fingerprint` au seul motif
de métadonnées identiques.

---

## 10. Rendements périodiques (artefact dérivé)

Dérivés **déterministes** de l'equity curve :

- la série de rendements commence à `t = 1` : `r_t = equity_t / equity_{t-1} − 1`. **On ne fabrique
  pas `r_0`.**
- **Périodes sans position** : rendement `0.0` explicite (pas de trou).
- **Type** : rendement **simple net** comme dérivation de référence ; le **log** est un dérivé
  supplémentaire calculable pour les usages statistiques.
- **Distinctions obligatoires** : `gross | net`, `train | test`,
  `in_sample | oos | holdout | paper_forward`.
- **Pas de `NaN` / infini** (cf. §9.1, validation dédiée).

### 10.1 Identité d'une série — `returns_content_fingerprint` (catégorie 3)

L'identité de contenu d'une série **inclut les valeurs**. Le `returns_content_fingerprint` couvre :
schéma, **timestamps**, **valeurs de rendement**, type (simple/log), fréquence, rôle
méthodologique, métadonnées déterminantes (net/gross, capital, politiques §11). **Deux séries
différentes ne peuvent jamais partager le même fingerprint** au seul motif de métadonnées
identiques. On distingue la *définition* de série (catégorie 1, sa provenance) de son *fingerprint
de contenu* (catégorie 3).

---

## 11. Politiques de capital et de frontière (explicites)

Le capital entre fenêtres et le traitement des positions aux frontières **ne sont pas des vérités
architecturales universelles**. Ce sont des **politiques explicites** :

```yaml
capital_mode:
  - reset_per_window      # capital réinitialisé à chaque fenêtre (fenêtres indépendantes)
  - continuous            # capital chaîné entre fenêtres

boundary_position_policy:
  - force_close           # clôture toute position résiduelle en fin de fenêtre
  - carry                 # reporte la position sur la fenêtre suivante
  - reject_open_position  # rejette/exclut une fenêtre se terminant avec une position ouverte
```

Règles applicables à toute politique :

- elle est **enregistrée** dans les artefacts ;
- elle **participe à l'identité méthodologique et aux fingerprints** (donc à
  `run_input_fingerprint`, `equity_content_fingerprint` et
  `returns_content_fingerprint`) ;
- elle est **visible dans les artefacts produits** ;
- elle n'est **jamais appliquée silencieusement**.

**Première implémentation future probable** : `capital_mode: reset_per_window` +
`boundary_position_policy: force_close`. Ce choix devra néanmoins être déclaré, enregistré et
fingerprinté comme n'importe quelle autre politique.

### 11.1 Validation de compatibilité des combinaisons

Toutes les combinaisons ne sont **pas** cohérentes. Une **règle de validation** explicite devra
rejeter les combinaisons incompatibles. Exemples indicatifs :

| Combinaison | Statut |
|---|---|
| `reset_per_window` + `force_close` | compatible |
| `reset_per_window` + `reject_open_position` | compatible |
| `reset_per_window` + `carry` | **incompatible** (réinitialiser le capital tout en reportant une position est contradictoire) |
| `continuous` + `carry` | potentiellement compatible |
| autres combinaisons | à **évaluer explicitement** |

La **validation définitive** des combinaisons et les **valeurs par défaut** restent **différées à
v0.53**.

---

## 12. Registre append-only — options (non figé)

Responsabilités d'un futur registre : enregistrer expériences, configurations, runs, fenêtres,
artefacts, fingerprints, statuts méthodologiques, relations parent/enfant et répétitions
techniques. Il **n'a jamais le droit de** : désigner un vainqueur, sélectionner/promouvoir une
configuration, calculer un score global automatique, ouvrir un holdout, ou décider seul qu'un run
compte comme un essai statistique.

Comparaison (rien n'est figé) :

| Option | Simplicité | Duplication | Intégrité réf. | Lecture | Append-only | Reconstruction | Migration schéma |
|---|---|---|---|---|---|---|---|
| (a) Journal événementiel unique `events.jsonl` typé | Élevée | Faible | Manuelle | Filtrage requis | Naturel | Par rejeu | Schémas mêlés (plus dur) |
| (b) Quelques registres spécialisés (2–3) | Moyenne | Faible | Plus claire | Ciblée | Naturel | Directe | Par fichier (plus simple) |
| (c) Cinq JSONL séparés | Faible | Moyenne | Surface large | Très ciblée | Naturel | Plus de jointures | Par fichier mais 5× |

**Orientation : structure minimale.** Point de départ probable — les **fenêtres et séries sont
intégrées dans un artefact de run** (un fichier par run) plutôt que dans un registre indépendant ;
le registre append-only reste **réduit** (au plus quelques fichiers : déclaration
d'études/configurations + faits de runs), en suivant le précédent `prospective_reserve.jsonl` via
`storage.append_jsonl`/`read_jsonl`. Le **nombre exact de fichiers est décidé en v0.55**, pas
maintenant. **SQLite reste hors périmètre** sauf justification forte.

---

## 13. Concepts statistiques

- **Expérience méthodologique** : un `experiment_id`.
- **Configuration candidate** : un `configuration_id`.
- **Exécution technique** : un `run_id`.
- **Réexécution des mêmes entrées déclarées** : ≥ 2 `run_id` partageant le même
  `run_input_fingerprint`. Leur relation exacte avec `experiment_id` dépendra du contrat v0.52.
  C'est une **répétition probable des entrées** ; une **répétition technique pleinement confirmée** exige en plus une provenance
  d'exécution compatible (future, cf. §7). Dans tous les cas, ces runs **ne comptent pas**
  automatiquement comme essais multiples.
- **Essai exploratoire** : exécution sur `exploratory_data`, jamais comptée comme preuve.
- **Essai statistique** : unité candidate au DSR/PBO ; **décision différée**, jamais déduite
  automatiquement.
- **Nombre brut de runs** : `count(run_id)`.
- **Nombre de configurations distinctes** : `count(distinct configuration_id)`.
- **Nombre d'essais déclaré (`n_trials`)** : valeur **déclarée et justifiée humainement**, **auditée
  contre** le périmètre immuable du `study_id`, les configurations enregistrées et l'historique
  d'exploration pertinent. Elle **n'est jamais déduite automatiquement du seul nombre de runs**.
- **Nombre effectif d'essais dépendants** : ajustement pour la corrélation entre configs/actifs ;
  **non calculé** maintenant — seuls les faits nécessaires sont enregistrés.

---

## 14. Matrices de rendements (préconditions futures)

> Distinction importante (catégorie de matrice) :
> - **`oos_returns_matrix`** : matrice de rendements **Walk-Forward OOS synchronisés**, destinée
>   aux **comparaisons et diagnostics** OOS. C'est l'objet construit en v0.56.
> - **`pbo_input_matrix`** : **future entrée du PBO/CSCV**. Elle **n'est pas** automatiquement
>   égale à l'`oos_returns_matrix`. La revue d'architecture DSR/PBO (au-delà de v0.56) devra
>   déterminer si l'`oos_returns_matrix` peut être réutilisée telle quelle, doit être transformée,
>   ou si une matrice distincte est nécessaire (partitions CSCV, dépendance temporelle, règle de
>   sélection in-sample pré-déclarée).

Préconditions de l'`oos_returns_matrix` (matrice **non implémentée** ici) :

- index temporel **commun** (même fréquence, UTC), aligné par timestamp de clôture de barre ;
- **identité des colonnes non figée en v0.51.** `configuration_id` **exclut l'actif** (§5), et une
  même configuration peut produire des séries différentes sur BTC, ETH ou d'autres marchés. v0.56
  devra **choisir explicitement** entre, notamment :
  - une **matrice distincte par actif**, colonnes = `configuration_id` ;
  - colonnes identifiées par `experiment_id` ;
  - colonnes identifiées par `(configuration_id, asset)` ;
  - une autre **identité composite explicitement définie**.

  Ce choix **n'est pas tranché en v0.51**.
- **scoping par `study_id`** ;
- uniquement des **rendements OOS non chevauchants** ;
- provenance traçable (`run_id`, fingerprints) ;
- **aucune valeur manquante après alignement** (sinon période exclue explicitement) ;
- métadonnées minimales par colonne : identité de colonne retenue, `run_id`, `asset`, fingerprints,
  type de rendement, `net|gross`, politiques §11 ;
- détection des doublons `(identité de colonne, période)` ;
- **garde anti-inflation** : refuser/signaler les configurations quasi-identiques créées
  uniquement pour gonfler le nombre de colonnes.

---

## 15. Préconditions DSR / PBO (aucune implémentation)

**DSR** nécessitera : séries de rendements **périodiques** (pas par-trade) ; **Sharpe par période**
(non annualisé) ; longueur d'échantillon `T` ; skewness ; kurtosis ; **nombre d'essais déclaré et
scoped par `study_id`** ; dépendance entre essais ; provenance du candidat évalué
(`configuration_id` / `run_id`).

**PBO / CSCV** nécessitera une **`pbo_input_matrix`** : sa relation avec l'`oos_returns_matrix`
(§14) reste à déterminer — réutilisation, transformation, ou matrice distincte. Préconditions :
`N_configs` ; `N_périodes` ; **règle de sélection in-sample explicite et pré-déclarée** ; critère
d'évaluation OOS ; partitions ; gestion de la dépendance temporelle ; garde anti-inflation de
configurations.

DSR et PBO restent **au-delà de v0.56** et exigeront une **nouvelle revue d'architecture**, qui
tranchera notamment la construction de la `pbo_input_matrix`.

---

## 16. Contrôles synthétiques (catalogue, non implémenté)

Tests méthodologiques futurs, chacun attendu pour vérifier un comportement **conservateur** (jamais
de fausse preuve) : equity constante ; rendement constant positif ; rendement constant négatif ;
bruit aléatoire sans edge ; signaux aléatoires ; rendements permutés ; edge synthétique injecté ;
configurations dupliquées ; runs dupliqués ; actifs fortement corrélés ; timestamps manquants ;
fenêtres OOS chevauchantes ; frais suffisamment élevés pour annuler un edge apparent.

---

## 17. Schéma relationnel conceptuel

```
study (study_id)
  └──< configuration (configuration_id ; inclut timeframe ; actif exclu)
          └──< experiment (experiment_id ; +asset,timeframe,données,goal,policy,WF ; created_at exclu)
                  └──< run (run_id occurrence ; run_input_fingerprint contenu)
                          ├── window_definition (window_definition_id, stable)
                          │        └── run_window (run_window_id, occurrence dans le run)
                          │                 └── equity_curve  ──dérive──>  returns_series
                          │                      (equity_content_fingerprint)  (returns_content_fingerprint)
                          │                      [valeurs incluses dans les deux fingerprints]
                          └── artifacts : chacun = { artifact_id (logique),
                                                     artifact_content_fingerprint (contenu),
                                                     artifact_path (emplacement modifiable) }
                                          ex. manifest.json, equity, returns, wf_report

returns_series (OOS, non chevauchants, scoped study_id)
        └─> oos_returns_matrix  (comparaisons / diagnostics WF, v0.56)
                └─?─> pbo_input_matrix  (réutilisation / transformation / matrice distincte : à
                                         trancher par la future revue DSR/PBO)  ─> [futur] DSR / PBO
```

---

## 18. Roadmap v0.52 → v0.56

- **v0.52** — primitives d'identité : `study_id`, `configuration_id`, `run_id` + fingerprints
  associés (`run_input_fingerprint`), **sans persistance**. `experiment_id` inchangé.
- **v0.53** — contrat **puis** constructeur **pur** d'**equity périodique** et rendements dérivés
  (selon §9–§10) ; tranche la politique de gaps (§9.2) ; contrôles synthétiques de base.
- **v0.54** — **artefact de run** reliant manifeste, fenêtres et séries périodiques.
- **v0.55** — **registre append-only minimal** enregistrant faits et relations ; le **nombre de
  fichiers** est décidé ici.
- **v0.56** — constructeur **et validations** de l'**`oos_returns_matrix`** (tranche l'identité des
  colonnes, §14). Ne construit **pas** la `pbo_input_matrix`.
- **DSR & PBO** restent **au-delà de v0.56** et exigent une **nouvelle revue d'architecture**, qui
  déterminera la relation entre `oos_returns_matrix` et `pbo_input_matrix`.

---

## 19. Décisions différées

- Formule exacte de `run_id` (catégorie 2) — v0.52.
- Collecte de la provenance code (`code_fingerprint` / git) — non planifiée, hors périmètre.
- Politique exacte de barres manquantes (§9.2) — v0.53.
- Choix par défaut **et validation de compatibilité** de `capital_mode` /
  `boundary_position_policy` (§11.1) — v0.53 (probablement `reset_per_window` + `force_close`, mais
  déclaré/fingerprinté).
- Nombre exact de fichiers du registre (§12) — v0.55.
- **Identité des colonnes de l'`oos_returns_matrix`** (§14) — v0.56.
- **Relation `oos_returns_matrix` ↔ `pbo_input_matrix`** (réutilisation / transformation / matrice
  distincte) — au-delà de v0.56 (revue DSR/PBO).
- Méthode d'agrégation et seuils numériques (DSR/PBO, garde anti-inflation) — au-delà de v0.56.
- Valeur de `n_trials` par étude — **déclarée, justifiée et auditée** humainement, jamais auto.

---

## 20. Rappels méthodologiques permanents

- `run_count != trial_count`
- `configuration_count != effective_trial_count`
- une **réexécution déterministe n'est pas automatiquement un nouvel essai**
- `n_trials` est une **valeur déclarée, justifiée et auditée**, **toujours scoped par `study_id`**,
  jamais déduite du seul nombre de runs
- Walk-Forward OOS **n'est pas** un holdout confirmatoire
- des actifs corrélés ne constituent **pas nécessairement** des preuves indépendantes
- la **traçabilité logicielle ne garantit pas la validité statistique**
