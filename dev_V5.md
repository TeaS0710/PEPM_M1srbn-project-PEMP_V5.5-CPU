# Documentation de développement – V5

**PEPM_M1srbn-project-PEMP_V4.x → V5 (core + orchestrateur)**

> Objectif : décrire précisément les **améliorations V5** envisagées au-dessus du core V4.x, leurs **hypothèses d’usage**, et leur **réalité technique** (config, structures, pseudo-code).
> Cette doc est destinée à être **implémentable** par un·e dev sans conversation annexe.

---

## 0. Contexte et vision

### 0.1. État actuel (V4.05)

Le projet en V4.05 repose sur :

* un **core** modulaire :

  * `core_prepare.py` : TEI → TSV + formats spaCy / matrices, split train/job, équilibrage ;
  * `core_train.py` : entraînement des familles de modèles (`spacy`, `sklearn`, `hf`) ;
  * `core_evaluate.py` : évaluation, métriques globales, rapports ;
  * `core_utils.py` : résolution des profils, configs, chemins.
* une **interface Makefile** :

  * `make run STAGE=... PROFILE=...` (stages : `check`, `prepare`, `train`, `evaluate`, `pipeline`).
* une **couche config** :

  * `configs/profiles/*.yml` : profils d’analyses (politique, idéologie, quick vs full),
  * `configs/common/*.yml` : `corpora.yml`, `hardware.yml`, `balance.yml`, `models.yml`,
  * `configs/label_maps/*.yml` : maps d’idéologie, acteurs, etc.
* un **pipeline actuellement mono-dataset** (un `corpus_id` par profil),
* une utilisation **manuelle** : sélection d’un profil, lancement de quelques commandes, extraction ponctuelle de métriques.

### 0.2. Objectif global V5

V5 vise deux grands axes :

1. **Achever / stabiliser la couche “multi-corpus + analyses comparatives” (V4.2)**
   → possibilité de combiner `web1`, `asr1`, `web2`, etc., dans un même profil, avec des métriques globales et par sous-groupe (`corpus_id`, `modality`, etc.).

2. **Ajouter une brique “orchestrateur expérimental” au-dessus du core (V5)** :

   * générer **automatiquement** des expériences (grilles de paramètres, axes d’analyse),
   * **piloter** le core (`prepare`, `train`, `evaluate`) via un scheduler,
   * **contrôler la RAM** et éviter les crashs (surtout pour spaCy / HF),
   * faciliter les **analyses multi-configurations** (courbes d’apprentissage, effets de l’équilibrage, cross-dataset, etc.).

L’idée : passer d’un pipeline utilisable “une config à la fois” à une **plateforme d’expérimentation quasi automatique**, mais **RAM-safe** pour une machine de travail classique.

---

## 1. Extension V4.2 – Multi-corpus & analyses juxtaposées

> Cette section synthétise les spécifications V4.2 (déjà discutées) en les intégrant dans la doc V5, car l’orchestrateur va exploiter directement ces concepts.

### 1.1. Concepts clés

* **dataset_id** :
  identifiant logique du “dataset d’analyse” = unité de sortie (répertoires `data/interim`, `data/processed`, `models`, `reports`).
  Exemples : `web1`, `asr1`, `web1_asr1`.

* **corpus_ids** :
  liste des corpus sources définis dans `configs/common/corpora.yml`.
  Ex : `[web1]`, `[web1, asr1]`, `[web2, asr2]`.

* **merge_mode** :

  * `"single"` : un seul corpus (comportement V4 actuel),
  * `"merged"` : combinaison de plusieurs corpus → un dataset fusionné + métriques globales,
  * `"juxtaposed"` : fusion + métriques **par groupe** (corpus, modality…),
  * `"separate"` : chaque corpus traité séparément (optionnel / futur via orchestrateur).

* **source_field** :
  nom de la colonne / clé meta qui identifie la source dans les TSV et meta (`corpus_id`, `source`, etc.).

* **analysis.compare_by** :
  liste de champs sur lesquels l’évaluation doit produire des métriques groupées (`metrics_by_<field>.json`).
  Ex : `["corpus_id"]`, `["corpus_id", "modality"]`.

### 1.2. Profils (`configs/profiles/*.yml`)

Nouveaux champs :

```yaml
profile: ideo_quick_web1_asr1
description: >
  Fusion web1+asr1, vue ideology_global, analyse juxtaposée.

dataset_id: web1_asr1    # ID logique de l’ensemble d’analyse

data:
  corpus_ids: [web1, asr1]        # sources (corpora.yml)
  merge_mode: juxtaposed          # single | merged | juxtaposed | separate
  source_field: corpus_id         # champ meta/TSV pour marquer la source

view: ideology_global

analysis:
  compare_by:
    - corpus_id                   # générer metrics_by_corpus_id.json
    # - modality                  # extension possible

# Reste : ideology, actors, families, models_*, hardware_preset, etc.
```

**Compatibilité :**

* Profil V4 “simple” (un seul corpus) :

  * pas de `data.corpus_ids`,
  * `corpus_id` est obligatoire,
  * `merge_mode` implicite = `"single"`.

### 1.3. `corpora.yml`

Toujours une entrée par corpus :

```yaml
web1:
  corpus_id: web1
  corpus_path: data/raw/web1/corpus.xml
  language: fr
  default_modality: web

asr1:
  corpus_id: asr1
  corpus_path: data/raw/asr1/corpus_asr.xml
  language: fr
  default_modality: asr
```

La fusion `web1_asr1` n’apparaît **pas** ici : elle est décrite au niveau **profil**.

### 1.4. Résolution de profil (`core_utils.py`)

Fonction : `resolve_profile_base(profile_name, overrides) -> Dict[str, Any]`

* lit :

  * `data.corpus_ids`, `data.merge_mode`, `data.source_field`,
  * `analysis.compare_by`,
  * `dataset_id`.
* construit :

```python
params["corpora"]      # liste de corpus_cfg (1+ éléments)
params["corpus"]       # le premier corpus (pour compat)
params["corpus_id"]    # = dataset_id (logique)
params["dataset_id"]   # ID logique final
params["merge_mode"]   # single / merged / juxtaposed / separate
params["source_field"] # ex : "corpus_id"
params["analysis"]     # bloc analysis du profil
```

### 1.5. `core_prepare.py` – multi-TEI + dataset_id

Fonction : `build_view(params, dry_run=False)`

* utilise `dataset_id` pour les chemins :

  ```python
  dataset_id = params.get("dataset_id") or params.get("corpus_id")
  view = params.get("view", "unknown_view")

  interim_dir   = data/interim/<dataset_id>/<view>/
  processed_dir = data/processed/<dataset_id>/<view>/
  ```

* lit plusieurs TEI si `params["corpora"]` a plusieurs éléments :

  ```python
  for corpus_cfg in params["corpora"]:
      tei_path = corpus_cfg["corpus_path"]
      corpus_id_src = corpus_cfg["corpus_id"]

      for elem in iter_tei_docs(tei_path, params):
          ...
          row_meta[source_field] = corpus_id_src
          docs.append({...})
  ```

* écrit `train.tsv` / `job.tsv` avec une colonne supplémentaire `source_field` :

  Colonnes minimales :

  * `id`, `label`, `label_raw`, `modality`, `<source_field>`, `text`.

* `meta_view.json` inclut :

  * `dataset_id`,
  * `source_field`,
  * `source_corpora` = liste des corpus sources.

### 1.6. `core_evaluate.py` – métriques groupées

Fonctionnalités :

* lecture de `job.tsv` avec toutes les colonnes (`label`, `text`, `modality`, `<source_field>`, etc.) ;
* calcul de métriques globales (comme aujourd’hui) ;
* si `analysis.compare_by` non vide :

  * groupby sur les champs demandés (`corpus_id`, `modality`, …),
  * pour chaque groupe : `compute_basic_metrics(y_true_group, y_pred_group)`,
  * écriture de fichiers :

    * `metrics_by_corpus_id.json`,
    * `metrics_by_modality.json`, etc.

Ces fichiers sont rangés dans :

```text
reports/<dataset_id>/<view>/<family>/<model_id>/metrics_by_<field>.json
```

**Exemple** pour `compare_by: [corpus_id]` :

```json
{
  "web1": { "accuracy": 0.82, "macro_f1": 0.80, ... },
  "asr1": { "accuracy": 0.75, "macro_f1": 0.72, ... }
}
```

---

## 2. Contrainte critique : saturation RAM & hardware

### 2.1. Problème rencontré

* **spaCy** est particulièrement gourmand (DocBin, embeddings, optimiseur…),
* des runs “full corpus” (3 Go de TEI) entraînent :

  * des pics de RAM,
  * des freezes ou OOM (Out Of Memory),
* sklearn / HF peuvent aussi consommer beaucoup.

Conclusion :
On ne peut plus se contenter d’**enchaîner des runs dans le même process Python** sans contrôle.

### 2.2. Solutions V4.x + V5

* V4.x : utiliser `hardware.yml` pour fixer :

  * `max_train_docs_spacy`,
  * `max_train_docs_sklearn`,
  * `max_train_docs_hf`,
  * `blas_threads`, `max_procs`.

* V5 : ajouter une brique **externes** de contrôle :

  * 1 run = 1 process isolé (`subprocess`),
  * scheduler qui limite le nombre de runs actifs,
  * optionnel : surveillance RAM en temps réel (via `psutil`),
  * strategies de backoff / early-stop en cas de OOM.

---

## 3. Brique d’orchestration V5 – `exp_orchestrator`

### 3.1. But

La brique `exp_orchestrator` :

* lit une **config d’expérience** (YAML) → `exp_config`,
* génère un **plan d’expériences** (liste de runs = combinaisons de paramètres),
* exécute ces runs en :

  * appelant le core via `make run` dans des processus séparés,
  * contrôlant le **parallélisme** et la **RAM**,
* trace chaque run (`runs.tsv`, logs),
* déclenche des **hooks d’analyse** à la fin (agrégats, graphes, rapports).

### 3.2. Structure de fichiers

* `scripts/experiments/exp_orchestrator.py`
* `scripts/experiments/run_single.py` (exécuteur d’un run individuel)
* `configs/experiments/*.yml` (config d’expériences)
* `experiments/<exp_id>/`

  * `plan.tsv` : plan complet des runs prévus
  * `runs.tsv` : statut et méta de chaque run
  * `logs/run_<run_id>.log`
  * `metrics_global.tsv`, `metrics_by_*.tsv` (agrégats)
  * `plots/`, `report.md`, etc. (hooks d’analyse)

### 3.3. Config d’expérience (`configs/experiments/*.yml`)

#### 3.3.1. Structure générique

```yaml
exp_id: ideo_balancing_sweep
description: >
  Étude de l'impact des stratégies d'équilibrage
  sur web1 et web1+asr1, avec ideo_quick.

base:
  profile: ideo_quick
  stage: pipeline
  fixed:
    CORPUS_ID: web1
    HARDWARE_PRESET: small
    TRAIN_PROP: 0.8
  overrides:
    ideology.view: global_five

axes:
  - name: dataset
    type: choice
    values:
      - label: web1_only
        overrides:
          data.corpus_ids: [web1]
          data.merge_mode: single
      - label: web1_asr1_juxt
        overrides:
          data.corpus_ids: [web1, asr1]
          data.merge_mode: juxtaposed
          analysis.compare_by: [corpus_id]

  - name: balance_strategy
    type: choice
    values:
      - label: no_balance
        make_vars:
          BALANCE_STRATEGY: none
      - label: oversample_parity
        make_vars:
          BALANCE_STRATEGY: oversample
          BALANCE_PRESET: parity

  - name: family_model
    type: choice
    values:
      - label: sklearn_svm
        make_vars:
          FAMILY: sklearn
        overrides:
          families: [sklearn]
          models_sklearn: [tfidf_svm_quick]
          hardware.max_train_docs_sklearn: 100000

      - label: spacy_cnn
        make_vars:
          FAMILY: spacy
        overrides:
          families: [spacy]
          models_spacy: [spacy_cnn_quick]
          hardware.max_train_docs_spacy: 20000

grid:
  mode: cartesian

run:
  repeats: 1
  seed_strategy: per_run   # fixed | per_run
  base_seed: 42

scheduler:
  parallel: 1              # 1 = séquentiel
  max_ram_gb: 14
  resource_classes:
    spacy: heavy
    sklearn: light
    hf: heavy
  weights:
    light: 1
    medium: 2
    heavy: 4
  max_weight: 4

analysis_hooks:
  after_experiment:
    - type: curves
      metrics: [accuracy, macro_f1]
      x_axis: TRAIN_PROP
      group_by: [family, dataset_id]
    - type: report_markdown
      path: experiments/${exp_id}/report.md
```

#### 3.3.2. Sémantique

* `base.profile` : nom du profil core (`configs/profiles/<profile>.yml`).

* `base.stage` : stage par défaut (`pipeline`, `train`, `evaluate`, …).

* `base.fixed` : variables Make toujours présentes (`PROFILE` sera surchargé par `base.profile`).

* `base.overrides` : overrides logiques appliqués à tous les runs.

* `axes[*].values[*].make_vars` :

  * variables Make spécifiques à cette valeur d’axe,
  * ex : `FAMILY=spacy`, `BALANCE_STRATEGY=oversample`.

* `axes[*].values[*].overrides` :

  * overrides logiques spécifiques (clé=val pour `OVERRIDES`).

* `grid.mode: cartesian` :

  * produit cartésien des axes → toutes les combinaisons.

* `run.repeats` :

  * nombre de répétitions (pour variance / stabilité).

* `seed_strategy` :

  * `fixed` : même `SEED` pour tous les runs,
  * `per_run` : `SEED = base_seed + run_index`.

* `scheduler` :

  * `parallel` : max de runs simultanés,
  * `max_ram_gb` : budget RAM global,
  * `resource_classes`, `weights`, `max_weight` : heuristique pour limiter la charge.

---

## 4. Structures internes V5

### 4.1. `RunSpec` (représentation d’un run)

```python
@dataclass
class RunSpec:
    run_id: str           # ex: "exp1_run_000123"
    exp_id: str
    profile: str
    stage: str            # "pipeline", "train", "evaluate", ...
    make_vars: Dict[str, str]   # PROFILE, STAGE, CORPUS_ID, FAMILY, etc.
    overrides: Dict[str, Any]   # cle=val pour OVERRIDES
    repeat_index: int
    axis_values: Dict[str, str] # { "dataset": "web1_asr1_juxt", ... }
    resource_class: str         # "light" / "medium" / "heavy"
```

### 4.2. `ExpConfig` (config d’expérience parsée)

```python
@dataclass
class ExpConfig:
    exp_id: str
    description: str
    base_profile: str
    base_stage: str
    base_make_vars: Dict[str, str]
    base_overrides: Dict[str, Any]
    axes: List[AxisConfig]
    grid_mode: str
    repeats: int
    seed_strategy: str
    base_seed: int
    scheduler: SchedulerConfig
    analysis_hooks: AnalysisHooksConfig
```

---

## 5. Exécution d’un run – `run_single.py`

### 5.1. Interface

CLI type :

```bash
python -m scripts.experiments.run_single \
  --exp-id ideo_balancing_sweep \
  --run-id exp1_run_0001 \
  --profile ideo_quick \
  --stage pipeline \
  --make-var CORPUS_ID=web1 \
  --make-var FAMILY=spacy \
  --override ideology.view=global_five \
  --override data.corpus_ids=[web1,asr1] \
  --max-ram-mb 12000 \
  --log-path experiments/ideo_balancing_sweep/logs/run_exp1_run_0001.log
```

### 5.2. Fonctionnement

1. Construire la commande `make run ...` :

   * `PROFILE` = `--profile`,
   * `STAGE` = `--stage`,
   * le reste des `--make-var` → `VAR=VAL` dans la commande,
   * `--override` → concaténés dans une chaîne `OVERRIDES="k1=v1 k2=v2 ..."`.

   Exemple :

   ```bash
   make run STAGE=pipeline PROFILE=ideo_quick \
     CORPUS_ID=web1 FAMILY=spacy \
     OVERRIDES="ideology.view=global_five data.corpus_ids=[web1,asr1]"
   ```

2. Lancer la commande via `subprocess.Popen` & redirection des logs vers `log_path`.

3. Si `--max-ram-mb` défini :

   * utiliser `psutil` pour surveiller le RSS du process enfant,
   * toutes les X secondes :

     * si `rss > max-ram-mb` → kill process, retourner un code spécifique / status `"OOM_KILLED"`.

4. Retourner (via `exit code` et/ou fichier JSON) :

   * `status` (`success`, `failed`, `oom`),
   * `return_code`,
   * temps d’exécution,
   * max RSS observé (optionnel).

---

## 6. Scheduler & exécution multi-runs – `exp_orchestrator.py`

### 6.1. CLI

```bash
python -m scripts.experiments.exp_orchestrator \
  --exp-config configs/experiments/ideo_balancing_sweep.yml \
  --parallel 2 \
  --max-ram-gb 14 \
  --max-runs 100 \
  --resume \
  --dry-run
```

Options :

* `--exp-config` : chemin vers le YAML,
* `--parallel` : nombre max de runs simultanés (par défaut : 1),
* `--max-ram-gb` : budget RAM global,
* `--max-runs` : limite pratique pour debug / découpe,
* `--resume` : ne relance pas les runs déjà marqués `success` dans `runs.tsv`,
* `--dry-run` : génère le plan, affiche ce qui serait lancé mais n’exécute rien.

### 6.2. Étapes

1. **Chargement exp_config**.
2. **Génération du plan** :

   * produit cartésien des axes (`grid.mode=cartesian`),
   * construction d’un `RunSpec` par combinaison × répétition,
   * calcul `resource_class` selon `family` + règles,
   * assignation d’un `run_id` unique.
3. **Écriture de `plan.tsv`**.
4. **Boucle de scheduling** (voir pseudo-code dans le message précédent) :

   * maintient :

     * `pending` : liste de `RunSpec` à lancer,
     * `active` : dict `run_id -> ProcessHandle`,
     * `completed` : dict `run_id -> status`.
   * lance des `run_single` en respectant :

     * `parallel`,
     * `scheduler.resource_classes`, `weights`, `max_weight`,
     * éventuellement, la RAM disponible réelle.
   * met à jour `runs.tsv` au fur et à mesure.

### 6.3. `runs.tsv`

Colonnes typiques :

```text
run_id  exp_id  profile stage status return_code
family  model_id  corpus_id  dataset_id  view
axis_values_json  make_vars_json  overrides_json
metrics_path  metrics_by_corpus_path  log_path
started_at  finished_at  duration_s  max_rss_mb
```

---

## 7. Expériences cross-dataset (train sur A, eval sur B)

### 7.1. Intuition

On veut :

* **entraîner** sur un dataset `A` (`web1`),
* **évaluer** sur un dataset `B` (`asr1`) avec les **modèles entraînés sur A**.

Intérêt :
tester la **généralisation** des modèles à des données de nature différente (texte web vs ASR, etc.).

### 7.2. Extension de config (exp_config)

On peut décrire un axe “dataset_pair” :

```yaml
axes:
  - name: dataset_pair
    type: choice
    values:
      - label: train_web1_eval_web1
        make_vars:
          CORPUS_ID: web1
        overrides:
          dataset_id: web1
          cross_dataset.train_on: web1
          cross_dataset.eval_on: web1

      - label: train_web1_eval_asr1
        make_vars:
          CORPUS_ID: web1
        overrides:
          dataset_id: web1
          cross_dataset.train_on: web1
          cross_dataset.eval_on: asr1
```

### 7.3. Fonctionnement côté orchestrateur

Pour chaque `RunSpec` avec `cross_dataset.train_on` / `cross_dataset.eval_on` :

* **décomposer** en deux runs logiques :

  1. `run_type=train` : STAGE=`train` sur `train_on`,
  2. `run_type=cross_eval` : STAGE=`evaluate` sur `eval_on`, mais en chargeant les modèles de `train_on`.

Cela peut être implémenté de façon :

* soit interne au core (avec un param `eval_dataset_id`),
* soit externe :

  * `exp_orchestrator` :

    * lance d’abord un pipeline “normal” sur `train_on`,
    * puis lance des runs `evaluate` en ajustant des overrides.

### 7.4. Extension possible dans `core_evaluate.py`

Pour simplifier, on peut prévoir :

* un override `eval_dataset_id` :

  ```python
  dataset_id_for_models = params["dataset_id"]        # où chercher les modèles
  dataset_id_for_eval   = params.get("eval_dataset_id", params["dataset_id"])
  ```

* chemins :

  * modèles : `models/<dataset_id_for_models>/<view>/...`
  * job.tsv : `data/interim/<dataset_id_for_eval>/<view>/job.tsv`

L’orchestrateur devra alors :

* faire en sorte que :

  * un run `train` crée les modèles sur `dataset_id=train_on`,
  * un run `cross_eval` lance l’éval avec `dataset_id=train_on` mais `eval_dataset_id=eval_on`.

---

## 8. Critères d’arrêt intelligents (quality & sécurité)

### 8.1. Critères de qualité modèle

Bloc optionnel dans `exp_config.yml` :

```yaml
early_stop:
  enabled: true
  min_accuracy: 0.30
  min_macro_f1: 0.25
  apply_to_families: [spacy, hf]
```

Fonctionnement :

* Après un run `train+evaluate` :

  * l’orchestrateur lit `metrics.json`,
  * si `accuracy < min_accuracy` ET `macro_f1 < min_macro_f1` :

    * loggue un warning,
    * éventuellement réduit les répétitions futures pour cette config,
    * ou marque certains runs futurs comme “skippables” (V2+).

Version minimale V5 :

* **log only** : on n’annule rien automatiquement, mais on stocke les infos dans `runs.tsv`.
* Un script d’analyse externe peut ensuite décider d’abandonner certaines branches.

### 8.2. Politique OOM

Bloc optionnel :

```yaml
oom_policy:
  on_oom: "backoff"  # "skip" | "backoff" | "stop"
  backoff_factor: 0.5
```

* Si un run est marqué `status=OOM_KILLED` :

  * `"skip"` : on n’essaie plus cette combinaison (l’orchestrateur marque la config comme impossible).
  * `"backoff"` : on modifie dynamiquement la config pour les runs suivants :

    * ex : `max_train_docs_spacy = max_train_docs_spacy * backoff_factor`,
    * et on relance une config plus légère.
  * `"stop"` : on arrête l’expérience (sécurité max).

Implementation V5 :
on peut commencer par log-only + skip, `backoff` à prévoir pour plus tard (nécessite un mécanisme de mutation dynamique de la config).

---

## 9. Hooks d’analyse & reporting

### 9.1. Types de hooks

Déclarés dans `analysis_hooks` :

```yaml
analysis_hooks:
  after_experiment:
    - type: curves
      metrics: [accuracy, macro_f1]
      x_axis: TRAIN_PROP
      group_by: [family, dataset_id]

    - type: report_markdown
      path: experiments/${exp_id}/report.md
```

Hooks à implémenter :

1. **curves** :

   * lit `runs.tsv` + `metrics.json`,
   * récupère :

     * `x_axis` (ex : `TRAIN_PROP`, `hardware.max_train_docs_spacy`),
     * `metrics` (accuracy, macro_f1, …),
     * `group_by` (ex : `family`, `dataset_id`),
   * construit des datasets pour `matplotlib`,
   * produit des PNG dans `experiments/<exp_id>/plots/`.

2. **report_markdown** :

   * compile les infos de `runs.tsv` + métriques,
   * génère un fichier Markdown avec :

     * résumé,
     * meilleurs runs par famille/dataset,
     * liens vers les graphes,
     * éventuellement des tableaux (top-N configs).

### 9.2. Pipeline hooks côté orchestrateur

* Après la boucle de runs :

  ```python
  if exp_config.analysis_hooks.after_experiment:
      run_analysis_hooks(exp_config, runs_metadata)
  ```

* `runs_metadata` = structure dérivée de `runs.tsv` (parsing).

---

## 10. Roadmap d’implémentation V5

1. **V5.0 – Finalisation V4.2 (multi-corpus)**

   * S’assurer que :

     * `params["corpora"]`, `dataset_id`, `merge_mode`, `source_field`, `analysis.compare_by` sont bien implémentés,
     * `core_prepare` écrit correctement les TSV multi-corpus,
     * `core_evaluate` génère `metrics_by_<field>.json`.

2. **V5.1 – Orchestrateur séquentiel sans RAM monitoring**

   * Implémenter :

     * parsing de `exp_config.yml`,
     * génération de `plan.tsv`,
     * exécution séquentielle des runs via `subprocess.make run`,
     * `runs.tsv` + logs.

3. **V5.2 – Orchestrateur parallèle simple**

   * Ajouter `scheduler.parallel`,
   * implémenter `resource_class` + `weights` + `max_weight`,
   * interdire plusieurs runs “heavy” en parallèle.

4. **V5.3 – Monitoring RAM & OOM policy**

   * Implémenter `run_single` avec `psutil` + `--max-ram-mb`,
   * retourner `status=OOM_KILLED`,
   * orchestrateur prend en compte `oom_policy`.

5. **V5.4 – Hooks d’analyse de base**

   * Générer `metrics_global.tsv` pour une expérience,
   * implémenter hook `curves` + `report_markdown`.

6. **V5.5 – Expériences cross-dataset**

   * Étendre le core (`core_evaluate`) avec `eval_dataset_id` (ou équivalent),
   * ajuster orchestrateur pour décomposer les runs train / cross-eval.

---

Ce document `dev_V5.md` doit servir de **référence de développement** :

* chaque bloc est conçu pour être implémenté de façon incrémentale,
* les interfaces (config YAML, structures Python, fichiers générés) sont explicitement décrites,
* l’orchestrateur + scheduler sont pensés pour **enfin exploiter à fond** toute la richesse de paramètres du core, sans faire exploser ta RAM.
