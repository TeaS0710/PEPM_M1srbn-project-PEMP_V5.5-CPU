# ============================================================================
# Makefile V4 - Pipeline config-first
# ============================================================================
# Idée :
#   - 3 scripts core : prepare / train / evaluate
#   - toute la combinatoire dans les profils YAML + OVERRIDES
#   - le Makefile ne fait que router proprement
#
# Usage typique :
#   make prepare  PROFILE=ideo_quick
#   make train    PROFILE=ideo_quick FAMILY=sklearn
#   make evaluate PROFILE=ideo_quick
#   make pipeline PROFILE=crawl_full
#
#   make check_profile PROFILE=ideo_full
#   make prepare_dry   PROFILE=custom OVERRIDES="corpus_id=web2 train_prop=0.7"
# ============================================================================

# Commande Python (adapter si besoin)
PYTHON ?= python

# Profil par défaut (fichier configs/profiles/$(PROFILE).yml)
PROFILE ?= ideo_quick

# Overrides de config, séparés par des espaces :
#   ex: OVERRIDES="corpus_id=web2 train_prop=0.7 hardware_preset=lab"
OVERRIDES ?=

# Limiter train/evaluate à une seule famille (spacy|sklearn|hf|check) :
#   ex: FAMILY=sklearn
FAMILY ?=

# Pour le script de génération de squelette d'idéologie
CORPUS_XML        ?= data/raw/web1/corpus.xml
IDEO_MAP_OUT      ?= configs/label_maps/ideology_actors.yml
IDEO_REPORT_OUT   ?= data/configs/actors_counts_web1.tsv
MIN_CHARS_IDEO    ?= 200
TOP_VARIANTS_IDEO ?= 5

# PYTHONPATH pour que "scripts" soit importable (scripts.core.core_utils, etc.)
export PYTHONPATH := .

# Conversion OVERRIDES -> "--override key=val" répétés
OVR_FLAGS   = $(foreach o,$(OVERRIDES),--override $(o))
FAMILY_FLAG = $(if $(FAMILY),--only-family $(FAMILY),)

# ============================================================================

.PHONY: help \
        list_profiles \
        check_profile \
        prepare prepare_dry \
        train evaluate pipeline \
        ideology_skeleton

# ============================================================================

help:
	@echo "Pipeline V4 (config-first)"
	@echo ""
	@echo "Variables principales :"
	@echo "  PROFILE   (defaut: ideo_quick)  -> configs/profiles/\$$PROFILE.yml"
	@echo "  OVERRIDES (optionnel)           -> ex: OVERRIDES=\"corpus_id=web2 train_prop=0.7\""
	@echo "  FAMILY    (optionnel)           -> spacy | sklearn | hf | check"
	@echo ""
	@echo "Cibles principales :"
	@echo "  make list_profiles                     # lister les profils disponibles"
	@echo "  make check_profile PROFILE=...         # valider un profil (config-check)"
	@echo "  make prepare       PROFILE=...         # TEI -> TSV (+ formats spacy)"
	@echo "  make prepare_dry    ...                # idem mais dry-run (stats seulement)"
	@echo "  make train         PROFILE=...         # entrainement (tous modèles du profil)"
	@echo "  make train FAMILY=sklearn              # entrainement d'une seule famille"
	@echo "  make evaluate      PROFILE=...         # évaluation (toutes familles du profil)"
	@echo "  make pipeline      PROFILE=...         # prepare + train + evaluate"
	@echo ""
	@echo "Outils pré-analyse :"
	@echo "  make ideology_skeleton                 # construit un YAML squelette ideologie"
	@echo ""
	@echo "Exemples :"
	@echo "  make pipeline PROFILE=ideo_quick"
	@echo "  make train PROFILE=ideo_full FAMILY=hf"
	@echo "  make prepare PROFILE=custom OVERRIDES=\"corpus_id=web2 train_prop=0.7\""

# ============================================================================

list_profiles:
	@echo "Profils disponibles (configs/profiles/*.yml) :"
	@ls configs/profiles/*.yml | sed 's|configs/profiles/||;s|\.yml||'

# Validation de config (profil + modèles + label_map + hardware)
check_profile:
	$(PYTHON) scripts/pre/pre_check_config.py \
		--profile $(PROFILE) \
		$(OVR_FLAGS) \
		--verbose

# ============================================================================

# PREPARE : TEI -> TSV (+ formats par famille, ex: DocBin spaCy)
prepare:
	$(PYTHON) scripts/core/core_prepare.py \
		--profile $(PROFILE) \
		$(OVR_FLAGS) \
		--verbose

# Même chose mais sans écrire les fichiers (stats / meta uniquement)
prepare_dry:
	$(PYTHON) scripts/core/core_prepare.py \
		--profile $(PROFILE) \
		$(OVR_FLAGS) \
		--dry-run \
		--verbose

# ============================================================================

# TRAIN : entraînement multi-familles
train:
	$(PYTHON) scripts/core/core_train.py \
		--profile $(PROFILE) \
		$(OVR_FLAGS) \
		$(FAMILY_FLAG) \
		--verbose

# ============================================================================

# EVALUATE : évaluation multi-familles
evaluate:
	$(PYTHON) scripts/core/core_evaluate.py \
		--profile $(PROFILE) \
		$(OVR_FLAGS) \
		$(FAMILY_FLAG) \
		--verbose

# ============================================================================

# PIPELINE complet : prepare + train + evaluate
pipeline: prepare train evaluate

# ============================================================================

# Génération du squelette d'idéologie à partir du XML
# (adapter les paramètres si l'interface de make_ideology_skeleton.py diffère)
ideology_skeleton:
	$(PYTHON) scripts/pre/make_ideology_skeleton.py \
		--corpus $(CORPUS_XML) \
		--out-yaml $(IDEO_MAP_OUT) \
		--out-report $(IDEO_REPORT_OUT) \
		--min-chars $(MIN_CHARS_IDEO) \
		--top-variants $(TOP_VARIANTS_IDEO)
	@echo "YAML squelette ideologie écrit dans $(IDEO_MAP_OUT)"
	@echo "Rapport d'acteurs écrit dans $(IDEO_REPORT_OUT)"
