# scripts/core/core_train.py

import argparse
import csv
import json
import os
import random
import importlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib

from scripts.core.core_utils import (
    resolve_profile_base,
    debug_print_params,
    PIPELINE_VERSION,
)


# ----------------- CLI -----------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="V4 core_train : entraînement multi-familles (spaCy, sklearn, HF, check)"
    )
    ap.add_argument(
        "--profile",
        required=True,
        help="Nom du profil (sans .yml) dans configs/profiles/",
    )
    ap.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config (clé=valeur, ex: hardware_preset=lab)",
    )
    ap.add_argument(
        "--only-family",
        choices=["spacy", "sklearn", "hf", "check"],
        help="Limiter l'entraînement à une seule famille (optionnel)",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Afficher les paramètres résolus",
    )
    return ap.parse_args()


# ----------------- Utils généraux -----------------


def set_blas_threads(n_threads: int) -> None:
    """
    Limiter les threads BLAS (MKL/OPENBLAS/OMP) pour éviter la sur-souscription.
    """
    if n_threads is None or n_threads <= 0:
        return
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(n_threads)
    print(f"[core_train] BLAS threads fixés à {n_threads}")


def import_string(path: str):
    """
    Import dynamique d'une classe ou fonction à partir d'une chaîne:
    ex: 'sklearn.svm.LinearSVC' -> class.
    """
    module_name, cls_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, cls_name)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_model_output_dir(params: Dict[str, Any], family: str, model_id: str) -> Path:
    corpus_id = params.get("corpus_id", params["corpus"].get("corpus_id", "unknown_corpus"))
    view = params.get("view", "unknown_view")
    return Path("models") / corpus_id / view / family / model_id


def load_tsv_dataset(params: Dict[str, Any]) -> Tuple[List[str], List[str], List[str]]:
    """
    Charger train.tsv et job.tsv depuis data/interim/{corpus_id}/{view}/
    Retourne (train_texts, train_labels, job_texts).
    job_labels ne sont pas strictement nécessaires pour l'entraînement (évent. early stopping),
    on peut les charger plus tard lors de l'évaluation.
    """
    corpus_id = params.get("corpus_id", params["corpus"].get("corpus_id", "unknown_corpus"))
    view = params.get("view", "unknown_view")
    interim_dir = Path("data") / "interim" / corpus_id / view
    train_path = interim_dir / "train.tsv"
    job_path = interim_dir / "job.tsv"

    if not train_path.exists():
        raise SystemExit(f"[core_train] train.tsv introuvable: {train_path}")

    def read_tsv(path: Path) -> Tuple[List[str], List[str]]:
        texts: List[str] = []
        labels: List[str] = []
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                text = row.get("text") or ""
                label = row.get("label")
                if not text or not label:
                    continue
                texts.append(text)
                labels.append(label)
        return texts, labels

    train_texts, train_labels = read_tsv(train_path)

    if job_path.exists():
        job_texts, _job_labels = read_tsv(job_path)
    else:
        print(f"[core_train] WARNING: job.tsv introuvable, on utilisera train comme job pour certains usages.")
        job_texts = train_texts

    return train_texts, train_labels, job_texts


def maybe_debug_subsample(
    texts: List[str],
    labels: List[str],
    params: Dict[str, Any],
) -> Tuple[List[str], List[str]]:
    """
    Si debug_mode=True, limiter la taille du dataset (ex: 1000 docs max).
    """
    if not params.get("debug_mode"):
        return texts, labels

    max_docs = 1000
    if len(texts) <= max_docs:
        return texts, labels

    print(f"[core_train] debug_mode actif : sous-échantillon de {max_docs} docs sur {len(texts)}")
    indices = list(range(len(texts)))
    random.Random(42).shuffle(indices)
    idx_sel = sorted(indices[:max_docs])
    texts_sub = [texts[i] for i in idx_sel]
    labels_sub = [labels[i] for i in idx_sel]
    return texts_sub, labels_sub


def save_meta_model(
    params: Dict[str, Any],
    family: str,
    model_id: str,
    model_dir: Path,
    extra: Dict[str, Any],
) -> None:
    meta = {
        "profile": params.get("profile"),
        "description": params.get("description", ""),
        "corpus_id": params.get("corpus_id", params["corpus"].get("corpus_id")),
        "view": params.get("view"),
        "family": family,
        "model_id": model_id,
        "hardware": params.get("hardware", {}),
        "debug_mode": params.get("debug_mode", False),
        "pipeline_version": PIPELINE_VERSION,
    }
    meta.update(extra)
    meta_path = model_dir / "meta_model.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[core_train] meta_model.json écrit : {meta_path}")


# ----------------- Entraînement spaCy -----------------


def train_spacy_model(params: Dict[str, Any], model_id: str) -> None:
    try:
        import spacy
        from spacy.util import minibatch
        from spacy.tokens import DocBin
    except ImportError:
        raise SystemExit("[core_train] spaCy n'est pas installé, impossible de lancer la famille 'spacy'.")

    models_cfg = params["models_cfg"]["families"]["spacy"][model_id]
    lang = models_cfg.get("lang", "fr")
    epochs = int(models_cfg.get("epochs", 5))
    dropout = float(models_cfg.get("dropout", 0.2))
    arch = models_cfg.get("arch", None)  # "bow" ou "cnn"
    config_template = models_cfg.get("config_template")  # pour V4+

    corpus_id = params.get("corpus_id", params["corpus"].get("corpus_id", "unknown_corpus"))
    view = params.get("view", "unknown_view")

    # Chemins vers éventuels DocBin produits par core_prepare
    spacy_proc_dir = Path("data") / "processed" / corpus_id / view / "spacy"
    train_docbin_path = spacy_proc_dir / "train.spacy"

    nlp = spacy.blank(lang)

    train_data: List[Tuple[str, Dict[str, float]]] = []
    labels_set: List[str] = []

    if train_docbin_path.exists():
        # ---- Cas 1 : on utilise les DocBin créés par core_prepare ----
        print(f"[core_train:spacy] Utilisation de DocBin : {train_docbin_path}")
        db = DocBin().from_disk(train_docbin_path)
        docs = list(db.get_docs(nlp.vocab))

        # Sous-échantillon si debug_mode
        if params.get("debug_mode") and len(docs) > 1000:
            print(f"[core_train:spacy] debug_mode actif : sous-échantillon de 1000 docs sur {len(docs)}")
            docs = docs[:1000]

        labels_set = sorted(
            {lab for doc in docs for lab, val in doc.cats.items() if val}
        )
        for doc in docs:
            # doc.cats est déjà un dict {label: bool}
            train_data.append((doc.text, dict(doc.cats)))
    else:
        # ---- Cas 2 : fallback TSV (compatible avec l'ancien flux V2) ----
        print("[core_train:spacy] Aucun DocBin trouvé, fallback sur train.tsv")
        train_texts, train_labels, _job_texts = load_tsv_dataset(params)
        train_texts, train_labels = maybe_debug_subsample(train_texts, train_labels, params)

        labels_set = sorted(set(train_labels))
        for text, label in zip(train_texts, train_labels):
            cats = {lab: (lab == label) for lab in labels_set}
            train_data.append((text, cats))

    print(f"[core_train:spacy] Modèle={model_id}, labels={labels_set}, n_train_docs={len(train_data)}")

    # Construction du pipe textcat
    if "textcat" not in nlp.pipe_names:
        textcat = nlp.add_pipe("textcat")
    else:
        textcat = nlp.get_pipe("textcat")

    # On part sur un textcat multi-classes exclusif
    textcat.add_label  # type: ignore[attr-defined]
    for label in labels_set:
        textcat.add_label(label)  # type: ignore[call-arg]

    # Initialisation
    optimizer = nlp.begin_training()
    print(f"[core_train:spacy] Entraînement pour {epochs} epochs.")
    for epoch in range(epochs):
        random.shuffle(train_data)
        losses: Dict[str, float] = {}
        batches = minibatch(train_data, size=8)
        for batch in batches:
            texts = [t for t, _ in batch]
            annotations = [{"cats": cats} for _, cats in batch]
            nlp.update(texts, annotations, sgd=optimizer, drop=dropout, losses=losses)
        print(f"[core_train:spacy] Epoch {epoch+1}/{epochs} - pertes={losses}")

    model_dir = get_model_output_dir(params, "spacy", model_id)
    ensure_dir(model_dir)
    nlp.to_disk(model_dir)
    print(f"[core_train:spacy] Modèle spaCy sauvegardé dans {model_dir}")

    save_meta_model(
        params,
        "spacy",
        model_id,
        model_dir,
        extra={
            "epochs": epochs,
            "dropout": dropout,
            "arch": arch,
            "config_template": config_template,
            "labels": labels_set,
            "n_train_docs": len(train_data),
            "train_source": "docbin" if train_docbin_path.exists() else "tsv",
        },
    )



# ----------------- Entraînement sklearn -----------------


def train_sklearn_model(params: Dict[str, Any], model_id: str) -> None:
    models_cfg = params["models_cfg"]["families"]["sklearn"][model_id]
    vect_cfg = models_cfg["vectorizer"]
    est_cfg = models_cfg["estimator"]

    vect_class = import_string(vect_cfg["class"])
    est_class = import_string(est_cfg["class"])

    vect_params = dict(vect_cfg.get("params", {}))
    est_params = dict(est_cfg.get("params", {}))

    # Ajuster n_jobs si possible
    max_procs = params.get("hardware", {}).get("max_procs")
    if max_procs and "n_jobs" in est_params and est_params["n_jobs"] in (None, -1):
        est_params["n_jobs"] = max_procs

    vectorizer = vect_class(**vect_params)
    estimator = est_class(**est_params)

    train_texts, train_labels, job_texts = load_tsv_dataset(params)
    train_texts, train_labels = maybe_debug_subsample(train_texts, train_labels, params)

    print(f"[core_train:sklearn] Modèle={model_id}, {len(train_texts)} docs d'entraînement.")

    X_train = vectorizer.fit_transform(train_texts)
    estimator.fit(X_train, train_labels)

    model_dir = get_model_output_dir(params, "sklearn", model_id)
    ensure_dir(model_dir)
    model_path = model_dir / "model.joblib"
    joblib.dump({"vectorizer": vectorizer, "estimator": estimator}, model_path)
    print(f"[core_train:sklearn] Modèle sklearn sauvegardé dans {model_path}")

    save_meta_model(
        params,
        "sklearn",
        model_id,
        model_dir,
        extra={
            "vectorizer_class": vect_cfg["class"],
            "estimator_class": est_cfg["class"],
            "vectorizer_params": vect_params,
            "estimator_params": est_params,
            "n_train_docs": len(train_texts),
            "n_features": int(getattr(X_train, "shape", (0, 0))[1]),
        },
    )


# ----------------- Entraînement HF (squelette) -----------------


def train_hf_model(params: Dict[str, Any], model_id: str) -> None:
    """
    Squelette pour HuggingFace. À implémenter plus finement plus tard
    (datasets, Trainer, etc.). Pour l'instant : placeholder avec message.
    """
    print(f"[core_train:hf] TODO: entraînement HF pour le modèle '{model_id}' n'est pas encore implémenté.")
    # Quand tu seras prêt :
    # - Charger datasets à partir de train.tsv/job.tsv
    # - Initialiser tokenizer & modèle (AutoTokenizer, AutoModelForSequenceClassification)
    # - Trainer(...) avec training_args
    # - Sauver le modèle + meta_model.json


# ----------------- Entraînement "check" -----------------


def train_check_model(params: Dict[str, Any], model_id: str = "check_default") -> None:
    """
    Famille 'check' vue comme un "pseudo-modèle" :
    il peut générer des stats, des sanity checks, etc., et écrire un meta_model.json.
    Pour l'instant on se contente de consigner les stats de base.
    """
    train_texts, train_labels, job_texts = load_tsv_dataset(params)
    labels_set = sorted(set(train_labels))
    label_counts = {l: train_labels.count(l) for l in labels_set}

    model_dir = get_model_output_dir(params, "check", model_id)
    ensure_dir(model_dir)

    save_meta_model(
        params,
        "check",
        model_id,
        model_dir,
        extra={
            "n_train_docs": len(train_texts),
            "n_labels": len(labels_set),
            "label_counts": label_counts,
            "note": "Famille 'check' = modèle virtuel pour sanity checks / stats",
        },
    )
    print(f"[core_train:check] Checks de base consignés dans {model_dir}")


# ----------------- main -----------------


def main() -> None:
    args = parse_args()
    params = resolve_profile_base(args.profile, args.override)

    if args.verbose:
        debug_print_params(params)

    # Seed de base pour une reproductibilité minimaliste
    random.seed(42)

    hw = params.get("hardware", {})
    blas_threads = hw.get("blas_threads", 1)
    set_blas_threads(blas_threads)

    families = params.get("families", []) or []
    if args.only_family and args.only_family in families:
        families = [args.only_family]

    # Construire la liste des modèles à entraîner
    models_to_train: List[Dict[str, Any]] = []

    if "check" in families:
        # Pour l'instant un seul pseudo-modèle check_default
        models_to_train.append({"family": "check", "model_id": "check_default"})

    if "spacy" in families:
        for mid in params.get("models_spacy", []) or []:
            models_to_train.append({"family": "spacy", "model_id": mid})

    if "sklearn" in families:
        for mid in params.get("models_sklearn", []) or []:
            models_to_train.append({"family": "sklearn", "model_id": mid})

    if "hf" in families:
        for mid in params.get("models_hf", []) or []:
            models_to_train.append({"family": "hf", "model_id": mid})

    if not models_to_train:
        print(f"[core_train] Aucun modèle à entraîner pour le profil '{params.get('profile')}'. Rien à faire.")
        return

    print("[core_train] Modèles à entraîner :")
    for m in models_to_train:
        print(f"  - {m['family']}::{m['model_id']}")

    # Entraînement
    for m in models_to_train:
        family = m["family"]
        mid = m["model_id"]
        if family == "spacy":
            train_spacy_model(params, mid)
        elif family == "sklearn":
            train_sklearn_model(params, mid)
        elif family == "hf":
            train_hf_model(params, mid)
        elif family == "check":
            train_check_model(params, mid)
        else:
            print(f"[core_train] WARNING: famille inconnue '{family}', ignorée.")


if __name__ == "__main__":
    main()
