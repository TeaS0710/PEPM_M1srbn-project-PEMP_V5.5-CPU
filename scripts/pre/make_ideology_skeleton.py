#Projet PEPM By Yi Fan && Adrien
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_ideology_skeleton.py
- Parcourt un teiCorpus XML (TEI) par <TEI> (streaming avec iterparse)
- Récupère le label de crawl (ou folder/folder_path, sinon xml:id)
- Agrège les acteurs uniques (normalisés en snake_case)
- Écrit:
    1) ideology.yml  -> clés = acteurs à annoter manuellement (valeurs vides)
    2) actors_counts.tsv -> stats (#docs, échantillon des libellés bruts)

Usage:
  python3 make_ideology_skeleton.py \
      --corpus data/for_txm/corpus.xml \
      --out-yaml ideology.yml \
      --out-report actors_counts.tsv \
      --min-chars 200 \
      --top-variants 3
"""

from __future__ import annotations
import argparse, re, sys, unicodedata, json
from pathlib import Path
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

def norm_key(s: str) -> str:
    """normalise en snake_case ASCII (accents supprimés)."""
    s = (s or "").lower().strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")

def get_label_from_tei(tei_el: ET.Element) -> str:
    """<term type='crawl'>, sinon folder/folder_path, sinon xml:id."""
    # 1) crawl
    for term in tei_el.findall(".//{*}keywords/{*}term"):
        if (term.attrib.get("type","").lower() == "crawl") and (term.text or "").strip():
            return term.text.strip()
    # 2) folder / folder_path
    for term in tei_el.findall(".//{*}keywords/{*}term"):
        if term.attrib.get("type","").lower() in {"folder","folder_path"} and (term.text or "").strip():
            return term.text.strip()
    # 3) xml:id
    return tei_el.attrib.get("{http://www.w3.org/XML/1998/namespace}id", "").strip()

def text_len_chars(tei_el: ET.Element) -> int:
    """estime la longueur du texte (head + p) pour filtrer les doc trop courts."""
    total = 0
    head = tei_el.find(".//{*}text/{*}body/{*}div/{*}head")
    if head is not None and head.text:
        total += len(head.text)
    for p in tei_el.findall(".//{*}text/{*}body//{*}p"):
        if p.text:
            total += len(p.text)
    return total

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True, help="teiCorpus XML (corpus.xml)")
    ap.add_argument("--out-yaml", type=Path, default=Path("ideology.yml"), help="fichier YAML de sortie")
    ap.add_argument("--out-report", type=Path, default=Path("actors_counts.tsv"), help="rapport TSV (stats)")
    ap.add_argument("--min-chars", type=int, default=0, help="longueur mini pour compter un doc (0=pas de filtre)")
    ap.add_argument("--top-variants", type=int, default=3, help="nb de variantes originales à afficher en commentaire")
    return ap.parse_args()

def main():
    args = parse_args()
    if not args.corpus.exists():
        print(f"[ERR] corpus introuvable: {args.corpus}", file=sys.stderr); sys.exit(1)

    # Agrégations
    # map: norm_label -> Counter(original_label -> count), total_docs
    variants: Dict[str, Counter] = defaultdict(Counter)
    totals: Dict[str, int] = Counter()

    # iterparse (streaming) pour gros corpus
    ctx = ET.iterparse(args.corpus, events=("end",))
    for event, elem in ctx:
        # on ne s'intéresse qu'aux éléments TEI (fin de noeud)
        if elem.tag.endswith("TEI"):
            # filtre longueur optionnel
            if args.min_chars > 0 and text_len_chars(elem) < args.min_chars:
                elem.clear()
                continue

            raw = get_label_from_tei(elem) or ""
            if not raw:
                elem.clear()
                continue

            key = norm_key(raw)
            if not key:
                elem.clear()
                continue

            totals[key] += 1
            variants[key][raw] += 1

            # libérer la mémoire de ce sous-arbre
            elem.clear()

    if not totals:
        print("[WARN] Aucun acteur trouvé (vérifie le XML et --min-chars).")
        return

    # Tri décroissant par #docs
    ordered = sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))

    # 1) YAML squelette
    args.out_yaml.parent.mkdir(parents=True, exist_ok=True)
    with args.out_yaml.open("w", encoding="utf-8") as f:
        f.write("# Remplis chaque valeur avec: gauche / droite (ou autre label si besoin)\n")
        f.write("# Clés normalisées depuis les libellés du corpus. Les commentaires indiquent #docs et exemples.\n")
        for key, n in ordered:
            # variantes les plus fréquentes
            varlist = ", ".join(f"{v}×{c}" for v,c in variants[key].most_common(args.top_variants))
            f.write(f"{key}: ''  # {n} docs | ex: {varlist}\n")

    # 2) Rapport TSV (clé, total_docs, variantes_json)
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    with args.out_report.open("w", encoding="utf-8") as f:
        f.write("key\ttotal_docs\tvariants_json\n")
        for key, n in ordered:
            js = json.dumps(variants[key], ensure_ascii=False)
            f.write(f"{key}\t{n}\t{js}\n")

    print(f"[OK] YAML squelette → {args.out_yaml}")
    print(f"[OK] Rapport TSV   → {args.out_report}")
    print(f"[INFO] Acteurs uniques: {len(ordered)}  |  Total docs: {sum(totals.values())}")

if __name__ == "__main__":
    main()
