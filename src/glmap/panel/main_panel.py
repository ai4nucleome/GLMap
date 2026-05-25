"""Main biological panel construction (10K probes, Stage 2 of GOAL.md).

ProbeRow schema (11 fields):
    probe_id          str   "human_promoter_0001" — deterministic, panel-wide unique
    sequence          str   uppercase ACGT, length ∈ [128, 1024]
    length_bp         int   len(sequence)
    functional_element str  one of 14 element_ids defined in panel_sources.yaml
    species_group     str   Human / Plant / Fungi / Virus (or "synthetic" for control panel)
                              biological-taxonomy grouping, self-contained on the parquet so
                              downstream consumers do not need to join panel_manifest.json
    species           str   binomial name or "covid_var_3" / "fungi_sp_12" for multi-class
    GC_content        float fraction of G+C bases
    dinuc_vec         list[float]  16-D dinucleotide frequencies (lex order, sum to 1)
    trinuc_vec        list[float]  64-D trinucleotide frequencies (lex order, sum to 1)
    source            str   "{collection}::{dataset}::{split}::row_{idx}" provenance
    label_source      str   how labels were treated; one of:
                              positive_only        — kept rows with label == 1
                              keep_label           — kept rows whose label was in keep_labels
                              keep_all             — used all rows (both labels real DNA)
                              multiclass_species   — label encodes species (kept all, label → species)
                              multi_label_any_pos  — multi-label task, kept rows with ≥1 positive

Allocation source of truth: data/panel_sources.yaml.

Three operations live here:
  - load_panel_config():  parse the yaml into typed dataclasses
  - build_main_panel():   call readers per dataset, sample, assemble ProbeRow rows
  - write_panel_outputs(): emit parquet + manifest + markdown summary

Readers/filters live in src/panel/readers.py; this module only does sampling
and assembly.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .composition import dinuc_vec, gc_fraction, trinuc_vec
from .readers import ReaderResult, read_dataset


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCES_YAML = REPO_ROOT / "data" / "panel_sources.yaml"


# ─────────────────────────── Schema ───────────────────────────

@dataclass(frozen=True)
class ProbeRow:
    probe_id: str
    sequence: str
    length_bp: int
    functional_element: str
    species_group: str
    species: str
    GC_content: float
    dinuc_vec: list[float]
    trinuc_vec: list[float]
    source: str
    label_source: str


# ───────────────────────── Config parsing ─────────────────────────

@dataclass(frozen=True)
class DatasetSpec:
    path: str                    # repo-relative
    format: str                  # csv / fasta_pgb_chromatin / fasta_pgb_binary / fasta_pgb_binary_lenfilter
    seq_col: str | None
    label_col: str | None
    label_policy: str            # positive_only / keep_label / keep_all / multiclass_species
    keep_labels: tuple[str, ...] | None
    target_n: int
    species: str | None
    crop_to: int | None
    crop_strategy: str | None    # "center" only for now
    balance_per_label: int | None


@dataclass(frozen=True)
class ElementSpec:
    element_id: str              # e.g. "promoter", "splice_donor", "chromatin_access"
    species_group: str           # Human / Plant / Yeast / Fungi / Virus
    species: str | None          # binomial or "per_row" / "per_dataset"
    n_probes: int
    datasets: list[DatasetSpec]


@dataclass(frozen=True)
class PanelConfig:
    total_probes: int
    seed: int
    length_min_bp: int
    length_max_bp: int
    species_groups: dict[str, dict[str, Any]]
    elements: list[ElementSpec]


def load_panel_config(yaml_path: Path = DEFAULT_SOURCES_YAML) -> PanelConfig:
    raw = yaml.safe_load(yaml_path.read_text())
    panel = raw["panel"]
    elements: list[ElementSpec] = []
    for elem_id, e in raw["elements"].items():
        ds_list: list[DatasetSpec] = []
        for d in e["datasets"]:
            ds_list.append(DatasetSpec(
                path=d["path"],
                format=d["format"],
                seq_col=d.get("seq_col"),
                label_col=d.get("label_col"),
                label_policy=d.get("label_policy", "keep_all"),
                keep_labels=tuple(d["keep_labels"]) if d.get("keep_labels") else None,
                target_n=d["target_n"],
                species=d.get("species"),
                crop_to=d.get("crop_to"),
                crop_strategy=d.get("crop_strategy"),
                balance_per_label=d.get("balance_per_label"),
            ))
        elements.append(ElementSpec(
            element_id=elem_id,
            species_group=e["species_group"],
            species=e.get("species"),
            n_probes=e["n_probes"],
            datasets=ds_list,
        ))
    return PanelConfig(
        total_probes=panel["total_probes"],
        seed=panel["seed"],
        length_min_bp=panel["length_min_bp"],
        length_max_bp=panel["length_max_bp"],
        species_groups=raw["species_groups"],
        elements=elements,
    )


# ───────────────────────── Build ─────────────────────────

def _crop_center(seq: str, target: int) -> str:
    if len(seq) <= target:
        return seq
    start = (len(seq) - target) // 2
    return seq[start:start + target]


def _is_acgt(seq: str) -> bool:
    return all(c in "ACGT" for c in seq)


def _assign_species(
    elem: ElementSpec, ds: DatasetSpec, raw_label: str
) -> str:
    """Resolve the species field for a probe.

    Priority:
      1. dataset-level species (e.g. each PGB chromatin file)
      2. element-level species
      3. per_row from label (multi-class species tasks)
    """
    if ds.species and ds.species != "per_row":
        return ds.species
    if elem.species and elem.species not in ("per_row", "per_dataset"):
        return elem.species
    # multi-class species: encode label into species id
    if elem.element_id == "fungi_genome":
        return f"fungi_sp_{raw_label}"
    if elem.element_id == "virus_variants":
        return f"covid_var_{raw_label}"
    if elem.element_id == "virus_species":
        return f"virus_sp_{raw_label}"
    return f"{elem.element_id}_label_{raw_label}"


def _sample_records(
    records: list[ReaderResult],
    target_n: int,
    balance_per_label: int | None,
    rng: random.Random,
) -> list[ReaderResult]:
    """Uniform-random or label-balanced sampling.

    When balance_per_label is set (multi-class species tasks), draw exactly
    `balance_per_label` rows from each unique raw_label; if a label has fewer
    rows, take all of them. After balancing, if total < target_n, sample
    more uniformly from the pool to fill; if total > target_n, randomly
    downsample to target_n.
    """
    if not records:
        return []
    if balance_per_label is None:
        if len(records) <= target_n:
            return list(records)
        return rng.sample(records, target_n)

    # Balanced sampling
    by_label: dict[str, list[ReaderResult]] = {}
    for r in records:
        by_label.setdefault(r.raw_label, []).append(r)
    picked: list[ReaderResult] = []
    for label, pool in by_label.items():
        if len(pool) <= balance_per_label:
            picked.extend(pool)
        else:
            picked.extend(rng.sample(pool, balance_per_label))
    # Adjust to target_n
    if len(picked) > target_n:
        picked = rng.sample(picked, target_n)
    elif len(picked) < target_n:
        # Top up with uniform random from rows not yet picked
        chosen_ids = {id(p) for p in picked}
        remaining = [r for r in records if id(r) not in chosen_ids]
        need = target_n - len(picked)
        if len(remaining) >= need:
            picked.extend(rng.sample(remaining, need))
        else:
            picked.extend(remaining)
    return picked


def _make_probe_row(
    elem: ElementSpec,
    ds: DatasetSpec,
    rec: ReaderResult,
    probe_id: str,
    length_min: int,
    length_max: int,
) -> ProbeRow | None:
    seq = rec.sequence.upper()
    if ds.crop_to and len(seq) > ds.crop_to:
        seq = _crop_center(seq, ds.crop_to)
    if not (length_min <= len(seq) <= length_max):
        return None
    if not _is_acgt(seq):
        return None
    return ProbeRow(
        probe_id=probe_id,
        sequence=seq,
        length_bp=len(seq),
        functional_element=elem.element_id,
        species_group=elem.species_group,
        species=_assign_species(elem, ds, rec.raw_label),
        GC_content=gc_fraction(seq),
        dinuc_vec=dinuc_vec(seq),
        trinuc_vec=trinuc_vec(seq),
        source=f"{ds.path}#row_{rec.row_idx}",
        label_source=ds.label_policy,
    )


def build_main_panel(
    cfg: PanelConfig, repo_root: Path = REPO_ROOT
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read every dataset, filter, sample, assemble panel.

    Returns (DataFrame, manifest_dict). Manifest captures per-element
    emitted counts, dataset coverage, and a single seed for reproducibility.
    """
    rng = random.Random(cfg.seed)
    all_rows: list[ProbeRow] = []
    manifest: dict[str, Any] = {
        "total_probes_requested": cfg.total_probes,
        "seed": cfg.seed,
        "length_range": [cfg.length_min_bp, cfg.length_max_bp],
        "elements": {},
    }
    # Stable element order = config insertion order
    for elem in cfg.elements:
        elem_rows: list[ProbeRow] = []
        elem_manifest: dict[str, Any] = {
            "species_group": elem.species_group,
            "n_probes_target": elem.n_probes,
            "datasets": [],
        }
        for ds in elem.datasets:
            ds_path = repo_root / ds.path
            if not ds_path.exists():
                elem_manifest["datasets"].append({
                    "path": ds.path,
                    "status": "MISSING",
                    "target_n": ds.target_n,
                    "emitted_n": 0,
                })
                continue
            records = read_dataset(ds_path, ds)
            # Sub-seed per (element, dataset) for reproducibility
            sub_seed = rng.randint(0, 2**31 - 1)
            sub_rng = random.Random(sub_seed)
            # Datasets that crop the sequence late can still drop rows whose
            # cropped region contains N. Oversample by 25% and trim after
            # validation to hit the exact target.
            oversample_factor = 1.25 if ds.crop_to else 1.0
            initial_target = min(int(ds.target_n * oversample_factor), len(records))
            picked = _sample_records(records, initial_target, ds.balance_per_label, sub_rng)
            ds_emitted = 0
            seen_ids: set[int] = set()
            for rec in picked:
                seen_ids.add(id(rec))
                if ds_emitted >= ds.target_n:
                    break
                probe_id = f"{elem.element_id}_{len(elem_rows) + 1:05d}"
                row = _make_probe_row(elem, ds, rec, probe_id, cfg.length_min_bp, cfg.length_max_bp)
                if row is not None:
                    elem_rows.append(row)
                    ds_emitted += 1
            # Top up if oversample wasn't enough
            if ds_emitted < ds.target_n and len(records) > len(picked):
                remaining = [r for r in records if id(r) not in seen_ids]
                sub_rng.shuffle(remaining)
                for rec in remaining:
                    if ds_emitted >= ds.target_n:
                        break
                    probe_id = f"{elem.element_id}_{len(elem_rows) + 1:05d}"
                    row = _make_probe_row(elem, ds, rec, probe_id, cfg.length_min_bp, cfg.length_max_bp)
                    if row is not None:
                        elem_rows.append(row)
                        ds_emitted += 1
            elem_manifest["datasets"].append({
                "path": ds.path,
                "status": "OK",
                "pool_size": len(records),
                "target_n": ds.target_n,
                "emitted_n": ds_emitted,
                "sub_seed": sub_seed,
            })
        all_rows.extend(elem_rows)
        elem_manifest["n_probes_emitted"] = len(elem_rows)
        manifest["elements"][elem.element_id] = elem_manifest

    df = pd.DataFrame([r.__dict__ for r in all_rows])
    manifest["total_probes_emitted"] = len(all_rows)
    return df, manifest


# ───────────────────────── Output ─────────────────────────

def write_panel_outputs(
    out_dir: Path,
    main_df: pd.DataFrame,
    manifest: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    main_df.to_parquet(out_dir / "main_panel.parquet", index=False)
    with (out_dir / "panel_manifest.json").open("w") as h:
        json.dump(manifest, h, indent=2)
    _write_summary_md(out_dir / "panel_summary.md", main_df, manifest)


def _write_summary_md(path: Path, df: pd.DataFrame, manifest: dict) -> None:
    lines: list[str] = []
    lines.append("# Main panel summary")
    lines.append("")
    lines.append(f"- Total probes emitted: **{manifest['total_probes_emitted']}** "
                 f"(target {manifest['total_probes_requested']})")
    lines.append(f"- Seed: {manifest['seed']}")
    lines.append(f"- Length range: {manifest['length_range'][0]}-{manifest['length_range'][1]} bp")
    lines.append("")
    lines.append("## Per-element breakdown")
    lines.append("")
    rows = []
    for elem_id, e in manifest["elements"].items():
        rows.append({
            "element": elem_id,
            "group": e["species_group"],
            "target": e["n_probes_target"],
            "emitted": e["n_probes_emitted"],
            "datasets": len(e["datasets"]),
        })
    lines.append(pd.DataFrame(rows).to_markdown(index=False))
    lines.append("")
    if not df.empty:
        lines.append("## Length distribution")
        lines.append("")
        ld = df["length_bp"].describe().round(1).to_frame().T
        lines.append(ld.to_markdown(index=False))
        lines.append("")
        lines.append("## Species × functional element cross-tab")
        lines.append("")
        ct = df.groupby(["functional_element", "species"]).size().rename("n").reset_index()
        # Limit width by showing one row per (element, species) pair
        lines.append(ct.to_markdown(index=False))
        lines.append("")
        lines.append("## GC content by element")
        lines.append("")
        gc = df.groupby("functional_element")["GC_content"].agg(["count","mean","min","max"]).round(3)
        lines.append(gc.to_markdown())
    path.write_text("\n".join(lines) + "\n")


__all__ = [
    "ProbeRow",
    "DatasetSpec",
    "ElementSpec",
    "PanelConfig",
    "load_panel_config",
    "build_main_panel",
    "write_panel_outputs",
]
