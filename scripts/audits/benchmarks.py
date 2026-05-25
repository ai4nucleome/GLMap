#!/usr/bin/env python3
"""Benchmark-side audit.

Walks the four benchmark roots under `data/` and produces:

  - `data/audits/benchmarks.json` — per-task records + per-collection summaries
  - `data/audits/benchmarks.md`   — human-readable report

Per-task record schema:

    {
      "collection":    "gue",                       # source collection
      "task_group":    "EMP",                       # first path segment
      "dataset_name":  "EMP/H3K4me3",
      "split":         "train",
      "file":          "data/GUE/EMP/H3K4me3/train.csv",
      "format":        "csv",                       # csv | tsv | fasta
      "n_records":     32000,
      "n_labels":      2,
      "task_type":     "binary",                    # binary | multiclass | regression | unlabeled | unknown
      "len_min":       70,
      "len_q25":       500,
      "len_median":    500,
      "len_q75":       500,
      "len_q95":       500,
      "len_max":       500,
      "mean_len":      500.0,
      "top_labels":    "0:16000;1:16000"
    }

Usage:
    $PY scripts/audits/benchmarks.py
    $PY scripts/audits/benchmarks.py --collections gue,pgb     # subset
    $PY scripts/audits/benchmarks.py --max-files 5             # debug
"""

from __future__ import annotations

import argparse
import csv
import gzip
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from common import (
    AUDIT_DIR,
    BENCHMARK_ROOTS,
    percentile_from_counts,
    write_json,
    write_markdown,
)

# Bench files like DFB's `causal/*/neg_seqs_long.csv` and
# `pathogenic/seqs_pathogenic_long.csv` carry the entire DNA sequence in a
# single CSV field (sometimes >300 kB per row), which exceeds Python's
# default csv field-size limit of 131072 bytes. Without lifting the limit
# the reader silently raises csv.Error and the file shows up as
# n_records=0 / task_type="unknown" in the audit.
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)


# ---------------------------------------------------------------------------
# File walking
# ---------------------------------------------------------------------------

DATA_SUFFIXES = (".csv", ".tsv", ".fa", ".fasta", ".fna")
GZIP_SUFFIXES = tuple(s + ".gz" for s in DATA_SUFFIXES)
ALL_SUFFIXES = DATA_SUFFIXES + GZIP_SUFFIXES

# Path segments / filenames to skip — non-data artifacts that ship in the
# benchmark folders (figures, README files, OS junk).
SKIP_DIRS = {"ISM_Tables", "Figures", ".git", "__pycache__"}
SKIP_FILE_GLOBS = (
    "*.DS_Store",
    "README*",
    "readme*",
    "*.png",
    "*.pdf",
    "*.svg",
    "*.json",
    "*.md",
    "*.ipynb",
)

# Per-collection blacklist of files that pass the format check but shouldn't
# count as probes. Path is relative to the collection root.
_SKIP_REL_PATHS: dict[str, set[str]] = {
    # TAD/hg38.ml.fa is the entire hg38 reference (23 records, one per
    # chromosome, max record = 248,956,422 bp). DFB bundles it as background
    # context for the TAD task; it is never used as a scoring probe.
    "dna_foundation_benchmark": {"TAD/hg38.ml.fa"},
}


def _is_data_file(path: Path) -> bool:
    name = path.name.lower()
    for ext in ALL_SUFFIXES:
        if name.endswith(ext):
            for glob in SKIP_FILE_GLOBS:
                if path.match(glob):
                    return False
            return True
    return False


def walk_collection(root: Path) -> Iterator[Path]:
    """Yield every data file under `root`, skipping non-data subdirs.

    Dedups paired `<name>.csv` + `<name>.csv.gz`: when both exist (as in
    `data/genomic-benchmarks/`), prefer the uncompressed copy.
    """
    if not root.exists():
        return
    candidates: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if _is_data_file(path):
            candidates.append(path)

    uncompressed = {str(p) for p in candidates if not p.name.endswith(".gz")}
    for p in candidates:
        if p.name.endswith(".gz"):
            sibling = str(p.with_name(p.name[: -len(".gz")]))
            if sibling in uncompressed:
                continue
        yield p


# ---------------------------------------------------------------------------
# Dataset / split naming
# ---------------------------------------------------------------------------

_SPLIT_KEYWORDS = ("train", "test", "dev", "val", "valid", "validation")
# Suffixes used in PGB-style filenames: `<species>_<split>.fa`.
_FILENAME_SPLIT_RE = re.compile(
    r"^(.*)_(" + "|".join(_SPLIT_KEYWORDS) + r")$",
    flags=re.IGNORECASE,
)


def _file_stem(path: Path) -> str:
    """Strip .csv / .csv.gz / .fa.gz / etc."""
    name = path.name
    for suffix in ALL_SUFFIXES:
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _file_format(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".fa") or name.endswith(".fa.gz"):
        return "fasta"
    if name.endswith(".fasta") or name.endswith(".fasta.gz"):
        return "fasta"
    if name.endswith(".fna") or name.endswith(".fna.gz"):
        return "fasta"
    if name.endswith(".tsv") or name.endswith(".tsv.gz"):
        return "tsv"
    return "csv"


def derive_dataset_split(rel_path: Path) -> tuple[str, str]:
    """Resolve (dataset_name, split) from a path relative to the collection root.

    Rules (first match wins):
      - `<dataset>/<split>/<class>.csv`   -> dataset=<dataset>, split=<split>
        (genomic-benchmarks layout: class label lives in the filename, scan_collection
        later picks it up via the filename-as-label fallback)
      - `<group>/<dataset>/<split>.csv`   -> dataset=<group>/<dataset>, split=<split>
      - `<group>/<name>_<split>.fa`       -> dataset=<group>/<name>,    split=<split>
      - `<group>/<name>.csv`              -> dataset=<group>/<name>,    split=""
      - `<name>.csv`                      -> dataset=<name>,            split=""
    """
    parts = list(rel_path.parts)
    stem = _file_stem(rel_path)
    if len(parts) >= 3:
        # 3-level layout: parent-of-parent dir is the dataset, parent is the
        # split, file stem is the class label (genomic-benchmarks pattern).
        if (
            parts[-2].lower() in _SPLIT_KEYWORDS
            and stem.lower() not in _SPLIT_KEYWORDS
        ):
            return "/".join(parts[:-2]), parts[-2].lower()
    if len(parts) >= 2:
        parent = "/".join(parts[:-1])
        if stem.lower() in _SPLIT_KEYWORDS:
            return parent, stem.lower()
        m = _FILENAME_SPLIT_RE.match(stem)
        if m:
            return f"{parent}/{m.group(1)}", m.group(2).lower()
        return f"{parent}/{stem}", ""
    return stem, ""


def derive_task_group(dataset_name: str) -> str:
    return dataset_name.split("/", 1)[0] if "/" in dataset_name else dataset_name


# ---------------------------------------------------------------------------
# Per-file scanners
# ---------------------------------------------------------------------------

_SEQ_COL_CANDIDATES = ("sequence", "seq", "dna", "Sequence", "Seq", "DNA")
_LABEL_COL_CANDIDATES = ("label", "class", "target", "y", "Label", "Class", "Target")


@dataclass
class FileStats:
    n_records: int = 0
    length_counts: Counter[int] = field(default_factory=Counter)
    label_counts: Counter[str] = field(default_factory=Counter)
    total_bp: int = 0
    label_column: str = ""


def _open_text(path: Path):
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


def _detect_delimiter(path: Path) -> str:
    if ".tsv" in path.name.lower():
        return "\t"
    return ","


def _scan_tabular(path: Path) -> FileStats:
    """Streaming scan of CSV/TSV. Picks the first known sequence column, the
    first known label column. Skips rows without an obvious DNA-like value."""
    stats = FileStats()
    delim = _detect_delimiter(path)

    try:
        with _open_text(path) as f:
            reader = csv.reader(f, delimiter=delim)
            try:
                header = next(reader)
            except StopIteration:
                return stats

            seq_idx: int | None = None
            label_idx: int | None = None
            for cand in _SEQ_COL_CANDIDATES:
                if cand in header:
                    seq_idx = header.index(cand)
                    break
            for cand in _LABEL_COL_CANDIDATES:
                if cand in header:
                    label_idx = header.index(cand)
                    stats.label_column = cand
                    break

            # Fallback: no header named 'sequence' — find the column whose
            # first value looks DNA-like (>= 80% ACGT).
            first_row: list[str] | None = None
            if seq_idx is None and header:
                try:
                    first_row = next(reader)
                except StopIteration:
                    return stats
                for i, value in enumerate(first_row):
                    if _looks_like_dna(value):
                        seq_idx = i
                        break

            if seq_idx is None:
                # Coordinate / metadata file (genomic-benchmarks pattern):
                # no sequence column found. Count rows for the record total
                # but leave length stats empty. Sequence is implicit (must
                # be extracted from a reference FASTA the caller will know
                # about).
                if first_row is not None:
                    stats.n_records += 1
                    if label_idx is not None and label_idx < len(first_row):
                        stats.label_counts[first_row[label_idx]] += 1
                for row in reader:
                    stats.n_records += 1
                    if label_idx is not None and label_idx < len(row):
                        stats.label_counts[row[label_idx]] += 1
                return stats

            if first_row is not None:
                _consume_row(first_row, stats, seq_idx, label_idx)
            for row in reader:
                _consume_row(row, stats, seq_idx, label_idx)
    except (UnicodeDecodeError, csv.Error, OSError):
        # Skip unreadable file silently — never crash the audit.
        return stats
    return stats


def _consume_row(row: list[str], stats: FileStats, seq_idx: int, label_idx: int | None) -> None:
    if seq_idx >= len(row):
        return
    seq = row[seq_idx]
    if not seq:
        return
    length = len(seq)
    stats.n_records += 1
    stats.length_counts[length] += 1
    stats.total_bp += length
    if label_idx is not None and label_idx < len(row):
        stats.label_counts[row[label_idx]] += 1


def _looks_like_dna(value: str) -> bool:
    if len(value) < 20:
        return False
    sample = value[:200].upper()
    ok = sum(1 for ch in sample if ch in "ACGTN")
    return ok / max(len(sample), 1) >= 0.8


def _scan_fasta(path: Path) -> FileStats:
    """FASTA streaming. Counts each `>` record. Headers like
    `>id|label` (PGB convention) populate label_counts."""
    stats = FileStats()
    seq_len = 0
    current_label: str | None = None
    try:
        with _open_text(path) as f:
            for line in f:
                if line.startswith(">"):
                    if seq_len > 0:
                        _commit_fasta(stats, seq_len, current_label)
                    header = line[1:].strip()
                    current_label = _parse_fasta_label(header)
                    seq_len = 0
                else:
                    seq_len += len(line.strip())
            if seq_len > 0:
                _commit_fasta(stats, seq_len, current_label)
    except (UnicodeDecodeError, OSError):
        return stats
    if current_label is not None:
        stats.label_column = "fasta_header_label"
    return stats


def _commit_fasta(stats: FileStats, seq_len: int, label: str | None) -> None:
    stats.n_records += 1
    stats.length_counts[seq_len] += 1
    stats.total_bp += seq_len
    if label is not None:
        stats.label_counts[label] += 1


def _parse_fasta_label(header: str) -> str | None:
    """PGB headers look like `>2:2960_3UTR|1`. Anything after the last `|`
    that is short (<= 4 chars) we treat as the label."""
    if "|" in header:
        tail = header.rsplit("|", 1)[-1].strip()
        if 1 <= len(tail) <= 4:
            return tail
    return None


def scan_file(path: Path) -> FileStats:
    fmt = _file_format(path)
    if fmt == "fasta":
        return _scan_fasta(path)
    return _scan_tabular(path)


# ---------------------------------------------------------------------------
# Task-type derivation
# ---------------------------------------------------------------------------

_REGRESSION_KEYWORDS = (
    "strength",
    "expression",
    "_exp",
    "polya",
    "rna_abundance",
)


def derive_task_type(
    n_labels: int,
    n_records: int,
    dataset_name: str,
    label_counts: Counter[str],
) -> str:
    name = dataset_name.lower()
    if any(kw in name for kw in _REGRESSION_KEYWORDS):
        return "regression"
    if n_labels == 0:
        return "unlabeled"
    if n_labels == 1:
        return "single_label"
    if n_labels == 2:
        return "binary"
    # Many distinct labels relative to record count = could be regression with
    # discretized targets, but in practice DFB / GUE / PGB use small label sets
    # so >2 = multiclass.
    if n_labels > 2:
        return "multiclass"
    return "unknown"


# ---------------------------------------------------------------------------
# Main scan + aggregation
# ---------------------------------------------------------------------------


def scan_collection(name: str, root: Path, max_files: int | None) -> list[dict[str, Any]]:
    if not root.exists():
        print(f"[benchmarks] {name}: root missing ({root}), skipping", file=sys.stderr)
        return []
    tasks: list[dict[str, Any]] = []
    skip_rel = _SKIP_REL_PATHS.get(name, set())
    for idx, path in enumerate(walk_collection(root), start=1):
        if max_files is not None and idx > max_files:
            break
        rel = path.relative_to(root)
        if str(rel) in skip_rel:
            print(f"[benchmarks] {name} {idx}: {rel} (skipped per blacklist)", file=sys.stderr)
            continue
        dataset_name, split = derive_dataset_split(rel)
        task_group = derive_task_group(dataset_name)
        print(
            f"[benchmarks] {name} {idx}: {rel}",
            file=sys.stderr,
        )
        stats = scan_file(path)

        # Filename-as-label fallback: when the file itself doesn't expose a
        # label column but sits next to sibling files in the same directory
        # whose stems are not split keywords, treat the stem as the implicit
        # class label. Activates on genomic-benchmarks's `positive.csv` /
        # `negative.csv` pairs and similar conventions.
        if stats.n_records > 0 and not stats.label_counts:
            stem = _file_stem(path)
            stem_is_split = stem.lower() in _SPLIT_KEYWORDS or bool(
                _FILENAME_SPLIT_RE.match(stem)
            )
            if not stem_is_split:
                try:
                    siblings = [
                        p
                        for p in path.parent.iterdir()
                        if p != path
                        and _is_data_file(p)
                        and _file_stem(p).lower() not in _SPLIT_KEYWORDS
                    ]
                except OSError:
                    siblings = []
                if siblings:
                    stats.label_counts[stem] = stats.n_records
                    stats.label_column = "filename_stem"

        if stats.n_records == 0:
            # Empty or unparsable file — record as zero-row entry so the user
            # can see something was found but not parsed.
            tasks.append(
                {
                    "collection": name,
                    "task_group": task_group,
                    "dataset_name": dataset_name,
                    "split": split,
                    "file": str(rel),
                    "format": _file_format(path),
                    "n_records": 0,
                    "n_labels": 0,
                    "task_type": "unknown",
                    "len_min": None,
                    "len_q25": None,
                    "len_median": None,
                    "len_q75": None,
                    "len_q95": None,
                    "len_max": None,
                    "mean_len": None,
                    "top_labels": "",
                }
            )
            continue
        n_labels = len(stats.label_counts)
        task_type = derive_task_type(
            n_labels=n_labels,
            n_records=stats.n_records,
            dataset_name=dataset_name,
            label_counts=stats.label_counts,
        )
        tasks.append(
            {
                "collection": name,
                "task_group": task_group,
                "dataset_name": dataset_name,
                "split": split,
                "file": str(rel),
                "format": _file_format(path),
                "n_records": stats.n_records,
                "n_labels": n_labels,
                "task_type": task_type,
                "len_min": percentile_from_counts(stats.length_counts, 0.0),
                "len_q25": percentile_from_counts(stats.length_counts, 0.25),
                "len_median": percentile_from_counts(stats.length_counts, 0.5),
                "len_q75": percentile_from_counts(stats.length_counts, 0.75),
                "len_q95": percentile_from_counts(stats.length_counts, 0.95),
                "len_max": percentile_from_counts(stats.length_counts, 1.0),
                "mean_len": (
                    round(stats.total_bp / stats.n_records, 1)
                    if stats.total_bp
                    else None
                ),
                "top_labels": _compact_counter(stats.label_counts, max_items=6),
            }
        )
    return tasks


def _compact_counter(counter: Counter[Any], max_items: int = 6) -> str:
    if not counter:
        return ""
    top = counter.most_common(max_items)
    return ";".join(f"{k}:{v}" for k, v in top)


def build_collection_summaries(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per source_collection: unique datasets, task_type counts,
    length min/max over all included tasks."""
    by_collection: dict[str, list[dict[str, Any]]] = {}
    for t in tasks:
        by_collection.setdefault(t["collection"], []).append(t)
    out: list[dict[str, Any]] = []
    for collection in sorted(by_collection):
        rows = by_collection[collection]
        unique_datasets = sorted({r["dataset_name"] for r in rows})
        task_groups = sorted({r["task_group"] for r in rows})
        type_counts: dict[str, int] = {}
        for r in rows:
            type_counts[r["task_type"]] = type_counts.get(r["task_type"], 0) + 1
        len_min_values = [r["len_min"] for r in rows if r["len_min"] is not None]
        len_max_values = [r["len_max"] for r in rows if r["len_max"] is not None]
        len_med_values = [r["len_median"] for r in rows if r["len_median"] is not None]
        n_records_total = sum(r["n_records"] for r in rows)
        out.append(
            {
                "collection": collection,
                "n_files": len(rows),
                "n_datasets_unique": len(unique_datasets),
                "n_task_groups": len(task_groups),
                "task_groups": task_groups,
                "task_types": type_counts,
                "n_records_total": n_records_total,
                "len_min": min(len_min_values) if len_min_values else None,
                "len_median_of_medians": (
                    sorted(len_med_values)[len(len_med_values) // 2]
                    if len_med_values
                    else None
                ),
                "len_max": max(len_max_values) if len_max_values else None,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def render_markdown(
    tasks: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    out_path: Path,
) -> None:
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover
        pd = None  # type: ignore[assignment]

    n_tasks = len(tasks)
    n_records = sum(t["n_records"] for t in tasks)
    type_counts: dict[str, int] = {}
    for t in tasks:
        type_counts[t["task_type"]] = type_counts.get(t["task_type"], 0) + 1
    summary_lines = [
        f"Source collections scanned: **{len(summaries)}**",
        f"Total task-file rows: **{n_tasks}**",
        f"Total records across all files: **{n_records:,}**",
        "Task-type breakdown: "
        + ", ".join(
            f"{k}={v}" for k, v in sorted(type_counts.items(), key=lambda kv: -kv[1])
        ),
        "Source JSON: `data/audits/benchmarks.json`.",
    ]

    sections: list[tuple[str, str]] = []
    if pd is not None and summaries:
        # Flatten task_groups / task_types into compact strings for table view.
        sum_rows = []
        for s in summaries:
            row = dict(s)
            row["task_groups"] = ", ".join(s["task_groups"])
            row["task_types"] = ", ".join(
                f"{k}={v}" for k, v in sorted(s["task_types"].items(), key=lambda kv: -kv[1])
            )
            sum_rows.append(row)
        df_sum = pd.DataFrame(sum_rows)
        sections.append(("Collection summary", df_sum.to_markdown(index=False)))

    if pd is not None and tasks:
        df_tasks = pd.DataFrame(tasks)
        sections.append(("All tasks", df_tasks.to_markdown(index=False)))

    write_markdown(out_path, "Benchmark audit", summary_lines, sections=sections)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=AUDIT_DIR)
    parser.add_argument(
        "--collections",
        type=str,
        default=None,
        help="Comma-separated subset of collection names "
        f"(default: all in {sorted(BENCHMARK_ROOTS)}).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Scan at most N files per collection (debug).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    requested = (
        {s.strip() for s in args.collections.split(",") if s.strip()}
        if args.collections
        else set(BENCHMARK_ROOTS)
    )
    unknown = requested - set(BENCHMARK_ROOTS)
    if unknown:
        print(
            f"[benchmarks] unknown collections: {sorted(unknown)}; "
            f"known: {sorted(BENCHMARK_ROOTS)}",
            file=sys.stderr,
        )
        sys.exit(1)

    tasks: list[dict[str, Any]] = []
    for name in sorted(BENCHMARK_ROOTS):
        if name not in requested:
            continue
        tasks.extend(scan_collection(name, BENCHMARK_ROOTS[name], args.max_files))

    summaries = build_collection_summaries(tasks)

    json_path = args.out_dir / "benchmarks.json"
    md_path = args.out_dir / "benchmarks.md"
    write_json(
        {
            "generated_by": "scripts/audits/benchmarks.py",
            "n_files_scanned": len(tasks),
            "collections": summaries,
            "tasks": tasks,
        },
        json_path,
    )
    render_markdown(tasks, summaries, md_path)
    print(
        f"[benchmarks] wrote {json_path} + {md_path.name} "
        f"({len(tasks)} task-files across {len(summaries)} collections)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
