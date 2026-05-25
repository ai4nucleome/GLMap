#!/usr/bin/env python3
"""Build the Stage 2 panel summary table (paper Table 1 + Supplementary S1).

For each of the 14 functional_element entries in `out_panel/main_panel.parquet`,
aggregate:
  - source benchmark + original task path
  - species_group (domain) and dominant species
  - probe count, percentage of the 10,000-probe panel
  - sequence length (median + range)
  - GC content (median + IQR)

Two outputs (regenerated; the TSV is gitignored, the markdown is tracked
under docs/ for the paper write-up):

  out_panel/panel_summary_table.tsv          machine-readable per-element row
  out_panel/panel_summary_per_dataset.tsv    detail: per (element, source file)
  docs/panel_summary.md                      human-readable markdown for the
                                             paper Table 1 + Supplementary S1

Run:
  $PY scripts/build_panel_summary_table.py [--panel out_panel/main_panel.parquet]
                                            [--out-tsv out_panel]
                                            [--out-md docs]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


# Benchmark name → (short label, citation). Used to fill the "Source" column
# with a paper-friendly tag instead of the raw directory name from
# panel_sources.yaml.
BENCHMARK_LABELS: dict[str, tuple[str, str]] = {
    "GUE": ("GUE", "DNABERT-2, Zhou et al. 2024"),
    "PGB": ("PGB", "Plant Genomic Benchmark, Mendoza-Revilla et al. 2024"),
    "dna_foundation_benchmark": ("NT-DFB", "Nucleotide Transformer, Dalla-Torre et al. 2024"),
}

# Display order for the 14 functional_elements: grouped by species_group
# (Human → Plant → Fungi → Virus). Matches phase1_main_figure.CLASS_ORDER.
ELEMENT_ORDER = (
    # Human (4 elements)
    "promoter",
    "enhancer",
    "splice_donor",
    "splice_acceptor",
    # Plant (6 elements)
    "chromatin_access",
    "polyA",
    "lncRNA",
    "nascent_RNA",
    "splicing_plant_donor",
    "splicing_plant_acceptor",
    # Fungi (2 elements)
    "yeast_genome",
    "fungi_genome",
    # Virus (2 elements)
    "virus_variants",
    "virus_species",
)


def parse_source(s: str) -> tuple[str, str]:
    """Extract (benchmark_dir, task_path_without_row_suffix) from a source
    column entry like 'data/GUE/EMP/H3/train.csv#row_1001'."""
    parts = str(s).split("/")
    if len(parts) >= 2 and parts[0] == "data":
        bench = parts[1]
        rest = "/".join(parts[2:]) if len(parts) > 2 else ""
        rest = re.sub(r"#row_\d+.*$", "", rest)
        return bench, rest
    return "other", str(s)


def _fmt_range(lo: float, hi: float, fmt: str = "{:.0f}") -> str:
    if lo == hi:
        return fmt.format(lo)
    return f"{fmt.format(lo)}–{fmt.format(hi)}"


def _fmt_median_iqr(median: float, q25: float, q75: float, fmt: str = "{:.2f}") -> str:
    return f"{fmt.format(median)} ({fmt.format(q25)}–{fmt.format(q75)})"


def _species_summary(sub: pd.DataFrame) -> str:
    """Top species in this element's slice, or 'multi-species (N)' if many."""
    species_counts = sub["species"].value_counts()
    if len(species_counts) == 1:
        return species_counts.index[0]
    # If one species is heavily dominant (>=85%), name it
    top = species_counts.iloc[0]
    if top / len(sub) >= 0.85:
        return f"{species_counts.index[0]} ({top}/{len(sub)})"
    return f"multi-species (n={len(species_counts)})"


def _tasks_for_element(sub: pd.DataFrame, benchmark: str) -> str:
    """Compact task description: top tasks within the (element, benchmark)
    slice, comma-separated."""
    task_counts = sub["task_path"].value_counts()
    # Strip the trailing /train.csv or /<species>_train.fa to keep the task
    # identifier; drop the file extension at the end.
    short = []
    for t in task_counts.index[:5]:
        # Last segment is the file; the task is the parent path
        parent = "/".join(t.split("/")[:-1]) or t
        short.append(parent)
    # Unique while preserving order
    seen = []
    for s in short:
        if s not in seen:
            seen.append(s)
        if len(seen) >= 3:
            break
    return ", ".join(seen)


def build_per_element(panel: pd.DataFrame, n_total: int) -> pd.DataFrame:
    """One row per functional_element, with all summary stats."""
    rows = []
    for elem in ELEMENT_ORDER:
        sub = panel[panel["functional_element"] == elem]
        if len(sub) == 0:
            continue
        n = len(sub)
        species_group = sub["species_group"].iloc[0]
        species = _species_summary(sub)
        benchmarks_in_element = sub["benchmark"].unique()
        if len(benchmarks_in_element) == 1:
            bench = benchmarks_in_element[0]
            short, citation = BENCHMARK_LABELS.get(bench, (bench, ""))
            bench_label = short
        else:
            bench_label = "/".join(
                BENCHMARK_LABELS.get(b, (b, ""))[0] for b in benchmarks_in_element
            )
        tasks = _tasks_for_element(sub, bench_label)

        len_med = sub["length_bp"].median()
        len_q25 = sub["length_bp"].quantile(0.25)
        len_q75 = sub["length_bp"].quantile(0.75)
        len_min = sub["length_bp"].min()
        len_max = sub["length_bp"].max()
        gc_med = sub["GC_content"].median()
        gc_q25 = sub["GC_content"].quantile(0.25)
        gc_q75 = sub["GC_content"].quantile(0.75)

        rows.append({
            "Element": elem,
            "Species group": species_group,
            "Species": species,
            "Source benchmark": bench_label,
            "Original task path": tasks,
            "n": n,
            "Length (bp)": _fmt_range(len_min, len_max) if len_min != len_max else f"{int(len_min)}",
            "Length median (IQR)": _fmt_median_iqr(len_med, len_q25, len_q75, "{:.0f}"),
            "GC median (IQR)": _fmt_median_iqr(gc_med, gc_q25, gc_q75, "{:.2f}"),
            "% of panel": f"{100*n/n_total:.1f}",
        })
    return pd.DataFrame(rows)


def build_per_dataset(panel: pd.DataFrame) -> pd.DataFrame:
    """One row per (functional_element, task_path) pair — Supplementary S1."""
    rows = []
    for (elem, task), sub in panel.groupby(["functional_element", "task_path"]):
        n = len(sub)
        species_group = sub["species_group"].iloc[0]
        species = _species_summary(sub)
        bench = sub["benchmark"].iloc[0]
        bench_short = BENCHMARK_LABELS.get(bench, (bench, ""))[0]

        len_med = sub["length_bp"].median()
        gc_med = sub["GC_content"].median()

        rows.append({
            "Element": elem,
            "Species group": species_group,
            "Species": species,
            "Source benchmark": bench_short,
            "Source file (relative to data/)": f"{bench}/{task}",
            "n": n,
            "Length median": int(len_med),
            "GC median": f"{gc_med:.3f}",
        })
    df = pd.DataFrame(rows)
    # Sort by ELEMENT_ORDER then by n descending
    df["__order"] = df["Element"].map({e: i for i, e in enumerate(ELEMENT_ORDER)})
    df = df.sort_values(["__order", "n"], ascending=[True, False]).drop("__order", axis=1)
    return df.reset_index(drop=True)


# ───────────────────────────── markdown rendering ─────────────────────────── #


def _df_to_markdown(df: pd.DataFrame) -> str:
    # Plain markdown table (avoids pandas's optional `tabulate` dependency).
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def render_markdown(
    per_elem: pd.DataFrame, per_dataset: pd.DataFrame,
    n_total: int,
) -> str:
    benchmark_legend = "\n".join(
        f"- **{short}** — {citation}"
        for short, citation in BENCHMARK_LABELS.values()
    )
    out = []
    out.append("# Stage 2 probe panel — composition summary")
    out.append("")
    out.append(
        f"Total probes: **{n_total:,}** across **{len(ELEMENT_ORDER)} "
        f"functional elements** × **4 species groups** "
        f"(Human / Plant / Fungi / Virus)."
    )
    out.append("")
    out.append("**Source benchmarks**:")
    out.append("")
    out.append(benchmark_legend)
    out.append("")
    out.append("Probes are uniformly the **positive** examples from each "
               "source classification task (i.e. functional element instances, "
               "not background sequences); the gLM-scoring pipeline ignores "
               "the original task labels and uses the sequences only.")
    out.append("")
    out.append("## Table 1 — per-element summary")
    out.append("")
    out.append(_df_to_markdown(per_elem))
    out.append("")
    out.append("Notes:")
    out.append("- `Length (bp)` is the observed range; `Length median (IQR)` "
               "shows the central tendency. For most elements the panel build "
               "fixes the probe length, so range collapses to a single value.")
    out.append("- `GC median (IQR)` is computed on the panel's final selected "
               "probes after any length truncation / filtering.")
    out.append("- `% of panel` rounds to one decimal; columns may not sum to "
               "exactly 100 due to rounding.")
    out.append("")
    out.append("## Supplementary Table S1 — per-source file detail")
    out.append("")
    out.append(
        f"Within each functional element, the panel draws from one or more "
        f"source files. This table shows the breakdown ({len(per_dataset)} "
        f"rows). Useful when reviewers ask which species / which underlying "
        f"task contributed a given subset of probes."
    )
    out.append("")
    out.append(_df_to_markdown(per_dataset))
    out.append("")
    return "\n".join(out)


# ───────────────────────────────── main ──────────────────────────────────── #


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--panel", type=Path,
                   default=REPO_ROOT / "out_panel" / "main_panel.parquet",
                   help="Stage 2 main panel parquet.")
    p.add_argument("--out-tsv", type=Path,
                   default=REPO_ROOT / "out_panel",
                   help="Directory for TSV outputs (gitignored).")
    p.add_argument("--out-md", type=Path,
                   default=REPO_ROOT / "docs",
                   help="Directory for the markdown summary (tracked).")
    p.add_argument("--md-name", type=str, default="panel_summary.md",
                   help="Filename for the markdown output.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.panel.exists():
        sys.exit(f"panel parquet not found at {args.panel}")

    panel = pd.read_parquet(args.panel)
    panel["benchmark"], panel["task_path"] = zip(*panel["source"].apply(parse_source))
    n_total = len(panel)
    print(f"[panel_summary] loaded {n_total} probes from {args.panel}", flush=True)

    per_elem = build_per_element(panel, n_total)
    per_dataset = build_per_dataset(panel)
    md = render_markdown(per_elem, per_dataset, n_total)

    args.out_tsv.mkdir(parents=True, exist_ok=True)
    args.out_md.mkdir(parents=True, exist_ok=True)

    tsv_elem = args.out_tsv / "panel_summary_table.tsv"
    tsv_ds = args.out_tsv / "panel_summary_per_dataset.tsv"
    md_path = args.out_md / args.md_name

    per_elem.to_csv(tsv_elem, sep="\t", index=False)
    per_dataset.to_csv(tsv_ds, sep="\t", index=False)
    md_path.write_text(md)

    print(f"[done] wrote {tsv_elem}  ({len(per_elem)} rows)", flush=True)
    print(f"[done] wrote {tsv_ds}  ({len(per_dataset)} rows)", flush=True)
    print(f"[done] wrote {md_path}", flush=True)


if __name__ == "__main__":
    main()
