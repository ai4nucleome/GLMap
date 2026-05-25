#!/usr/bin/env python3
"""Generate Table 1: GLMap probe panel composition.

Reads out_panel/main_panel.parquet, aggregates per (biological category,
functional element), and emits two formats:

  - Markdown to stdout (for visual review)
  - LaTeX (booktabs) to tables/table1_panel_composition.tex

Re-run after any panel rebuild (build_panel.py / build_control_panel.py)
to keep the figure in sync.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
PANEL_PQ = REPO_ROOT / "out_panel" / "main_panel.parquet"
TEX_OUT = REPO_ROOT / "tables" / "table1_panel_composition.tex"

GROUP_ORDER = ["Human", "Plant", "Fungi", "Virus"]


def src_to_dataset(s: str) -> str:
    """Extract a short dataset identifier from a per-probe source string.

    Inputs look like ``data/GUE/prom/prom_300_all/train.csv#row_3555`` or
    ``data/PGB/chromatin_access/.../seq.fasta``. We strip the ``data/``
    prefix, the ``#row_N`` suffix, and the trailing ``train.csv`` or
    ``.fasta`` filename so what remains identifies the benchmark dataset
    (e.g. ``GUE/prom/prom_300_all``).

    Apply DATASET_RENAME afterwards to give a few benchmarks their
    canonical short name (e.g. ``dna_foundation_benchmark`` is called
    ``NT Bench`` in the GLMap paper).
    """
    s = s.split("#")[0]
    if s.startswith("data/"):
        s = s[5:]
    if s.endswith(".csv"):
        s = s.rsplit("/", 1)[0]
    if s.endswith("/train"):
        s = s[:-6]
    if s.endswith(".fasta") or s.endswith(".fa"):
        s = s.rsplit("/", 1)[0]
    return DATASET_RENAME.get(s, s)


# Canonical short names used in the published Table 1.
DATASET_RENAME = {
    "dna_foundation_benchmark/enhancers/enhancer": "NT Bench/enhancers",
}


def fmt_length(mean_bp: float) -> str:
    """Mean per-probe length rounded to nearest integer."""
    return f"{round(mean_bp):,}"


def fmt_gc(mean: float) -> str:
    return f"{100 * mean:.1f}"


def build_rows(df) -> list[dict]:
    """Aggregate per (group, element). Returns rows sorted by canonical order."""
    df = df.copy()
    df["dataset"] = df["source"].map(src_to_dataset)

    rows = []
    for elem in df["functional_element"].drop_duplicates().tolist():
        sub = df[df["functional_element"] == elem]
        group = sub["species_group"].iloc[0]
        top_ds = Counter(sub["dataset"]).most_common(1)[0][0]
        rows.append(
            dict(
                group=group,
                element=elem,
                N=len(sub),
                L_mean=float(sub["length_bp"].mean()),
                GC_mean=float(sub["GC_content"].mean()),
                n_species=int(sub["species"].nunique()),
                dataset=top_ds,
            )
        )
    rows.sort(key=lambda r: (GROUP_ORDER.index(r["group"]), -r["N"]))
    return rows


def render_markdown(rows: list[dict], group_totals: dict[str, int], grand_total: int) -> str:
    """Markdown table for human review."""
    head = (
        "| Category | Functional element | N | Mean Length (bp) | "
        "# Species | Source |\n"
        "|---|---|:---:|:---:|:---:|---|"
    )
    lines = [head]
    current_group = None
    for r in rows:
        group = r["group"] if r["group"] != current_group else ""
        current_group = r["group"]
        lines.append(
            f"| {group} | {r['element']} | {r['N']} | "
            f"{fmt_length(r['L_mean'])} | "
            f"{r['n_species']} | "
            f"`{r['dataset']}` |"
        )
    # subtotal rows + grand total
    lines.append("|  |  |  |  |  |  |")
    for g in GROUP_ORDER:
        lines.append(f"| **{g} subtotal** |  | **{group_totals[g]}** |  |  |  |")
    lines.append(f"| **Grand total** |  | **{grand_total}** |  |  |  |")
    return "\n".join(lines)


def render_latex(rows: list[dict], group_totals: dict[str, int], grand_total: int) -> str:
    """LaTeX booktabs table. Compatible with OUP authoring template.

    "Mean Length" is the per-probe arithmetic mean within each functional
    element, rounded to the nearest integer. The N / Mean Length /
    \\# Species columns are centered; Category / Functional element /
    Source are left-aligned. Subtotal / grand-total rows are intentionally
    omitted; per-category counts are stated in the caption / surrounding
    prose.
    """
    body_lines = []
    for i, r in enumerate(rows):
        if i > 0 and r["group"] != rows[i - 1]["group"]:
            body_lines.append("\\midrule")
        group_cell = r["group"] if (i == 0 or rows[i - 1]["group"] != r["group"]) else ""
        dataset_tt = r["dataset"].replace("_", "\\_")
        elem_tt = r["element"].replace("_", "\\_")
        body_lines.append(
            f"{group_cell} & \\texttt{{{elem_tt}}} & "
            f"{r['N']:,} & {fmt_length(r['L_mean'])} & "
            f"{r['n_species']} & "
            f"\\texttt{{{dataset_tt}}} \\\\"
        )
    body = "\n".join(body_lines)
    category_breakdown = "; ".join(
        f"{g}: {group_totals[g]:,}" for g in GROUP_ORDER
    )
    caption = (
        f"GLMap probe panel composition. The main panel contains "
        f"{grand_total:,} probes distributed across 4 biological "
        f"categories ({category_breakdown}) and 14 tasks."
    )
    tex = (
        "\\begin{table}[t]\n"
        "  \\centering\n"
        f"  \\caption{{{caption}}}\n"
        "  \\label{tab:panel_composition}\n"
        "  \\small\n"
        "  \\begin{tabular}{l l c c c l}\n"
        "    \\toprule\n"
        "    Category & Functional element & $N$ & Mean Length (bp) & \\# Species & Source \\\\\n"
        "    \\midrule\n"
        f"{body}\n"
        "    \\bottomrule\n"
        "  \\end{tabular}\n"
        "\\end{table}\n"
    )
    return tex


def main() -> None:
    if not PANEL_PQ.exists():
        sys.exit(f"main panel parquet not found: {PANEL_PQ}")
    df = pq.read_table(PANEL_PQ).to_pandas()
    rows = build_rows(df)

    group_totals = {
        g: sum(r["N"] for r in rows if r["group"] == g) for g in GROUP_ORDER
    }
    grand_total = sum(r["N"] for r in rows)

    md = render_markdown(rows, group_totals, grand_total)
    tex = render_latex(rows, group_totals, grand_total)

    print("# Table 1 — GLMap probe panel composition (Markdown preview)\n")
    print(md)
    print()
    print(f"# LaTeX written to: {TEX_OUT.relative_to(REPO_ROOT)}\n")

    TEX_OUT.parent.mkdir(parents=True, exist_ok=True)
    TEX_OUT.write_text(tex)


if __name__ == "__main__":
    main()
