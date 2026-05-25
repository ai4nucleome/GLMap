#!/usr/bin/env python3
"""Generate Table 2: downstream benchmark task summary.

The phase 2 downstream evaluation scores embeddings from each scored
model on 6 binary classification tasks drawn from established gLM
benchmarks (GUE, NT Bench, iDNA-ABF, iPro-WAEL). This script summarises
each task — biological domain, source benchmark, sequence length, train
/ test counts, and label class balance — in a single paper-ready table.

Inputs
------
  data/dna_foundation_benchmark/<benchmark>/<task>/{train,test}.csv
    For train/test row counts, sequence lengths, and label distributions.

Output
------
  tables/table2_downstream_tasks.tex   — booktabs LaTeX, paper-ready.
  Markdown preview printed to stdout.

Re-run whenever the downstream-task set changes:

  $PY scripts/tables/table2_downstream_tasks.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_ROOT = REPO_ROOT / "data" / "dna_foundation_benchmark"
TEX_OUT = REPO_ROOT / "tables" / "table2_downstream_tasks.tex"

# Canonical task list — the 6 binary classification tasks used by
# scripts/run_downstream_classify.py. Each entry carries
# human-readable metadata; the numeric statistics come directly from
# the curated dna_foundation_benchmark CSVs.
TASKS = [
    {
        "task_id": "EMP/Yeast_H4",
        "domain": "Histone modification (H4, yeast)",
    },
    {
        "task_id": "enhancers/enhancer",
        "domain": "Enhancer detection (human)",
    },
    {
        "task_id": "iDNA_ABF/5mC",
        "domain": "5-methylcytosine detection",
    },
    {
        "task_id": "iPro-WAEL/Promoter_Arabidopsis_TATA",
        "domain": "Plant TATA promoter (Arabidopsis)",
    },
    {
        "task_id": "mouse/mouse_TFBS_3",
        "domain": "Mouse TF binding site (TF \\#3)",
    },
    {
        "task_id": "prom/promoter_tata_300bps",
        "domain": "Human TATA promoter (300 bp)",
    },
]


def _read_split(task_id: str, split: str) -> pd.DataFrame | None:
    csv_path = BENCH_ROOT / task_id / f"{split}.csv"
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)


def _find_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    lookup = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in lookup:
            return lookup[c]
    return None


def _split_stats(df: pd.DataFrame | None) -> dict:
    if df is None:
        return {"n": None, "labels": None, "lengths": None}
    label_col = _find_column(df, ("label", "labels", "target", "y"))
    seq_col = _find_column(df, ("sequence", "seq", "dna"))
    labels = None if label_col is None else df[label_col].value_counts().to_dict()
    lengths = None if seq_col is None else df[seq_col].astype(str).str.len()
    return {"n": len(df), "labels": labels, "lengths": lengths}


def _seq_length_range(stats_train: dict, stats_test: dict) -> tuple[int, int, int]:
    length_parts = [
        s["lengths"] for s in (stats_train, stats_test)
        if s["lengths"] is not None
    ]
    if not length_parts:
        return (None, None, None)
    lengths = pd.concat(length_parts, ignore_index=True)
    return int(lengths.min()), int(lengths.max()), int(lengths.median())


def _format_seq_len(lo: int | None, hi: int | None, med: int | None) -> str:
    if lo is None:
        return "—"
    if lo == hi:
        return f"{lo}"
    return f"{lo}--{hi}"


def _format_class_balance(stats_train: dict, stats_test: dict) -> str:
    """Minority-class fraction over (train + test). Both splits are
    pooled because most tasks are balanced or stratified-split."""
    pooled: dict = {}
    for stats in (stats_train, stats_test):
        if stats["labels"] is None:
            continue
        for k, v in stats["labels"].items():
            pooled[k] = pooled.get(k, 0) + v
    if not pooled:
        return "—"
    total = sum(pooled.values())
    minor = min(pooled.values())
    return f"{100*minor/total:.1f}\\%"


def _tex_escape(s: str) -> str:
    """Minimal LaTeX-escape for the task id text — only underscores
    need quoting in our usage; the rest stays verbatim."""
    return s.replace("_", r"\_")


def main() -> None:
    print(f"[table2] source: {BENCH_ROOT.relative_to(REPO_ROOT)}", flush=True)
    rows = []
    for t in TASKS:
        train_stats = _split_stats(_read_split(t["task_id"], "train"))
        test_stats = _split_stats(_read_split(t["task_id"], "test"))
        lo, hi, med = _seq_length_range(train_stats, test_stats)
        row = {
            **t,
            "n_train":      train_stats["n"],
            "n_test":       test_stats["n"],
            "seq_len_str":  _format_seq_len(lo, hi, med),
            "class_balance": _format_class_balance(train_stats, test_stats),
            "train_labels": train_stats["labels"],
            "test_labels":  test_stats["labels"],
        }
        rows.append(row)

    # ── Markdown preview ── #
    print("\n# Table 2 (Markdown preview)\n")
    print("| Tasks | Seq length (bp) | n train | n test | Minority class |")
    print("|---|---:|---:|---:|---:|")
    for r in rows:
        print(
            f"| `{r['task_id']}` | "
            f"{r['seq_len_str']} | {r['n_train']:,} | {r['n_test']:,} | "
            f"{r['class_balance'].replace(chr(92) + chr(37), '%')} |"
        )
    print()

    # ── LaTeX booktabs ── #
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    n_tasks = len(rows)
    total_train = sum(r["n_train"] for r in rows if r["n_train"])
    total_test  = sum(r["n_test"]  for r in rows if r["n_test"])
    caption = (
        f"Downstream task summary. The phase 2 evaluation uses {n_tasks} "
        f"binary classification tasks selected from "
        f"\\texttt{{data/dna\\_foundation\\_benchmark/}}, spanning histone "
        f"modification, enhancer detection, DNA methylation, promoter "
        f"detection, and TF binding. Sequence lengths, train/test sizes, "
        f"and label balance are computed directly from each task's curated "
        f"\\texttt{{train.csv}} and \\texttt{{test.csv}} files. Minority "
        f"class is shown as percentage of pooled train + test labels. "
        f"Total samples: "
        f"{total_train:,} train + {total_test:,} test = "
        f"{total_train + total_test:,}."
    )
    lines.append(f"  \\caption{{{caption}}}")
    lines.append(r"  \label{tab:downstream_tasks}")
    lines.append(r"  \small")
    lines.append(r"  \begin{tabular}{l c r r r}")
    lines.append(r"    \toprule")
    lines.append(
        r"    Tasks & Seq.\ length (bp) & "
        r"$N_{\mathrm{train}}$ & $N_{\mathrm{test}}$ & "
        r"Minority class (\%) \\"
    )
    lines.append(r"    \midrule")
    for r in rows:
        lines.append(
            f"    \\texttt{{{_tex_escape(r['task_id'])}}} & "
            f"{r['seq_len_str']} & "
            f"{r['n_train']:,} & "
            f"{r['n_test']:,} & "
            f"{r['class_balance']} \\\\"
        )
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    tex = "\n".join(lines) + "\n"

    TEX_OUT.parent.mkdir(parents=True, exist_ok=True)
    TEX_OUT.write_text(tex)
    print(f"# LaTeX written to: {TEX_OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
