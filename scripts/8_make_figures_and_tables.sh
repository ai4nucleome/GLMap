#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Regenerate EVERY paper figure + table from the precomputed artefacts that
# ship with the repo (results/scores/matrices/, results/analysis/, the panel,
# the audit). No GPU, no model weights, no scoring — just install the analysis
# stack first:
#
#     pip install -e .
#     bash scripts/8_make_figures_and_tables.sh
#
# Outputs land in results/figures/ (PDF) and results/tables/ (LaTeX).
# Override the interpreter with PY=/path/to/python if `python` is not on PATH.
#
# Note: Table 2 needs the external DNA Foundation Benchmark CSVs for its
# sequence-length / sample-count columns; without them it still renders the
# task list with "—" placeholders (and prints the download command).
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
PY="${PY:-python}"
cd "$(dirname "$0")/.."

pass=0; fail=0; failed=()
run() {
    echo "── $1"
    if "$PY" "$1" >/dev/null 2>&1; then
        pass=$((pass+1))
    else
        # re-run showing output so the failure is visible
        "$PY" "$1" 2>&1 | tail -3
        fail=$((fail+1)); failed+=("$1")
    fi
}

echo "===== Tables (results/tables/) ====="
for t in scripts/tables/table1_panel_composition.py \
         scripts/tables/table2_downstream_tasks.py \
         scripts/tables/table3_evo_family_distances.py \
         scripts/tables/table4_phenotype_prediction_metrics.py; do
    run "$t"
done

echo "===== Figures (results/figures/) ====="
for f in scripts/figures/fig2ab_panel_composition.py \
         scripts/figures/fig2c_split_half_consistency.py \
         scripts/figures/fig2d_q_heatmap.py \
         scripts/figures/fig2e_glmap_dendrogram.py \
         scripts/figures/fig3a_model_map_family.py \
         scripts/figures/fig3b_model_map_model_weight.py \
         scripts/figures/fig3c_model_map_6task_mean.py \
         scripts/figures/fig4a_downstream_auc_distribution.py \
         scripts/figures/fig4b_phenotype_prediction_scatter.py \
         scripts/figures/figS2_ar_mlm_merge.py \
         scripts/figures/figS3_stride_pll_per_model_r.py \
         scripts/figures/D_matrix_heatmap.py; do
    run "$f"
done

echo
echo "===== Done: ${pass} ok, ${fail} failed ====="
if [ "${fail}" -gt 0 ]; then
    printf '  failed: %s\n' "${failed[@]}"
    exit 1
fi
echo "Figures -> results/figures/ ; tables -> results/tables/"
