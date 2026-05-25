# Per-model scores (slimmed)

Each subdirectory contains one `probes.parquet` file with the likelihood
response of that model on the full 10,000-probe panel.

The `token_log_probs` column (per-token log-probability lists) has been
removed to reduce repository size. The remaining columns — `sum_log_p`,
`ell_per_base`, `bpb`, and probe metadata — are sufficient for matrix
construction and all downstream analyses reported in the paper.

To regenerate per-token vectors, re-run the scoring pipeline:

```bash
python scripts/run_phase1_scoring.py --from-audit --device cuda:0
```
