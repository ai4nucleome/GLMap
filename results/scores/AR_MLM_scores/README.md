# Per-model scores (slimmed)

Each subdirectory is one model (slug = `hf_id` with `/` → `__`) and holds:

- `probes.parquet` — the model's likelihood response on the full
  10,000-probe panel (one row per probe, aligned by `probe_id`).
- `<slug>.json` — per-model metadata snapshot (`hf_id`, `branch`,
  `context_tokens`, `loader_kind`, `is_codon`, `stride_primary`,
  `scoring_protocol`, …), written by the scoring pipeline alongside the
  parquet.

The `token_log_probs` column (per-token log-probability lists) has been
removed to reduce repository size. The remaining columns — `sum_log_p`,
`ell_per_base`, `bpb`, and probe metadata — are sufficient for matrix
construction and all downstream analyses reported in the paper.

To regenerate per-token vectors, re-run the scoring pipeline:

```bash
python scripts/score/scoring_worker.py --from-audit --device cuda:0
```
