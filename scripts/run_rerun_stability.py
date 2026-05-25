#!/usr/bin/env python3
"""Standalone rerun-stability gate.

Three model sources are supported:

  --from-audit    iterate every scorable model in data/audits/models.json
                  (123 in the current catalog), auto-building ModelSpec from
                  the audit fields (hf_id / branch / context_tokens).
                  Special-cased loader routing (see _audit_entry_to_spec below
                  for the authoritative dispatch table):
                      lingxusb/megaDNA           -> MegaDNALoader
                      lingxusb/PlasmidGPT        -> PlasmidGPTLoader
                      GenSLM-*                   -> GenSLMLoader
                      living-models/Botanic0-*   -> BotanicLoader (AutoModel)
                      plant-llms/PlantBiMoE      -> PlantBiMoELoader
                      JadenLong/MutBERT*         -> MutBERTLoader (RoPE scaling)
                      LongSafari/hyenadna-*      -> HyenaDNALoader
                      genbio-ai/AIDO.DNA*        -> AIDOLoader
                      arcinstitute/evo2_* / evo-design/evo-2-* -> Evo2Loader
                      togethercomputer/evo-1-* / evo-design/evo-1-* / etc.
                                                 -> Evo1Loader
                      GenerTeam/GENERator-*      -> GENERatorLoader (k=6 right-trunc)
                      InstaDeepAI/NTv3_*         -> NTv3MaskedLMLoader (U-Net
                                                    N-right-pad; length_multiple
                                                    = 32 for *5downsample*,
                                                    128 otherwise)
                  Everything else is treated as a HF causal- or masked-LM.
                  Supervised models in the audit are skipped (no LM head).
  --hf-ids       comma-separated list of exact HF ids (overrides --from-audit
                  and DEFAULT_MODELS).
  (default)      iterate DEFAULT_MODELS hardcoded in run_phase1_scoring.py
                  (the phase 1 pilot set, 13 models).

`--only` further filters whichever source by substring match on hf_id.
`--skip-done` drops any model whose stability JSON already exists under
out_phase1/stability/<slug>.json, so an interrupted parallel sweep can
resume cheaply.

Usage:
    $PY scripts/run_rerun_stability.py
    $PY scripts/run_rerun_stability.py --from-audit --skip-done
    $PY scripts/run_rerun_stability.py --hf-ids 'InstaDeepAI/NTv3_8M_pre,kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16'
    $PY scripts/run_rerun_stability.py --device cuda:3 --n-probes 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from glmap.loaders.dispatch import (  # noqa: E402
    ModelSpec,
    audit_entry_to_spec,
    specs_from_audit,
    specs_from_hf_ids,
)
from scripts.run_phase1_scoring import (  # noqa: E402
    DEFAULT_MODELS,
    _score_ar_one_probe,
    _score_mlm_one_probe,
    _score_model,
)

# _audit_entry_to_spec / _specs_from_audit / _specs_from_hf_ids have been
# extracted to the library (glmap.loaders.dispatch) and imported above.
# The authoritative dispatch table, ModelSpec kw_only dataclass, and
# strict validation now live there; this script delegates to them.


def _pearson(xs: np.ndarray, ys: np.ndarray) -> float:
    """Cross-run Pearson with constant / 0-variance edge cases handled."""
    if xs.size < 2:
        return 1.0 if np.allclose(xs, ys) else 0.0
    a = xs - xs.mean()
    b = ys - ys.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if denom == 0.0:
        return 1.0 if np.allclose(xs, ys) else 0.0
    return float((a * b).sum() / denom)


def _rerun_one_model(
    spec: ModelSpec,
    panel: pd.DataFrame,
    n_probes: int,
    stride: int,
    device: str,
) -> dict:
    """Load `spec`, score the first `n_probes` panel rows twice, compare."""
    t0 = time.time()
    # Reuse _score_model to do the load + first-pass scoring, then immediately
    # rerun only the first N probes (avoids re-implementing loader dispatch).
    sub = panel.head(n_probes).copy()
    _df, loader = _score_model(spec, sub, device=device, stride=stride,
                               progress_every=10**9)
    # _df is the first-pass result on `sub`; pull ell_per_base directly.
    run1 = _df["ell_per_base"].to_numpy(dtype=np.float64)

    # Second pass: score `sub` again with the same (already-loaded) loader.
    if spec.branch == "ar":
        recs = [_score_ar_one_probe(loader, p) for p in sub.to_dict("records")]
    else:
        recs = [_score_mlm_one_probe(loader, p, stride) for p in sub.to_dict("records")]
    run2 = np.array([r["ell_per_base"] for r in recs], dtype=np.float64)

    finite = np.isfinite(run1) & np.isfinite(run2)
    n_finite = int(finite.sum())
    diffs = np.abs(run1[finite] - run2[finite]) if finite.any() else np.array([np.inf])
    pearson_r = _pearson(run1[finite], run2[finite])
    # Gate must require enough successfully-scored probes. NaN==NaN trivially
    # satisfies r=1.0 when every probe errors, which would otherwise look like
    # a PASS — guard against that by demanding ≥80% finite.
    min_finite = max(2, int(np.ceil(0.8 * n_probes)))
    return {
        "hf_id": spec.hf_id,
        "branch": spec.branch,
        "n_probes": int(n_probes),
        "n_finite": n_finite,
        "pearson_r": pearson_r,
        "max_abs_diff": float(diffs.max()),
        "mean_abs_diff": float(diffs.mean()),
        "elapsed_seconds": round(time.time() - t0, 2),
        "passes_gate": bool(n_finite >= min_finite and pearson_r >= 0.95),
    }


def _save_report(out_dir: Path, report: dict) -> None:
    slug = report["hf_id"].replace("/", "__")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{slug}.json").write_text(json.dumps(report, indent=2))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel", type=Path,
                   default=REPO_ROOT / "out_panel/main_panel.parquet",
                   help="Path to the main panel parquet.")
    p.add_argument("--out", type=Path,
                   default=REPO_ROOT / "out_phase1/stability",
                   help="Where to write per-model JSON reports.")
    p.add_argument("--audit", type=Path,
                   default=REPO_ROOT / "data/audits/models.json",
                   help="Path to models audit (used by --from-audit / --hf-ids).")
    p.add_argument("--from-audit", action="store_true",
                   help="Iterate every scorable model in data/audits/models.json.")
    p.add_argument("--hf-ids", type=str, default=None,
                   help="Comma-separated list of exact HF ids; takes precedence "
                   "over --from-audit and DEFAULT_MODELS.")
    p.add_argument("--n-probes", type=int, default=10,
                   help="How many of the first N probes to rerun (default 10).")
    p.add_argument("--stride", type=int, default=6,
                   help="MLM stride k (matches phase_1.md primary k=6).")
    p.add_argument("--device", type=str, default=None,
                   help="cuda:N or cpu. Default auto-picks cuda:0 when available.")
    p.add_argument("--only", type=str, default=None,
                   help="Substring filter on hf_id; only matching models run.")
    p.add_argument("--skip-done", action="store_true",
                   help="Skip models whose stability JSON already exists under --out.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    panel = pd.read_parquet(args.panel)
    if len(panel) == 0:
        raise SystemExit(f"empty panel at {args.panel}")
    n_probes = min(args.n_probes, len(panel))

    if args.device is None:
        try:
            import torch
            args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        except Exception:
            args.device = "cpu"
    print(f"[rerun_stability] panel={args.panel.name} n_probes={n_probes} "
          f"device={args.device}", file=sys.stderr)

    if args.hf_ids:
        hf_ids = [s.strip() for s in args.hf_ids.split(",") if s.strip()]
        specs = specs_from_hf_ids(hf_ids, audit_path=args.audit, strict=False)
        print(f"[rerun_stability] --hf-ids -> {len(specs)} ModelSpec",
              file=sys.stderr)
    elif args.from_audit:
        specs = specs_from_audit(audit_path=args.audit)
        print(f"[rerun_stability] --from-audit {args.audit.name} -> "
              f"{len(specs)} scorable specs", file=sys.stderr)
    else:
        specs = list(DEFAULT_MODELS)

    if args.only:
        needle = args.only.lower()
        specs = [s for s in specs if needle in s.hf_id.lower()]
        print(f"[rerun_stability] --only {args.only!r} -> {len(specs)} models",
              file=sys.stderr)

    if args.skip_done:
        before = len(specs)
        specs = [
            s for s in specs
            if not (args.out / f"{s.hf_id.replace('/', '__')}.json").exists()
        ]
        print(f"[rerun_stability] --skip-done dropped {before - len(specs)} "
              f"already-done models -> {len(specs)} remaining", file=sys.stderr)

    summary_rows: list[dict] = []
    for i, spec in enumerate(specs, start=1):
        print(f"\n[{i}/{len(specs)}] {spec.hf_id} ({spec.branch})",
              file=sys.stderr)
        try:
            report = _rerun_one_model(spec, panel, n_probes,
                                      stride=args.stride, device=args.device)
            _save_report(args.out, report)
            mark = "PASS" if report["passes_gate"] else "FAIL"
            print(f"    {mark}: pearson_r={report['pearson_r']:.6f} "
                  f"max_diff={report['max_abs_diff']:.2e} "
                  f"elapsed={report['elapsed_seconds']}s",
                  file=sys.stderr)
            summary_rows.append(report)
        except Exception as exc:
            err = {
                "hf_id": spec.hf_id, "branch": spec.branch,
                "n_probes": n_probes, "passes_gate": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
            _save_report(args.out, err)
            print(f"    ERROR: {err['error']}", file=sys.stderr)
            summary_rows.append(err)

    df = pd.DataFrame(summary_rows)
    cols = [c for c in (
        "hf_id", "branch", "n_probes", "pearson_r", "max_abs_diff",
        "elapsed_seconds", "passes_gate", "error",
    ) if c in df.columns]
    print("\n=== Summary ===")
    print(df[cols].to_string(index=False))
    n_pass = int(df.get("passes_gate", pd.Series([False] * len(df))).sum())
    print(f"\n{n_pass}/{len(df)} models passed the r ≥ 0.95 gate.")


if __name__ == "__main__":
    main()
