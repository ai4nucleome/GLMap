#!/usr/bin/env python3
"""Model-side audit.

Reads `models/download_models_list.txt`, fetches each model's HF config and
tokenizer_config, and emits a structured record per model.

Output:
    data/audits/models.json   — list[dict], machine-readable, primary artifact
    data/audits/models.md     — human-readable summary + table

Each model record schema:

    {
      "id":                   int,         # 1..N, stable, follows download list order
      "hf_id":                str,         # full HF repo path (e.g. "InstaDeepAI/NTv3_650M")
      "organization":         str,         # HF org/user (e.g. "InstaDeepAI")
      "family":               str,         # coarse family label (NT, DNABERT, ...)
      "architecture":         str,         # transformer_encoder / hyena / ...
      "training_paradigm":    str,         # ntp / mlm / supervised
      "branch":               str,         # ar_or_generative / mlm_or_encoder / ...
      "param_count":          int | null,
      "context_tokens":       int | null,
      "context_bp":           int | null,
      "tokenizer_type":       str,
      "score_protocol":       str
    }

Usage:
    $PY scripts/audits/models.py
    $PY scripts/audits/models.py --skip-hf       # offline, no config.json fetches
    $PY scripts/audits/models.py --max-models 5  # debug
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from common import (
    AUDIT_DIR,
    CONTEXT_OVERRIDES_PATH,
    MODELS_LIST_PATH,
    PARAM_OVERRIDES_PATH,
    choose_context,
    clean_model_list,
    estimate_context_bp,
    fetch_hf_json,
    infer_architecture,
    infer_branch,
    infer_context_from_name,
    infer_family,
    infer_param_count,
    infer_score_protocol,
    infer_tokenizer_type,
    load_context_overrides,
    load_param_overrides,
    repo_organization,
    training_paradigm,
    write_json,
    write_markdown,
)


def build_record(
    hf_id: str,
    model_id: int,
    context_overrides: dict[str, dict[str, Any]],
    param_overrides: dict[str, int],
    skip_hf: bool,
) -> dict[str, Any]:
    config = None if skip_hf else fetch_hf_json(hf_id, "config.json")
    tokenizer_config = None if skip_hf else fetch_hf_json(hf_id, "tokenizer_config.json")

    family = infer_family(hf_id)
    branch = infer_branch(hf_id)
    paradigm = training_paradigm(branch)
    architecture, _arch_source = infer_architecture(hf_id, config)
    tokenizer_type = infer_tokenizer_type(hf_id, config)
    score_protocol = infer_score_protocol(hf_id, branch)

    name_ctx, name_src = infer_context_from_name(hf_id)
    context_tokens, _ctx_source = choose_context(
        config,
        tokenizer_config,
        name_ctx,
        name_src,
        override=context_overrides.get(hf_id),
    )
    context_bp = estimate_context_bp(context_tokens, tokenizer_type, family=family)
    param_count, _param_source = infer_param_count(hf_id, config, param_overrides)

    return {
        "id": model_id,
        "hf_id": hf_id,
        "organization": repo_organization(hf_id),
        "family": family,
        "architecture": architecture,
        "training_paradigm": paradigm,
        "branch": branch,
        "param_count": param_count,
        "context_tokens": context_tokens,
        "context_bp": context_bp,
        "tokenizer_type": tokenizer_type,
        "score_protocol": score_protocol,
    }


def render_markdown(records: list[dict[str, Any]], out_path: Path) -> None:
    """Render the model list as a Markdown table with summary stats up top."""
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover
        pd = None  # type: ignore[assignment]

    n = len(records)
    branch_counts: dict[str, int] = {}
    paradigm_counts: dict[str, int] = {}
    arch_counts: dict[str, int] = {}
    family_counts: dict[str, int] = {}
    organization_counts: dict[str, int] = {}
    param_known = 0
    ctx_known = 0
    ctx_bp_known = 0
    for r in records:
        branch_counts[r["branch"]] = branch_counts.get(r["branch"], 0) + 1
        paradigm_counts[r["training_paradigm"]] = (
            paradigm_counts.get(r["training_paradigm"], 0) + 1
        )
        arch_counts[r["architecture"]] = arch_counts.get(r["architecture"], 0) + 1
        family_counts[r["family"]] = family_counts.get(r["family"], 0) + 1
        organization = r.get("organization") or "(unknown)"
        organization_counts[organization] = organization_counts.get(organization, 0) + 1
        if r["param_count"] is not None:
            param_known += 1
        if r["context_tokens"] is not None:
            ctx_known += 1
        if r["context_bp"] is not None:
            ctx_bp_known += 1

    def _fmt(d: dict[str, int]) -> str:
        return ", ".join(
            f"{k}={v}" for k, v in sorted(d.items(), key=lambda kv: -kv[1])
        )

    summary = [
        f"Total models: **{n}**",
        f"Training paradigm: {_fmt(paradigm_counts)}",
        f"Branch: {_fmt(branch_counts)}",
        f"Architecture: {_fmt(arch_counts)}",
        f"Families: **{len(family_counts)}** — {_fmt(family_counts)}",
        f"Organizations: **{len(organization_counts)}** — {_fmt(organization_counts)}",
        f"Coverage: param_count **{param_known}/{n}**, "
        f"context_tokens **{ctx_known}/{n}**, context_bp **{ctx_bp_known}/{n}**",
        "Source JSON: `data/audits/models.json`.",
    ]

    sections: list[tuple[str, str]] = []
    if pd is not None:
        df = pd.DataFrame(records)
        sections.append(("Full table", df.to_markdown(index=False)))

    write_markdown(out_path, "Model audit", summary, sections=sections)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-list", type=Path, default=MODELS_LIST_PATH)
    parser.add_argument("--out-dir", type=Path, default=AUDIT_DIR)
    parser.add_argument("--context-overrides", type=Path, default=CONTEXT_OVERRIDES_PATH)
    parser.add_argument("--param-overrides", type=Path, default=PARAM_OVERRIDES_PATH)
    parser.add_argument(
        "--skip-hf",
        action="store_true",
        help="Skip HF config / tokenizer_config fetches (offline mode).",
    )
    parser.add_argument(
        "--max-models",
        type=int,
        default=None,
        help="Truncate the model list (debug).",
    )
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "20")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "10")
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    models = clean_model_list(args.models_list)
    if args.max_models is not None:
        models = models[: args.max_models]
    print(f"[models] {len(models)} entries", file=sys.stderr)

    context_overrides = load_context_overrides(args.context_overrides)
    param_overrides = load_param_overrides(args.param_overrides)
    print(
        f"[models] overrides: context={len(context_overrides)}, "
        f"param={len(param_overrides)}",
        file=sys.stderr,
    )

    records: list[dict[str, Any]] = []
    for idx, hf_id in enumerate(models, start=1):
        if idx == 1 or idx % 10 == 0 or idx == len(models):
            print(f"[models] {idx}/{len(models)}: {hf_id}", file=sys.stderr)
        records.append(
            build_record(
                hf_id=hf_id,
                model_id=idx,
                context_overrides=context_overrides,
                param_overrides=param_overrides,
                skip_hf=args.skip_hf,
            )
        )

    json_path = args.out_dir / "models.json"
    md_path = args.out_dir / "models.md"
    write_json(
        {
            "generated_by": "scripts/audits/models.py",
            "n_models": len(records),
            "models": records,
        },
        json_path,
    )
    render_markdown(records, md_path)
    print(
        f"[models] wrote {json_path} + {md_path.name} ({len(records)} models)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
