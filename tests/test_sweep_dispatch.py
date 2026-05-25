"""Tests for the Stage 4 scoring entry: run_sweep.py --mode scoring +
run_phase1_scoring.py --from-audit.

Verifies the two-mode dispatch shape and the audit-roster loading path
without needing real model weights. The heavy scoring path is exercised
end-to-end by scripts/run_sweep.py + scripts/run_phase1_scoring.py at
Stage 4 launch time.
"""

from __future__ import annotations

import pytest

from scripts.run_sweep import GPUPool, RouteSpec, build_command, resolve_pool_gpu_ids


# ─────────────────────── build_command ───────────────────────

def test_stability_mode_dispatches_to_rerun_stability() -> None:
    route = RouteSpec(env="base", gpus_needed=1)
    args, _env = build_command(
        "lingxusb/megaDNA", route, gpus=[0], n_probes=3, panel=None,
        mode="stability",
    )
    assert args[1].endswith("run_rerun_stability.py")
    assert "--hf-ids" in args
    assert "lingxusb/megaDNA" in args
    assert "--n-probes" in args
    assert "3" in args
    # No --from-audit / --only / --skip-aggregate in stability mode
    assert "--from-audit" not in args
    assert "--only" not in args
    assert "--skip-aggregate" not in args


def test_scoring_mode_dispatches_to_phase1_scoring_from_audit() -> None:
    route = RouteSpec(env="evo2", gpus_needed=4)
    args, env = build_command(
        "arcinstitute/evo2_7b", route, gpus=[0, 1, 2, 3],
        n_probes=10000, panel=None, mode="scoring",
    )
    assert args[1].endswith("run_phase1_scoring.py")
    assert "--from-audit" in args
    # Uses --hf-ids (exact match), NOT --only (substring) — substring would
    # collide with evo2_7b_base / evo2_7b_262k in the audit and cause
    # parallel subprocesses to race on the same parquet.
    assert "--hf-ids" in args
    assert "--only" not in args
    hf_ids_idx = args.index("--hf-ids")
    assert args[hf_ids_idx + 1] == "arcinstitute/evo2_7b"
    assert "--skip-aggregate" in args
    # CUDA_VISIBLE_DEVICES should mask the assigned physical GPUs
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"


def test_scoring_mode_uses_physical_gpu_ids() -> None:
    route = RouteSpec(env="evo2", gpus_needed=4)
    args, env = build_command(
        "arcinstitute/evo2_20b", route, gpus=[0, 5, 6, 7],
        n_probes=10000, panel=None, mode="scoring",
    )
    assert args[1].endswith("run_phase1_scoring.py")
    assert env["CUDA_VISIBLE_DEVICES"] == "0,5,6,7"


def test_gpu_pool_acquires_noncontiguous_ids_in_order() -> None:
    pool = GPUPool([0, 5, 6, 7])
    assert pool.n_gpus == 4
    first = pool.acquire(2)
    assert first == [0, 5]
    second = pool.acquire(2)
    assert second == [6, 7]
    assert pool.acquire(1) is None
    pool.release(first)
    assert pool.acquire(1) == [0]


def test_resolve_pool_gpu_ids_honors_outer_cuda_visible_devices(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,5,6,7")
    ids, source = resolve_pool_gpu_ids(gpu_ids_arg=None, n_gpus_arg=None)
    assert ids == [0, 5, 6, 7]
    assert source == "CUDA_VISIBLE_DEVICES"

    capped, source = resolve_pool_gpu_ids(gpu_ids_arg=None, n_gpus_arg=2)
    assert capped == [0, 5]
    assert source == "CUDA_VISIBLE_DEVICES"


def test_resolve_pool_gpu_ids_explicit_arg_precedes_env(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,5,6,7")
    ids, source = resolve_pool_gpu_ids(gpu_ids_arg="2,3", n_gpus_arg=None)
    assert ids == [2, 3]
    assert source == "--gpu-ids"


def test_scoring_mode_passes_panel_override() -> None:
    route = RouteSpec(env="base", gpus_needed=1)
    args, _env = build_command(
        "lingxusb/megaDNA", route, gpus=[0],
        n_probes=10000, panel="/tmp/custom_panel.parquet", mode="scoring",
    )
    assert "--panel" in args
    panel_idx = args.index("--panel")
    assert args[panel_idx + 1] == "/tmp/custom_panel.parquet"


def test_unknown_mode_raises() -> None:
    route = RouteSpec(env="base", gpus_needed=1)
    with pytest.raises(ValueError, match="unknown mode"):
        build_command("lingxusb/megaDNA", route, [0], 3, None, mode="nonsense")


def test_classify_log_scoring_success_does_not_report_as_unknown(tmp_path) -> None:
    """The scoring worker emits '[done] --skip-aggregate; ...' on success,
    not 'PASS:'. classify_log in scoring mode must recognize this; the
    stability classifier would label it '?'."""
    from scripts.run_sweep import classify_log

    log = tmp_path / "ok.log"
    log.write_text("[ar] loading lingxusb/megaDNA\n"
                   "[ar] lingxusb__megaDNA wrote scores -> ...\n"
                   "[done] --skip-aggregate; skipping matrix build + report.\n")

    # Stability mode (legacy) — no PASS: tag → '?'
    assert classify_log(log, rc=0, mode="stability") == "?"
    # Scoring mode — recognized as DONE
    assert classify_log(log, rc=0, mode="scoring") == "DONE"


def test_classify_log_scoring_traceback_overrides_done(tmp_path) -> None:
    """If the worker printed scoring lines AND a traceback (e.g. crash
    during aggregate), prefer the failure signal."""
    from scripts.run_sweep import classify_log

    log = tmp_path / "mixed.log"
    log.write_text("[ar] wrote scores -> /tmp/scores.parquet\n"
                   "Traceback (most recent call last):\n"
                   "  File ..., line 1, in <module>\n"
                   "RuntimeError: boom\n")
    assert classify_log(log, rc=1, mode="scoring") == "CRASH"
    assert classify_log(log, rc=0, mode="scoring") == "ERR"


def test_scoring_mode_propagates_force_to_child() -> None:
    """Parent --force should append --force to the scoring child's argv,
    otherwise the child would no-op on its own existing parquet."""
    route = RouteSpec(env="base", gpus_needed=1)
    args_noforce, _ = build_command(
        "lingxusb/megaDNA", route, [0], 10000, None, mode="scoring", force=False,
    )
    args_force, _ = build_command(
        "lingxusb/megaDNA", route, [0], 10000, None, mode="scoring", force=True,
    )
    assert "--force" not in args_noforce
    assert "--force" in args_force


def test_scoring_mode_stability_is_unaffected_by_force() -> None:
    """Stability mode does not propagate --force (the worker re-runs
    unconditionally already)."""
    route = RouteSpec(env="base", gpus_needed=1)
    args_force, _ = build_command(
        "lingxusb/megaDNA", route, [0], 3, None, mode="stability", force=True,
    )
    assert "--force" not in args_force


def test_parquet_covers_panel_rejects_nan_rows(tmp_path) -> None:
    """The resume integrity helper must catch the case where a parquet
    has every probe_id (set equality passes) but the sum_log_p column
    is NaN — the per-probe failure path writes such rows."""
    import numpy as np
    import pandas as pd
    from scripts.run_phase1_scoring import parquet_covers_panel

    panel_ids = {f"probe_{i:04d}" for i in range(5)}
    score = tmp_path / "probes.parquet"

    # All-NaN parquet — must be rejected
    df_nan = pd.DataFrame({
        "probe_id": sorted(panel_ids),
        "sum_log_p": [np.nan] * len(panel_ids),
    })
    df_nan.to_parquet(score, index=False)
    ok, reason = parquet_covers_panel(score, panel_ids, n_panel=len(panel_ids))
    assert not ok
    assert "finite" in reason

    # Mixed NaN + finite — must be rejected
    df_mixed = pd.DataFrame({
        "probe_id": sorted(panel_ids),
        "sum_log_p": [-1.0, -2.0, np.nan, -3.0, -4.0],
    })
    df_mixed.to_parquet(score, index=False)
    ok, reason = parquet_covers_panel(score, panel_ids, n_panel=len(panel_ids))
    assert not ok
    assert "4/5" in reason or "finite" in reason

    # All finite — accepted
    df_ok = pd.DataFrame({
        "probe_id": sorted(panel_ids),
        "sum_log_p": [-1.0, -2.0, -1.5, -3.0, -4.0],
    })
    df_ok.to_parquet(score, index=False)
    ok, reason = parquet_covers_panel(score, panel_ids, n_panel=len(panel_ids))
    assert ok
    assert "complete" in reason


def test_parquet_covers_panel_rejects_missing_rows(tmp_path) -> None:
    """probe_id subset of panel — must be rejected (the smoke --max-probes=1
    failure mode)."""
    import pandas as pd
    from scripts.run_phase1_scoring import parquet_covers_panel

    panel_ids = {f"probe_{i:04d}" for i in range(5)}
    score = tmp_path / "probes.parquet"
    df_short = pd.DataFrame({
        "probe_id": ["probe_0000"],
        "sum_log_p": [-1.0],
    })
    df_short.to_parquet(score, index=False)

    ok, reason = parquet_covers_panel(score, panel_ids, n_panel=len(panel_ids))
    assert not ok
    assert "row count" in reason


def test_scoring_mode_uses_exact_match_to_avoid_substring_collisions() -> None:
    """The audit has 15 substring-collision pairs (e.g. evo2_7b matches
    evo2_7b, evo2_7b_base, evo2_7b_262k). The substring-mode --only flag
    would dispatch one subprocess that scores all three; meanwhile the
    other two subprocesses for the same trio would also try to write the
    overlapping parquets. --hf-ids exact match prevents this race."""
    import json
    from pathlib import Path
    REPO_ROOT = Path(__file__).resolve().parents[1]
    audit = json.loads((REPO_ROOT / "data" / "audits" / "models.json").read_text())
    hf_ids = [m["hf_id"] for m in audit["models"]]
    # At least these known collisions exist; the regression guard is that
    # build_command in scoring mode does NOT use a substring-style flag.
    assert "arcinstitute/evo2_7b" in hf_ids
    assert "arcinstitute/evo2_7b_base" in hf_ids
    # Build the scoring command for the prefix and confirm it would not
    # match the longer variants (because --hf-ids is consumed by
    # run_phase1_scoring's exact-match path).
    route = RouteSpec(env="evo2", gpus_needed=4)
    args, _env = build_command(
        "arcinstitute/evo2_7b", route, gpus=[0, 1, 2, 3],
        n_probes=10000, panel=None, mode="scoring",
    )
    assert "--hf-ids" in args
    assert "--only" not in args


# ─────────────────────── run_sweep integrity ───────────────────────

def test_run_sweep_scoring_demotes_done_when_parquet_has_nan_rows(tmp_path, monkeypatch) -> None:
    """If the scoring child exits with rc=0 and the log shows DONE, but its
    written parquet has sum_log_p=NaN rows (per-probe failures), run_sweep
    must demote the status from DONE to PARTIAL — classify_log alone is
    fooled by the worker's '[done]' banner."""
    import numpy as np
    import pandas as pd
    from scripts.run_sweep import run_sweep

    # Build a panel and the would-be parquet for a single fake model
    panel_ids = {f"probe_{i:04d}" for i in range(5)}
    hf_id = "fake/model-with-nan-rows"
    slug = hf_id.replace("/", "__")

    scores_dir = tmp_path / "out_phase1" / "scores" / slug
    scores_dir.mkdir(parents=True)
    df_nan = pd.DataFrame({
        "probe_id": sorted(panel_ids),
        "sum_log_p": [np.nan] * len(panel_ids),
    })
    df_nan.to_parquet(scores_dir / "probes.parquet", index=False)

    # Redirect REPO_ROOT used by the post-exit integrity check to tmp_path
    monkeypatch.setattr("scripts.run_sweep.REPO_ROOT", tmp_path)

    # Fake Popen that "succeeds" (rc=0) and writes the DONE banner to the log
    class _FakePopen:
        returncode = 0
        def __init__(self, args, env=None, stdout=None, stderr=None, cwd=None):
            self._log_path = stdout.name if hasattr(stdout, "name") else None
            if self._log_path:
                with open(self._log_path, "a") as f:
                    f.write("[ar] wrote scores -> ...\n")
                    f.write("[done] --skip-aggregate; skipping matrix build.\n")
        def poll(self): return 0
        def wait(self, timeout=None): return 0
        def terminate(self): pass
        def kill(self): pass

    monkeypatch.setattr("scripts.run_sweep.subprocess.Popen", _FakePopen)

    pool = GPUPool(1)
    log_dir = tmp_path / "logs"
    results = run_sweep(
        tasks=[(hf_id, RouteSpec(env="base", gpus_needed=1))],
        pool=pool,
        n_probes=5,
        panel=None,
        log_dir=log_dir,
        poll_interval=0.01,
        mode="scoring",
        force=True,
        panel_ids=panel_ids,
    )
    assert len(results) == 1
    assert results[0][0] == hf_id
    # The crucial assertion: NaN parquet must NOT be reported as DONE.
    assert results[0][1] == "PARTIAL", (
        f"expected PARTIAL after parquet integrity check; got {results[0][1]}"
    )


def test_run_sweep_unschedulable_task_exits_with_clear_error(tmp_path) -> None:
    """When --n-gpus < a task's gpus_needed, the scheduler loop would silently
    drop the task (its own 'not progress and not running -> break' makes the
    loop exit immediately). main() must fail-fast with a clear message
    enumerating the offending tasks."""
    import json as _json
    from pathlib import Path
    import sys as _sys
    import subprocess as _sp

    # Build a minimal fake audit with only the 8-GPU evo2_40b model
    audit = {
        "models": [
            {"hf_id": "arcinstitute/evo2_40b", "branch": "ar"},
            {"hf_id": "lingxusb/megaDNA", "branch": "ar"},  # 1-GPU
        ]
    }
    audit_path = tmp_path / "models.json"
    audit_path.write_text(_json.dumps(audit))

    REPO_ROOT = Path(__file__).resolve().parents[1]
    proc = _sp.run(
        [_sys.executable, str(REPO_ROOT / "scripts/run_sweep.py"),
         "--audit", str(audit_path),
         "--n-gpus", "4",
         "--dry-run",
         "--force",
         "--mode", "stability"],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    # Must exit non-zero with the offending hf_id named
    assert proc.returncode != 0, proc.stdout + proc.stderr
    combined = proc.stdout + proc.stderr
    assert "arcinstitute/evo2_40b" in combined
    assert "needs 8 GPUs" in combined or "needs 8" in combined


def test_max_gpus_per_model_filters_out_large_models(tmp_path) -> None:
    """--max-gpus-per-model should let a small pool run a subset without
    tripping the unschedulable guard on larger routed models."""
    import json as _json
    from pathlib import Path
    import sys as _sys
    import subprocess as _sp

    audit = {
        "models": [
            {"hf_id": "arcinstitute/evo2_40b", "branch": "ar"},  # 8-GPU
            {"hf_id": "arcinstitute/evo2_20b", "branch": "ar"},  # 4-GPU
            {"hf_id": "lingxusb/megaDNA", "branch": "ar"},       # 1-GPU
        ]
    }
    audit_path = tmp_path / "models.json"
    audit_path.write_text(_json.dumps(audit))

    REPO_ROOT = Path(__file__).resolve().parents[1]
    proc = _sp.run(
        [_sys.executable, str(REPO_ROOT / "scripts/run_sweep.py"),
         "--audit", str(audit_path),
         "--n-gpus", "1",
         "--dry-run",
         "--force",
         "--mode", "stability",
         "--max-gpus-per-model", "1"],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    combined = proc.stdout + proc.stderr
    assert "[dry-run] 1 tasks total" in combined
    assert "skipping 2 model(s)" in combined
    assert "lingxusb/megaDNA" not in combined  # dry-run doesn't list 1-GPU names


# ─────────────────────── --from-audit roster ───────────────────────

def test_phase1_scoring_from_audit_picks_up_full_roster() -> None:
    """run_phase1_scoring.py --from-audit must load the same specs that
    run_rerun_stability.py _specs_from_audit produces, so a Stage 4 sweep
    sees the full 123-model audit set (122 original + 3 HuggingFaceBio/
    Carbon − 2 gena-lm-bigbird-base-sparse{,t2t} excluded as their
    block-sparse attention requires seq_len ≥ 704 tokens, incompatible
    with phase 1 panel probes) instead of the 13-model DEFAULT_MODELS."""
    from pathlib import Path
    from glmap.loaders.dispatch import specs_from_audit

    REPO_ROOT = Path(__file__).resolve().parents[1]
    audit_json = REPO_ROOT / "data" / "audits" / "models.json"
    if not audit_json.exists():
        pytest.skip(f"{audit_json} not built yet")
    specs = specs_from_audit(audit_path=audit_json)
    # Loose lower bound — the audit has 123 candidates, a few are skipped as
    # non-scorable (supervised); 100+ is the realistic floor.
    assert len(specs) >= 100, f"expected ≥ 100 specs, got {len(specs)}"
    # Both branches present
    branches = {s.branch for s in specs}
    assert branches == {"ar", "mlm"}
    # Key special-cased loader_kinds wired up
    loader_kinds = {s.loader_kind for s in specs}
    assert "hf" in loader_kinds            # the default path
    assert "megadna" in loader_kinds       # custom .pt loader
    assert "genslm" in loader_kinds        # codon AR
    assert "generator" in loader_kinds     # k=6 right-truncate (post db0d2db)
