"""Regression tests for the two external-review-driven behaviors:

  1. run_sweep.py --mode embed parent resume calls
     `run_downstream_embed.parquet_complete(path, expected_n)` instead
     of just `path.exists()`. Bug it prevents: a half-written (correct
     row count but mostly-NaN values) or short (truncated mid-write)
     parquet would silently be accepted as "done" and never get
     re-extracted.

  2. run_downstream_classify.py aggregate pass rescans every
     out_phase2/downstream/<model>/<task>/result.json on disk and writes
     out_phase2/matrices/auc_matrix.{npy,meta.json}, even when the
     per-pair fit was scoped by --hf-ids / --tasks. The meta records
     aggregate_scope="all_existing_results" so consumers can detect this.

Neither needs GPU or real embeddings — both work on synthetic parquet
+ JSON fixtures.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PY = "/nvme-data3/yusen/micomamba/bin/python"


# ─────────────────────── parquet_complete contract ───────────────────────

def test_parquet_complete_rejects_missing_file(tmp_path: Path) -> None:
    """run_sweep.py parent calls this for embed resume; missing file must
    flip resume back to "re-run"."""
    from scripts.run_downstream_embed import parquet_complete
    assert parquet_complete(tmp_path / "missing.parquet", expected_n=100) is False


def test_parquet_complete_rejects_wrong_row_count(tmp_path: Path) -> None:
    """Half-written parquets (process killed mid-write OR --max-train
    subsample written under the resume directory) must NOT be accepted
    as complete by the parent resume."""
    from scripts.run_downstream_embed import parquet_complete

    path = tmp_path / "short.parquet"
    df = pd.DataFrame({
        "embed_0": np.ones(50, dtype=np.float32),
        "label": np.zeros(50, dtype=np.int64),
    })
    df.to_parquet(path, index=False)
    assert parquet_complete(path, expected_n=100) is False
    assert parquet_complete(path, expected_n=50) is True


def test_parquet_complete_rejects_mostly_nan_embed(tmp_path: Path) -> None:
    """Parquet where >5% of embed_0 is NaN must not pass the integrity
    check (catches the embed_split fallback where every row was an
    error → all-NaN row written)."""
    from scripts.run_downstream_embed import parquet_complete

    path = tmp_path / "nan_heavy.parquet"
    e0 = np.full(100, np.nan, dtype=np.float32)
    e0[:90] = 1.0     # 90% finite, 10% NaN — should reject (threshold 95%)
    df = pd.DataFrame({"embed_0": e0, "label": np.zeros(100, dtype=np.int64)})
    df.to_parquet(path, index=False)
    assert parquet_complete(path, expected_n=100) is False

    e0[:96] = 1.0     # 96% finite, 4% NaN — should accept
    df = pd.DataFrame({"embed_0": e0, "label": np.zeros(100, dtype=np.int64)})
    df.to_parquet(path, index=False)
    assert parquet_complete(path, expected_n=100) is True


# ─────────────────────── classify aggregate contract ───────────────────────

def test_classify_aggregate_builds_matrix(tmp_path: Path) -> None:
    """run_downstream_classify.py aggregate pass: given a few
    out_phase2/downstream/<model>/<task>/result.json files, produces an
    out_phase2/matrices/auc_matrix.npy with shape (M, T) sorted by
    model + task slug, NaN-filled for missing pairs."""
    out_phase2 = tmp_path / "out_phase2"
    downstream = out_phase2 / "downstream"
    # 3 models × 2 tasks, one pair intentionally missing
    fixtures = {
        ("modelA", "task1"): 0.92,
        ("modelA", "task2"): 0.71,
        ("modelB", "task1"): 0.85,
        # ("modelB", "task2") intentionally absent → NaN cell
        ("modelC", "task1"): 0.65,
        ("modelC", "task2"): 0.50,
    }
    for (m, t), auc in fixtures.items():
        d = downstream / m / t
        d.mkdir(parents=True)
        (d / "result.json").write_text(json.dumps({
            "auc_test": auc, "auc_cv_mean": auc - 0.01, "auc_cv_std": 0.02,
            "best_C": 1.0, "n_train": 100, "n_test": 30,
            "embed_dim": 64, "n_classes": 2, "multiclass": False,
            "classifier": "l2_logistic", "model": m, "task": t,
        }))

    # Need some embedding dir so the pre-aggregate loop's "pairs" discovery
    # doesn't crash (it iterates embeddings dir, can be empty).
    (out_phase2 / "embeddings").mkdir()

    # Invoke the classifier; --hf-ids picks zero models so the per-pair
    # loop is a no-op, but the aggregate pass at the end must still run
    # and pick up the 5 fixture result.json files.
    proc = subprocess.run(
        [PY, str(REPO_ROOT / "scripts/run_downstream_classify.py"),
         "--embeddings-dir", str(out_phase2 / "embeddings"),
         "--out", str(out_phase2),
         "--hf-ids", "no-such-model"],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, f"stderr={proc.stderr}\nstdout={proc.stdout}"

    matrices_dir = out_phase2 / "matrices"
    auc = np.load(matrices_dir / "auc_matrix.npy")
    meta = json.loads((matrices_dir / "auc_matrix_meta.json").read_text())

    assert auc.shape == (3, 2), auc.shape
    assert meta["shape"] == [3, 2]
    assert meta["model_ids"] == ["modelA", "modelB", "modelC"]
    assert meta["task_ids"] == ["task1", "task2"]
    # 5 fixtures → 5 finite + 1 NaN
    assert meta["n_finite"] == 5
    assert meta["n_missing"] == 1
    # Exact value mapping (modelA × task1 = 0.92)
    assert np.isclose(auc[0, 0], 0.92)
    # The intentionally missing pair (modelB × task2) is NaN
    assert np.isnan(auc[1, 1])
    # aggregate_scope is documented in meta so consumers can detect
    # subset-rerun mixing
    assert meta["aggregate_scope"] == "all_existing_results"
    assert meta["command_hf_ids_filter"] == "no-such-model"
    # No corrupt fixtures planted → 0 corrupt skipped
    assert meta["n_corrupt_skipped"] == 0


# ─────────────────────── parquet_complete — all embed dims ──────────────

def test_parquet_complete_catches_nan_in_non_first_dim(tmp_path: Path) -> None:
    """Previously only embed_0 was checked. A parquet where embed_0 is
    fully finite but embed_17 has NaN across many rows would have been
    accepted, then the downstream classifier would silently drop those
    rows. New check matches the classifier's row-drop logic exactly:
    a row is "valid" iff ALL embed dims are finite, then ≥95% rows valid."""
    from scripts.run_downstream_embed import parquet_complete

    path = tmp_path / "nan_in_dim17.parquet"
    n = 100
    cols = {f"embed_{i}": np.ones(n, dtype=np.float32) for i in range(64)}
    # 50/100 rows have NaN at dim 17 only — embed_0 is still 100% finite.
    cols["embed_17"][:50] = np.nan
    cols["label"] = np.zeros(n, dtype=np.int64)
    pd.DataFrame(cols).to_parquet(path, index=False)

    # Old (embed_0-only) check would have returned True. New check sees
    # only 50% of rows are all-finite → reject.
    assert parquet_complete(path, expected_n=n) is False


def test_parquet_complete_accepts_high_finite_fraction(tmp_path: Path) -> None:
    """Sanity: when every row has every dim finite, parquet passes."""
    from scripts.run_downstream_embed import parquet_complete

    path = tmp_path / "clean.parquet"
    n = 100
    cols = {f"embed_{i}": np.ones(n, dtype=np.float32) for i in range(64)}
    cols["label"] = np.zeros(n, dtype=np.int64)
    pd.DataFrame(cols).to_parquet(path, index=False)
    assert parquet_complete(path, expected_n=n) is True


def test_parquet_complete_requires_at_least_one_embed_col(tmp_path: Path) -> None:
    """A parquet with only a `label` column (no embed_*) is invalid."""
    from scripts.run_downstream_embed import parquet_complete

    path = tmp_path / "label_only.parquet"
    pd.DataFrame({"label": np.zeros(10, dtype=np.int64)}).to_parquet(path, index=False)
    assert parquet_complete(path, expected_n=10) is False


# ─────────────────────── classify result.json provenance ────────────────

def _write_fixture_embed_parquet(path: Path, n: int, n_dropped: int, dim: int = 8) -> None:
    """Write an embedding parquet with `n_dropped` NaN rows at the tail."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, dim)).astype(np.float32)
    if n_dropped > 0:
        X[-n_dropped:, :] = np.nan
    y = rng.integers(0, 2, size=n).astype(np.int64)
    cols = {f"embed_{i}": X[:, i] for i in range(dim)}
    cols["label"] = y
    pd.DataFrame(cols).to_parquet(path, index=False)


def test_classify_result_records_drop_counts(tmp_path: Path) -> None:
    """When the embed step left NaN rows, the per-pair result.json must
    record n_train_raw / n_train_dropped / n_test_raw / n_test_dropped
    so silent sample loss is auditable downstream."""
    out_phase2 = tmp_path / "out_phase2"
    emb = out_phase2 / "embeddings" / "modelA" / "task1"
    _write_fixture_embed_parquet(emb / "train.parquet", n=200, n_dropped=20)
    _write_fixture_embed_parquet(emb / "test.parquet",  n=80,  n_dropped=8)

    proc = subprocess.run(
        [PY, str(REPO_ROOT / "scripts/run_downstream_classify.py"),
         "--embeddings-dir", str(out_phase2 / "embeddings"),
         "--out", str(out_phase2)],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, f"stderr={proc.stderr}\nstdout={proc.stdout}"

    result = json.loads(
        (out_phase2 / "downstream" / "modelA" / "task1" / "result.json").read_text()
    )
    assert result["n_train_raw"] == 200
    assert result["n_train_dropped"] == 20
    assert result["n_test_raw"] == 80
    assert result["n_test_dropped"] == 8
    # Post-drop counts match
    assert result["n_train"] == 200 - 20
    assert result["n_test"] == 80 - 8
    # The drop-warning should have fired (drop > 5%)
    assert "drop-warning" in proc.stdout


# ─────────────────────── classify aggregate — corrupt result.json ───────

def test_classify_aggregate_skips_corrupt_and_reports_count(tmp_path: Path) -> None:
    """A malformed result.json (bad JSON, missing auc_test, or non-finite
    auc_test) must NOT silently disappear into "NaN cell". Must be counted
    in meta.n_corrupt_skipped + listed in meta.corrupt_examples and
    flagged in stdout."""
    out_phase2 = tmp_path / "out_phase2"
    downstream = out_phase2 / "downstream"

    # Two good results
    for (m, t, auc) in [("modelA", "task1", 0.81), ("modelA", "task2", 0.74)]:
        d = downstream / m / t
        d.mkdir(parents=True)
        (d / "result.json").write_text(json.dumps({
            "auc_test": auc, "n_train": 100, "n_test": 30,
        }))

    # One malformed JSON (bad syntax)
    bad1 = downstream / "modelB" / "task1"
    bad1.mkdir(parents=True)
    (bad1 / "result.json").write_text("{not valid json")

    # One missing auc_test
    bad2 = downstream / "modelB" / "task2"
    bad2.mkdir(parents=True)
    (bad2 / "result.json").write_text(json.dumps({"n_train": 100}))

    # One out-of-range auc
    bad3 = downstream / "modelC" / "task1"
    bad3.mkdir(parents=True)
    (bad3 / "result.json").write_text(json.dumps({"auc_test": 1.5}))

    (out_phase2 / "embeddings").mkdir()
    proc = subprocess.run(
        [PY, str(REPO_ROOT / "scripts/run_downstream_classify.py"),
         "--embeddings-dir", str(out_phase2 / "embeddings"),
         "--out", str(out_phase2),
         "--hf-ids", "no-such-model"],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, f"stderr={proc.stderr}\nstdout={proc.stdout}"
    assert "3 corrupt/invalid result.json" in proc.stdout

    meta = json.loads(
        (out_phase2 / "matrices" / "auc_matrix_meta.json").read_text()
    )
    assert meta["n_corrupt_skipped"] == 3
    assert len(meta["corrupt_examples"]) == 3
    # All three model dirs had a result.json on disk (even if corrupt),
    # so the axis includes A, B, C — modelB / modelC appear with NaN
    # cells. This is the "preserve all axes" behavior; the matrix's
    # shape now honestly reflects how many models were attempted.
    assert meta["model_ids"] == ["modelA", "modelB", "modelC"]
    assert meta["task_ids"] == ["task1", "task2"]
    # 2 finite cells (modelA × {task1, task2})
    assert meta["n_finite"] == 2
    # 4 NaN cells (modelB × 2 + modelC × 2)
    assert meta["n_missing"] == 4


# ─────────────────── run_sweep embed signature wiring ──────────────

def test_run_sweep_accepts_embed_integrity_kwargs() -> None:
    """Regression for the embed post-exit NameError: run_sweep() must
    accept embed_expected_n + parquet_complete_fn as explicit kwargs
    (rather than relying on main()-local names that don't propagate
    into the run_sweep scope)."""
    import inspect
    from scripts import run_sweep as rs
    sig = inspect.signature(rs.run_sweep)
    assert "embed_expected_n" in sig.parameters, list(sig.parameters)
    assert "parquet_complete_fn" in sig.parameters, list(sig.parameters)
    # Both must default to None so non-embed modes don't break.
    assert sig.parameters["embed_expected_n"].default is None
    assert sig.parameters["parquet_complete_fn"].default is None


# ─────────────────── classify feature-column selection ─────────────

def test_classify_load_split_uses_only_embed_columns(tmp_path: Path) -> None:
    """If the embedding parquet ever grows extra metadata columns
    (sequence_id, source, split, etc.), load_embed_split must keep only
    `embed_*` as features so the classifier doesn't get text columns
    cast to float or extra dims polluting the feature space."""
    from scripts.run_downstream_classify import load_embed_split

    path = tmp_path / "with_extras.parquet"
    n = 50
    df = pd.DataFrame({
        "embed_0": np.random.RandomState(0).randn(n).astype(np.float32),
        "embed_1": np.random.RandomState(1).randn(n).astype(np.float32),
        "embed_2": np.random.RandomState(2).randn(n).astype(np.float32),
        "sequence_id": [f"s{i}" for i in range(n)],   # string metadata
        "source":      ["benchmark_v1"] * n,
        "label":       np.random.RandomState(3).randint(0, 2, size=n).astype(np.int64),
    })
    df.to_parquet(path, index=False)
    X, y, n_raw, n_dropped = load_embed_split(path)
    assert X.shape == (n, 3), X.shape    # only 3 embed columns selected
    assert y.shape == (n,)
    assert n_dropped == 0


# ─────────────────── scoring aggregate defensive read ──────────────

def test_aggregate_safe_read_handles_corrupt_parquet(tmp_path: Path) -> None:
    """_build_branch_matrices' inner _safe_read_sum_log_p must treat a
    corrupt parquet (truncated bytes, missing columns, duplicated
    probe_id) as 'model unavailable' rather than aborting the whole
    aggregate."""
    from glmap.loaders.dispatch import ModelSpec; from scripts.run_phase1_scoring import _build_branch_matrices
    import pandas as pd

    panel = pd.DataFrame({
        "probe_id": [f"p_{i:03d}" for i in range(8)],
    })
    scores_dir = tmp_path / "scores"
    scores_dir.mkdir(parents=True)

    # spec A: clean parquet
    spec_a = ModelSpec(
        hf_id="ok/modelA", branch="ar", context_tokens=1024,
        trust_remote_code=False, is_codon=False, loader_kind="hf",
    )
    (scores_dir / spec_a.slug).mkdir(parents=True)
    pd.DataFrame({
        "probe_id": panel["probe_id"],
        "sum_log_p": np.linspace(-10.0, -1.0, 8),
    }).to_parquet(scores_dir / spec_a.slug / "probes.parquet", index=False)

    # spec B: missing parquet (don't write anything)
    spec_b = ModelSpec(
        hf_id="missing/modelB", branch="ar", context_tokens=1024,
        trust_remote_code=False, is_codon=False, loader_kind="hf",
    )

    # spec C: corrupted (non-parquet bytes)
    spec_c = ModelSpec(
        hf_id="corrupt/modelC", branch="ar", context_tokens=1024,
        trust_remote_code=False, is_codon=False, loader_kind="hf",
    )
    (scores_dir / spec_c.slug).mkdir(parents=True)
    (scores_dir / spec_c.slug / "probes.parquet").write_bytes(b"not a parquet file")

    # spec D: duplicated probe_id
    spec_d = ModelSpec(
        hf_id="dup/modelD", branch="ar", context_tokens=1024,
        trust_remote_code=False, is_codon=False, loader_kind="hf",
    )
    (scores_dir / spec_d.slug).mkdir(parents=True)
    dup_probes = list(panel["probe_id"]) + [panel["probe_id"].iloc[0]]   # 9 rows
    pd.DataFrame({
        "probe_id": dup_probes,
        "sum_log_p": np.linspace(-10.0, -1.0, 9),
    }).to_parquet(scores_dir / spec_d.slug / "probes.parquet", index=False)

    # Should NOT raise; should record diagnostics.
    bm, diag = _build_branch_matrices(
        [spec_a, spec_b, spec_c, spec_d],
        panel=panel,
        scores_dir=scores_dir,
        allow_missing=True,
    )
    assert bm.L.shape[0] == 1, "only spec_a should make it into L under allow_missing"
    assert "ok/modelA" in diag["scored_models"]
    assert "missing/modelB" in diag["missing_models"]
    # corrupt + dup are both surfaced as "corrupt" (per the _safe_read logic)
    corrupt_hfs = [hf for hf, _r in diag["corrupt_models"]]
    assert "corrupt/modelC" in corrupt_hfs
    assert "dup/modelD" in corrupt_hfs


# ─────────────────── classify class-aware gate ─────────────────────

def _write_class_imbalanced_parquet(path: Path, n_pos: int, n_neg: int, dim: int = 4) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n_pos + n_neg, dim)).astype(np.float32)
    y = np.array([1] * n_pos + [0] * n_neg, dtype=np.int64)
    cols = {f"embed_{i}": X[:, i] for i in range(dim)}
    cols["label"] = y
    pd.DataFrame(cols).to_parquet(path, index=False)


def test_classify_class_gate_records_skip_reason(tmp_path: Path) -> None:
    """When the smallest train class has fewer samples than n_cv (5),
    fit would crash mid-CV. Skip explicitly and write the reason into
    result.json so the matrix's NaN cell is traceable."""
    out_phase2 = tmp_path / "out_phase2"
    emb = out_phase2 / "embeddings" / "modelA" / "tinytask"
    # 3 positive + 50 negative in train → smallest class = 3 < n_cv=5
    _write_class_imbalanced_parquet(emb / "train.parquet", n_pos=3, n_neg=50)
    _write_class_imbalanced_parquet(emb / "test.parquet",  n_pos=10, n_neg=10)

    proc = subprocess.run(
        [PY, str(REPO_ROOT / "scripts/run_downstream_classify.py"),
         "--embeddings-dir", str(out_phase2 / "embeddings"),
         "--out", str(out_phase2)],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, f"stderr={proc.stderr}\nstdout={proc.stdout}"
    assert "[class-gate]" in proc.stdout

    result = json.loads(
        (out_phase2 / "downstream" / "modelA" / "tinytask" / "result.json").read_text()
    )
    assert "skip_reason" in result
    assert "smallest train class" in result["skip_reason"]
    assert "auc_test" not in result
    # JSON serializes int keys as strings.
    assert result["train_class_counts"] == {"0": 50, "1": 3}

    # Aggregate must classify this as a deliberate skip, NOT corrupt.
    meta = json.loads(
        (out_phase2 / "matrices" / "auc_matrix_meta.json").read_text()
    )
    assert meta["n_corrupt_skipped"] == 0
    assert meta["n_deliberate_skips"] == 1


# ─────────────────── aggregate axes preserve skip-only rows ─────────────

def test_aggregate_axes_include_skip_only_models(tmp_path: Path) -> None:
    """Regression for the matrix-axis-disappearance bug: if a model has
    ANY result.json under it (auc, skip, or corrupt), it must appear as
    a row in auc_matrix.npy. Otherwise a model whose every task was
    class-gate-skipped would silently vanish from the axis, making the
    matrix shape lie about coverage."""
    out_phase2 = tmp_path / "out_phase2"
    downstream = out_phase2 / "downstream"

    # modelA: clean fits on both tasks
    for (t, auc) in [("task1", 0.82), ("task2", 0.71)]:
        d = downstream / "modelA" / t
        d.mkdir(parents=True)
        (d / "result.json").write_text(json.dumps({"auc_test": auc}))

    # modelB: BOTH tasks deliberately skipped (would have been invisible
    # under the old "only-auc-rows" logic).
    for t in ("task1", "task2"):
        d = downstream / "modelB" / t
        d.mkdir(parents=True)
        (d / "result.json").write_text(json.dumps({
            "skip_reason": "smallest train class has 2 samples < n_cv=5",
            "error_type": "class_gate",
        }))

    # modelC: ONE task auc, one task skipped — verifies mixed-outcome
    # rows produce one finite cell + one NaN cell.
    (downstream / "modelC" / "task1").mkdir(parents=True)
    (downstream / "modelC" / "task1" / "result.json").write_text(
        json.dumps({"auc_test": 0.65})
    )
    (downstream / "modelC" / "task2").mkdir(parents=True)
    (downstream / "modelC" / "task2" / "result.json").write_text(
        json.dumps({
            "skip_reason": "tiny: n_train=10, n_test=2",
            "error_type": "tiny",
        })
    )

    (out_phase2 / "embeddings").mkdir()
    proc = subprocess.run(
        [PY, str(REPO_ROOT / "scripts/run_downstream_classify.py"),
         "--embeddings-dir", str(out_phase2 / "embeddings"),
         "--out", str(out_phase2),
         "--hf-ids", "no-such-model"],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, f"stderr={proc.stderr}\nstdout={proc.stdout}"

    meta = json.loads(
        (out_phase2 / "matrices" / "auc_matrix_meta.json").read_text()
    )
    # All three models MUST appear, including all-skip modelB
    assert meta["model_ids"] == ["modelA", "modelB", "modelC"], meta["model_ids"]
    assert meta["task_ids"] == ["task1", "task2"]
    assert meta["shape"] == [3, 2]
    # 3 finite (A×2 + C×1), 3 NaN (B×2 + C×1)
    assert meta["n_finite"] == 3
    assert meta["n_missing"] == 3
    assert meta["n_deliberate_skips"] == 3   # B×2 + C×1
    assert meta["n_corrupt_skipped"] == 0

    auc = np.load(out_phase2 / "matrices" / "auc_matrix.npy")
    # modelB row entirely NaN
    assert np.all(np.isnan(auc[1, :])), auc[1, :]
    # modelC has one finite (task1=0.65), one NaN (task2 skipped)
    assert np.isclose(auc[2, 0], 0.65)
    assert np.isnan(auc[2, 1])


# ─────────────────── classify multiclass class-set mismatch ─────────────

def _write_multiclass_parquet(
    path: Path, class_counts: dict[int, int], dim: int = 4
) -> None:
    """Write an embedding parquet with the given per-class sample counts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    labels = np.concatenate([
        np.full(n, c, dtype=np.int64) for c, n in class_counts.items()
    ])
    n_total = int(labels.size)
    X = rng.standard_normal((n_total, dim)).astype(np.float32)
    cols = {f"embed_{i}": X[:, i] for i in range(dim)}
    cols["label"] = labels
    pd.DataFrame(cols).to_parquet(path, index=False)


def test_classify_gate_catches_multiclass_test_missing_classes(tmp_path: Path) -> None:
    """Train has classes {0,1,2} with enough samples each; test has only
    {0,1} (the multiclass case where NaN-drop leaves test missing a
    class). roc_auc_score(..., multi_class="ovr") would crash; the gate
    must catch this and write a structured skip_reason instead of
    letting it fall through to [fit fail]."""
    out_phase2 = tmp_path / "out_phase2"
    emb = out_phase2 / "embeddings" / "modelX" / "multitask"
    _write_multiclass_parquet(
        emb / "train.parquet",
        class_counts={0: 30, 1: 30, 2: 30},   # n_cv=5 → all ≥ 5 ✓
    )
    _write_multiclass_parquet(
        emb / "test.parquet",
        class_counts={0: 10, 1: 10},          # class 2 missing
    )

    proc = subprocess.run(
        [PY, str(REPO_ROOT / "scripts/run_downstream_classify.py"),
         "--embeddings-dir", str(out_phase2 / "embeddings"),
         "--out", str(out_phase2)],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, f"stderr={proc.stderr}\nstdout={proc.stdout}"
    assert "[class-gate]" in proc.stdout

    result = json.loads(
        (out_phase2 / "downstream" / "modelX" / "multitask" / "result.json").read_text()
    )
    assert "skip_reason" in result
    assert "train classes [0, 1, 2]" in result["skip_reason"]
    assert "test classes [0, 1]" in result["skip_reason"]
    assert "auc_test" not in result


# ─────────────────── load_embed_split feature column ordering ───────────

def test_load_embed_split_orders_features_by_numeric_suffix(tmp_path: Path) -> None:
    """The feature matrix returned by load_embed_split must have its
    columns ordered embed_0, embed_1, ..., embed_{D-1} regardless of
    the physical column order in the parquet. Otherwise a parquet
    regenerated via pandas concat/merge could land columns shuffled,
    and the classifier would silently use mis-aligned features between
    train and test."""
    from scripts.run_downstream_classify import load_embed_split

    path = tmp_path / "shuffled_cols.parquet"
    n = 30
    # Build a parquet with embed columns DELIBERATELY out of order
    # (10, 2, 5, 0, 1, 3, 4, 6, 7, 8, 9). The value in embed_i is
    # constant `i` so we can verify the post-load column order maps
    # back to the numeric index.
    values_by_idx = {i: np.full(n, float(i), dtype=np.float32) for i in range(11)}
    shuffled_order = [10, 2, 5, 0, 1, 3, 4, 6, 7, 8, 9]
    cols: dict = {}
    for idx in shuffled_order:
        cols[f"embed_{idx}"] = values_by_idx[idx]
    cols["label"] = np.zeros(n, dtype=np.int64)
    pd.DataFrame(cols).to_parquet(path, index=False)

    X, y, n_raw, n_dropped = load_embed_split(path)
    assert X.shape == (n, 11)
    # Every row must have ascending values 0, 1, 2, ..., 10 across
    # columns, proving the sort restored numeric order.
    expected = np.arange(11, dtype=np.float32)
    assert np.allclose(X[0], expected), f"row 0 got {X[0]} expected {expected}"
    assert np.allclose(X[-1], expected)


def test_load_embed_split_rejects_embed_column_without_numeric_suffix(
    tmp_path: Path,
) -> None:
    """A bogus column `embed_garbage` must be flagged loudly rather than
    silently sorted to a random position."""
    from scripts.run_downstream_classify import load_embed_split

    path = tmp_path / "bad_suffix.parquet"
    pd.DataFrame({
        "embed_0": np.ones(10, dtype=np.float32),
        "embed_garbage": np.ones(10, dtype=np.float32),
        "label": np.zeros(10, dtype=np.int64),
    }).to_parquet(path, index=False)

    with pytest.raises(ValueError, match="no numeric suffix"):
        load_embed_split(path)


def test_load_embed_split_rejects_missing_intermediate_dim(tmp_path: Path) -> None:
    """embed_0, embed_2 with NO embed_1 must be rejected: the matrix
    that comes out would lie about its feature dimensionality, and the
    classifier would silently use a non-contiguous feature axis."""
    from scripts.run_downstream_classify import load_embed_split

    path = tmp_path / "missing_dim.parquet"
    pd.DataFrame({
        "embed_0": np.ones(10, dtype=np.float32),
        "embed_2": np.ones(10, dtype=np.float32),   # no embed_1
        "embed_3": np.ones(10, dtype=np.float32),
        "label": np.zeros(10, dtype=np.int64),
    }).to_parquet(path, index=False)

    with pytest.raises(ValueError, match="dense range"):
        load_embed_split(path)


def test_load_embed_split_rejects_duplicate_semantic_suffix(tmp_path: Path) -> None:
    """embed_01 and embed_1 both parse to suffix 1 — duplicate semantic
    dimension. Must reject so the classifier doesn't silently merge or
    pick one arbitrarily."""
    from scripts.run_downstream_classify import load_embed_split

    path = tmp_path / "dup_suffix.parquet"
    pd.DataFrame({
        "embed_0":  np.ones(10, dtype=np.float32),
        "embed_1":  np.ones(10, dtype=np.float32),
        "embed_01": np.ones(10, dtype=np.float32) * 2,   # collides with embed_1
        "embed_2":  np.ones(10, dtype=np.float32),
        "label": np.zeros(10, dtype=np.int64),
    }).to_parquet(path, index=False)

    with pytest.raises(ValueError, match="dense range"):
        load_embed_split(path)


# ─────────────────── parquet_complete ↔ load_embed_split lockstep ───────

def test_parquet_complete_and_load_embed_split_share_schema_contract(
    tmp_path: Path,
) -> None:
    """The sweep resume layer (parquet_complete) and the classify load
    layer (load_embed_split) must enforce IDENTICAL schema contracts.
    Otherwise a malformed parquet could pass resume ("DONE, skip!") and
    then fail classify ("ValueError mid-fit") — the worst kind of bug
    because the sweep summary lies. This test plants three bad
    parquets and verifies both layers reject each one."""
    from scripts.run_downstream_embed import parquet_complete
    from scripts.run_downstream_classify import load_embed_split

    cases = {
        # (filename, columns dict with N=10 rows, what's wrong)
        "missing_intermediate.parquet": {
            "embed_0": np.ones(10, dtype=np.float32),
            "embed_2": np.ones(10, dtype=np.float32),   # no embed_1
            "label":   np.zeros(10, dtype=np.int64),
        },
        "duplicate_semantic.parquet": {
            "embed_0":  np.ones(10, dtype=np.float32),
            "embed_1":  np.ones(10, dtype=np.float32),
            "embed_01": np.ones(10, dtype=np.float32),  # parses to suffix 1
            "label":    np.zeros(10, dtype=np.int64),
        },
        "non_numeric_suffix.parquet": {
            "embed_0":       np.ones(10, dtype=np.float32),
            "embed_garbage": np.ones(10, dtype=np.float32),
            "label":         np.zeros(10, dtype=np.int64),
        },
    }

    for filename, cols in cases.items():
        path = tmp_path / filename
        pd.DataFrame(cols).to_parquet(path, index=False)

        # resume layer rejects
        assert parquet_complete(path, expected_n=10) is False, (
            f"{filename}: parquet_complete falsely accepted malformed schema"
        )
        # load layer rejects with ValueError
        with pytest.raises(ValueError):
            load_embed_split(path)


# ─────────────────── classify cache validation ──────────────────────────

def _make_good_embed_pair(out_phase2: Path, model: str, task: str,
                          n_train: int = 60, n_test: int = 25,
                          dim: int = 8) -> None:
    emb = out_phase2 / "embeddings" / model / task
    emb.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for split, n in (("train", n_train), ("test", n_test)):
        X = rng.standard_normal((n, dim)).astype(np.float32)
        y = rng.integers(0, 2, size=n).astype(np.int64)
        cols = {f"embed_{i}": X[:, i] for i in range(dim)}
        cols["label"] = y
        pd.DataFrame(cols).to_parquet(emb / f"{split}.parquet", index=False)


def test_classify_refits_corrupt_cached_result(tmp_path: Path) -> None:
    """A pre-existing result.json that's corrupt JSON (or missing both
    auc_test and skip_reason) must trigger an automatic refit on the
    next classify run, not silently [skip]."""
    out_phase2 = tmp_path / "out_phase2"
    _make_good_embed_pair(out_phase2, "modelA", "task1")

    # Plant a corrupt cached result.json predating any embed parquet
    bad_path = out_phase2 / "downstream" / "modelA" / "task1" / "result.json"
    bad_path.parent.mkdir(parents=True)
    bad_path.write_text("{not valid json")

    proc = subprocess.run(
        [PY, str(REPO_ROOT / "scripts/run_downstream_classify.py"),
         "--embeddings-dir", str(out_phase2 / "embeddings"),
         "--out", str(out_phase2)],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, f"stderr={proc.stderr}\nstdout={proc.stdout}"
    assert "[refit]" in proc.stdout
    assert "unreadable" in proc.stdout

    # Refit must have OVERWRITTEN the corrupt cache with a valid result
    new_result = json.loads(bad_path.read_text())
    assert "auc_test" in new_result
    assert 0.0 <= new_result["auc_test"] <= 1.0


def test_classify_refits_on_param_fingerprint_change(tmp_path: Path) -> None:
    """A result.json fit under a different --n-cv / --seed / --c-grid
    must be refit automatically, without needing --force."""
    out_phase2 = tmp_path / "out_phase2"
    _make_good_embed_pair(out_phase2, "modelB", "taskX", n_train=200)

    # First run with seed=42
    proc1 = subprocess.run(
        [PY, str(REPO_ROOT / "scripts/run_downstream_classify.py"),
         "--embeddings-dir", str(out_phase2 / "embeddings"),
         "--out", str(out_phase2),
         "--seed", "42"],
        capture_output=True, text=True, timeout=120,
    )
    assert proc1.returncode == 0, proc1.stderr
    result_path = out_phase2 / "downstream" / "modelB" / "taskX" / "result.json"
    first = json.loads(result_path.read_text())
    assert first["fit_params"]["seed"] == 42

    # Same command, same seed → must [skip]
    proc2 = subprocess.run(
        [PY, str(REPO_ROOT / "scripts/run_downstream_classify.py"),
         "--embeddings-dir", str(out_phase2 / "embeddings"),
         "--out", str(out_phase2),
         "--seed", "42"],
        capture_output=True, text=True, timeout=120,
    )
    assert proc2.returncode == 0, proc2.stderr
    assert "[skip]" in proc2.stdout and "[refit]" not in proc2.stdout

    # Different seed → must [refit] with fit_params change reason
    proc3 = subprocess.run(
        [PY, str(REPO_ROOT / "scripts/run_downstream_classify.py"),
         "--embeddings-dir", str(out_phase2 / "embeddings"),
         "--out", str(out_phase2),
         "--seed", "7"],
        capture_output=True, text=True, timeout=120,
    )
    assert proc3.returncode == 0, proc3.stderr
    assert "[refit]" in proc3.stdout
    assert "fit params changed" in proc3.stdout
    third = json.loads(result_path.read_text())
    assert third["fit_params"]["seed"] == 7


def test_classify_load_fail_writes_structured_result(tmp_path: Path) -> None:
    """When load_embed_split raises (e.g. parquet missing label column),
    the script must write a structured result.json with skip_reason +
    error_type=load_fail so the aggregate matrix keeps the pair on
    its axes."""
    out_phase2 = tmp_path / "out_phase2"
    emb = out_phase2 / "embeddings" / "modelC" / "task1"
    emb.mkdir(parents=True)
    # Train parquet missing the `label` column → load_embed_split raises
    pd.DataFrame({
        "embed_0": np.ones(20, dtype=np.float32),
        # NO label column
    }).to_parquet(emb / "train.parquet", index=False)
    pd.DataFrame({
        "embed_0": np.ones(10, dtype=np.float32),
        "label":   np.zeros(10, dtype=np.int64),
    }).to_parquet(emb / "test.parquet", index=False)

    proc = subprocess.run(
        [PY, str(REPO_ROOT / "scripts/run_downstream_classify.py"),
         "--embeddings-dir", str(out_phase2 / "embeddings"),
         "--out", str(out_phase2)],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "[load fail]" in proc.stdout

    result_path = out_phase2 / "downstream" / "modelC" / "task1" / "result.json"
    assert result_path.exists(), "load_fail must still produce a result.json"
    result = json.loads(result_path.read_text())
    assert result["error_type"] == "load_fail"
    assert "load_fail" in result["skip_reason"]


# ─────────────────── scoring aggregate panel-exact contract ─────────────

def test_aggregate_safe_read_rejects_extra_probes(tmp_path: Path) -> None:
    """_safe_read_sum_log_p must reject a parquet whose probe_id set
    is a SUPERSET of the panel (parquet from a different/older panel
    build). Previously reindex would silently drop the extras and
    accept the model; now it must be classified as corrupt so the
    user re-scores against the current panel."""
    from glmap.loaders.dispatch import ModelSpec; from scripts.run_phase1_scoring import _build_branch_matrices
    import pandas as pd

    panel = pd.DataFrame({
        "probe_id": [f"p_{i:03d}" for i in range(5)],
    })
    scores_dir = tmp_path / "scores"
    spec = ModelSpec(
        hf_id="extra/probesModel", branch="ar", context_tokens=1024,
        trust_remote_code=False, is_codon=False, loader_kind="hf",
    )
    (scores_dir / spec.slug).mkdir(parents=True)
    # Parquet has all 5 panel probes PLUS one extra "p_999"
    pd.DataFrame({
        "probe_id": list(panel["probe_id"]) + ["p_999"],
        "sum_log_p": np.linspace(-10.0, -1.0, 6),
    }).to_parquet(scores_dir / spec.slug / "probes.parquet", index=False)

    bm, diag = _build_branch_matrices(
        [spec], panel=panel, scores_dir=scores_dir, allow_missing=True,
    )
    # Under allow_missing, the corrupt spec is dropped → 0 rows in L
    assert bm.L.shape[0] == 0
    corrupt_hfs = [hf for hf, _r in diag["corrupt_models"]]
    assert "extra/probesModel" in corrupt_hfs
    reason = next(r for hf, r in diag["corrupt_models"] if hf == "extra/probesModel")
    assert "row count" in reason or "probe_id set" in reason


# ─────────────── classify refit-then-fit-fail overwrites stale AUC ──────

def test_classify_refit_then_fit_fail_overwrites_stale_auc(tmp_path: Path) -> None:
    """When [refit] is triggered (e.g. cached corrupt JSON or param
    change) and the new fit_and_score raises, the cached result.json
    MUST be overwritten with a structured fit_fail record. Otherwise
    the aggregate would read the stale auc_test from the pre-refit
    cache. Regression for the Medium-High finding from round-13 review."""
    out_phase2 = tmp_path / "out_phase2"
    emb = out_phase2 / "embeddings" / "modelD" / "task1"
    emb.mkdir(parents=True)
    # Train: only ONE class → fit_and_score's roc_auc_score will fail
    # inside CV. NaN-drop won't help. With enough samples to pass the
    # tiny + class_gate gates we need to force class diversity ≥ 2
    # → write data with one class but enough samples to bypass tiny;
    # then class_gate catches it at n_classes_tr<2 → skipped, NOT
    # fit-failed. Hmm — easier to provoke fit_fail with a contrived
    # case: write degenerate features (all zero) of moderate size +
    # both classes balanced → LR converges but proba is degenerate;
    # roc_auc_score wouldn't crash on this. Hard to trigger fit_fail
    # without a deeper mock. Use monkeypatching:
    rng = np.random.default_rng(0)
    n = 200
    X = rng.standard_normal((n, 8)).astype(np.float32)
    y = np.array([0, 1] * (n // 2), dtype=np.int64)
    for split, n_split in (("train", n), ("test", 80)):
        if split == "test":
            X_s = rng.standard_normal((n_split, 8)).astype(np.float32)
            y_s = np.array([0, 1] * (n_split // 2), dtype=np.int64)
        else:
            X_s, y_s = X, y
        cols = {f"embed_{i}": X_s[:, i] for i in range(8)}
        cols["label"] = y_s
        pd.DataFrame(cols).to_parquet(emb / f"{split}.parquet", index=False)

    # Pre-plant a "successful" cached result.json with a stale auc and a
    # fit_params fingerprint that DIFFERS from what we'll invoke with.
    # The cache invalidation will trigger [refit].
    cached_path = out_phase2 / "downstream" / "modelD" / "task1" / "result.json"
    cached_path.parent.mkdir(parents=True)
    cached_path.write_text(json.dumps({
        "auc_test": 0.99,                   # ← stale, must be overwritten
        "fit_params": {"c_grid": [99.0], "n_cv": 99, "seed": 99},
    }))

    # Inject a monkey-patched run that makes fit_and_score raise. We
    # write a tiny wrapper script so subprocess inherits the patch.
    wrapper = tmp_path / "wrapper.py"
    wrapper.write_text(f"""
import sys
sys.path.insert(0, {repr(str(REPO_ROOT))})
import scripts.run_downstream_classify as rc
def _boom(*a, **kw):
    raise RuntimeError("synthetic fit failure")
rc.fit_and_score = _boom
sys.argv = [
    "rc",
    "--embeddings-dir", {repr(str(out_phase2 / 'embeddings'))},
    "--out", {repr(str(out_phase2))},
]
rc.main()
""")
    proc = subprocess.run(
        [PY, str(wrapper)],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, f"stderr={proc.stderr}\nstdout={proc.stdout}"
    assert "[refit]" in proc.stdout
    assert "[fit fail]" in proc.stdout

    # Cached result MUST have been overwritten with a fit_fail record.
    new_result = json.loads(cached_path.read_text())
    assert "auc_test" not in new_result, (
        f"stale auc_test should be gone, got {new_result}"
    )
    assert new_result["error_type"] == "fit_fail"
    assert "synthetic fit failure" in new_result["skip_reason"]
    # The refit_reason should be preserved (audit trail)
    assert "refit_reason" in new_result


# ───── skip records carry fit_params so n_cv change re-attempts ─────

def test_classify_skip_records_include_fit_params(tmp_path: Path) -> None:
    """Tiny / class_gate / load_fail / fit_fail records must all
    include `fit_params`, so changing --n-cv (e.g. from 5 to 3)
    correctly triggers a re-attempt on a previously-class-gated pair
    that would now meet the smaller min-class threshold."""
    out_phase2 = tmp_path / "out_phase2"

    # Pair 1: tiny — train has 15 rows (< 20 threshold)
    _write_class_imbalanced_parquet(
        out_phase2 / "embeddings" / "modelTiny" / "tiny" / "train.parquet",
        n_pos=8, n_neg=7,
    )
    _write_class_imbalanced_parquet(
        out_phase2 / "embeddings" / "modelTiny" / "tiny" / "test.parquet",
        n_pos=5, n_neg=5,
    )
    # Pair 2: class_gate — train has min class = 3, runs at n_cv=5 fails
    _write_class_imbalanced_parquet(
        out_phase2 / "embeddings" / "modelGate" / "gated" / "train.parquet",
        n_pos=3, n_neg=50,
    )
    _write_class_imbalanced_parquet(
        out_phase2 / "embeddings" / "modelGate" / "gated" / "test.parquet",
        n_pos=10, n_neg=10,
    )

    # First run with n_cv=5: both pairs get skipped
    proc1 = subprocess.run(
        [PY, str(REPO_ROOT / "scripts/run_downstream_classify.py"),
         "--embeddings-dir", str(out_phase2 / "embeddings"),
         "--out", str(out_phase2),
         "--n-cv", "5"],
        capture_output=True, text=True, timeout=120,
    )
    assert proc1.returncode == 0
    tiny_path = out_phase2 / "downstream" / "modelTiny" / "tiny" / "result.json"
    gate_path = out_phase2 / "downstream" / "modelGate" / "gated" / "result.json"

    tiny_rec = json.loads(tiny_path.read_text())
    gate_rec = json.loads(gate_path.read_text())
    # Both skip records must carry fit_params with n_cv=5
    assert tiny_rec["fit_params"]["n_cv"] == 5
    assert tiny_rec["error_type"] == "tiny"
    assert gate_rec["fit_params"]["n_cv"] == 5
    assert gate_rec["error_type"] == "class_gate"

    # Second run with n_cv=3: param fingerprint changed → refit attempted.
    # Tiny still blocked by aggregate size; class_gate now passes
    # (smallest class = 3 ≥ n_cv=3) and produces a real AUC.
    proc2 = subprocess.run(
        [PY, str(REPO_ROOT / "scripts/run_downstream_classify.py"),
         "--embeddings-dir", str(out_phase2 / "embeddings"),
         "--out", str(out_phase2),
         "--n-cv", "3"],
        capture_output=True, text=True, timeout=120,
    )
    assert proc2.returncode == 0, proc2.stderr
    assert "[refit]" in proc2.stdout
    assert "fit params changed" in proc2.stdout

    tiny_rec_2 = json.loads(tiny_path.read_text())
    gate_rec_2 = json.loads(gate_path.read_text())
    # Tiny stays a skip but now with n_cv=3 fingerprint
    assert tiny_rec_2["error_type"] == "tiny"
    assert tiny_rec_2["fit_params"]["n_cv"] == 3
    # class_gate now passes → gets a real auc_test
    assert "auc_test" in gate_rec_2, (
        f"class_gate at n_cv=3 should fit; got {gate_rec_2}"
    )
    assert gate_rec_2["fit_params"]["n_cv"] == 3


# ─────── aggregate distinguishes deliberate vs pipeline-error skips ─────

def test_aggregate_separates_deliberate_from_pipeline_error_skips(
    tmp_path: Path,
) -> None:
    """tiny/class_gate result.json records → n_deliberate_skips.
    load_fail/fit_fail records → n_pipeline_errors. Both stay NaN in
    the matrix but they're separately counted in meta so retry-vs-
    audit triage is straightforward."""
    out_phase2 = tmp_path / "out_phase2"
    downstream = out_phase2 / "downstream"

    cases = [
        ("modelA", "task1", "tiny",       "tiny: n_train=10"),
        ("modelA", "task2", "class_gate", "min class < n_cv"),
        ("modelB", "task1", "load_fail",  "load_fail: FileNotFoundError"),
        ("modelB", "task2", "fit_fail",   "fit_fail: RuntimeError: x"),
    ]
    for model, task, err_type, reason in cases:
        d = downstream / model / task
        d.mkdir(parents=True)
        (d / "result.json").write_text(json.dumps({
            "model": model, "task": task,
            "skip_reason": reason,
            "error_type": err_type,
            "fit_params": {"c_grid": [1.0], "n_cv": 5, "seed": 42},
        }))

    (out_phase2 / "embeddings").mkdir()
    proc = subprocess.run(
        [PY, str(REPO_ROOT / "scripts/run_downstream_classify.py"),
         "--embeddings-dir", str(out_phase2 / "embeddings"),
         "--out", str(out_phase2),
         "--hf-ids", "no-such-model"],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    meta = json.loads(
        (out_phase2 / "matrices" / "auc_matrix_meta.json").read_text()
    )
    assert meta["n_deliberate_skips"] == 2     # tiny + class_gate
    assert meta["n_pipeline_errors"] == 2      # load_fail + fit_fail
    assert meta["n_corrupt_skipped"] == 0
    # Print summary contains the new categorization
    assert "deliberate=" in proc.stdout
    assert "pipeline_err=" in proc.stdout


# ─── pipeline-error records auto-retry on next same-param run ───────────

def test_classify_auto_retries_pipeline_error_on_rerun(tmp_path: Path) -> None:
    """A cached load_fail / fit_fail record must be re-attempted on the
    NEXT classify invocation with the same params (default behavior).
    Pipeline errors are usually transient — bug fixed, env restored,
    GPU contention cleared — and silent [skip] would block recovery."""
    out_phase2 = tmp_path / "out_phase2"
    emb = out_phase2 / "embeddings" / "modelR" / "task1"
    _write_fixture_embed_parquet(emb / "train.parquet", n=200, n_dropped=0)
    _write_fixture_embed_parquet(emb / "test.parquet",  n=80,  n_dropped=0)

    # Plant a cached load_fail record from a hypothetical earlier run.
    cached_path = out_phase2 / "downstream" / "modelR" / "task1" / "result.json"
    cached_path.parent.mkdir(parents=True)
    cached_path.write_text(json.dumps({
        "skip_reason": "load_fail: FileNotFoundError: prior bug",
        "error_type": "load_fail",
        "fit_params": {"c_grid": [0.01, 0.1, 1, 10, 100], "n_cv": 5, "seed": 42},
    }))

    # Same params as the cached record — without auto-retry this would
    # [skip]. With the new default, must [refit] and produce a real AUC.
    proc = subprocess.run(
        [PY, str(REPO_ROOT / "scripts/run_downstream_classify.py"),
         "--embeddings-dir", str(out_phase2 / "embeddings"),
         "--out", str(out_phase2)],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "[refit]" in proc.stdout
    assert "auto-retrying" in proc.stdout
    new_result = json.loads(cached_path.read_text())
    assert "auc_test" in new_result, (
        f"pipeline-error record should have been replaced with a fit; got {new_result}"
    )


def test_classify_no_retry_pipeline_errors_keeps_failure(tmp_path: Path) -> None:
    """--no-retry-pipeline-errors keeps an existing load_fail/fit_fail
    record as-is on a SAME-STATE same-params re-run. Use case: model is
    known-broken and the user doesn't want repeated noise in the logs."""
    out_phase2 = tmp_path / "out_phase2"
    emb = out_phase2 / "embeddings" / "modelK" / "task1"
    _write_fixture_embed_parquet(emb / "train.parquet", n=200, n_dropped=0)
    _write_fixture_embed_parquet(emb / "test.parquet",  n=80,  n_dropped=0)

    cached_path = out_phase2 / "downstream" / "modelK" / "task1" / "result.json"
    cached_path.parent.mkdir(parents=True)
    cached_payload = {
        "skip_reason": "fit_fail: RuntimeError: known broken",
        "error_type": "fit_fail",
        "fit_params": {"c_grid": [0.01, 0.1, 1, 10, 100], "n_cv": 5, "seed": 42},
    }
    cached_path.write_text(json.dumps(cached_payload))
    # Make sure the cached result is newer than the embeddings — otherwise
    # the mtime check would trigger a refit independent of pipeline-error
    # logic.
    import os, time
    time.sleep(0.05)
    os.utime(cached_path, None)

    proc = subprocess.run(
        [PY, str(REPO_ROOT / "scripts/run_downstream_classify.py"),
         "--embeddings-dir", str(out_phase2 / "embeddings"),
         "--out", str(out_phase2),
         "--no-retry-pipeline-errors"],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "[skip]" in proc.stdout and "[refit]" not in proc.stdout
    # Cache unchanged
    assert json.loads(cached_path.read_text()) == cached_payload


def test_classify_no_retry_still_refits_on_embed_mtime(tmp_path: Path) -> None:
    """--no-retry-pipeline-errors is a SOFT opt-out: it disables ONLY
    the pipeline-error auto-retry. A newer embedding parquet must STILL
    trigger a refit (the user just re-extracted embeddings, presumably
    fixing the bug that caused the original fit_fail). This documents
    the intentional difference between "freeze auto-retry" and
    "freeze across all future runs"."""
    import os
    import time
    out_phase2 = tmp_path / "out_phase2"
    emb = out_phase2 / "embeddings" / "modelM" / "task1"
    _write_fixture_embed_parquet(emb / "train.parquet", n=200, n_dropped=0)
    _write_fixture_embed_parquet(emb / "test.parquet",  n=80,  n_dropped=0)

    cached_path = out_phase2 / "downstream" / "modelM" / "task1" / "result.json"
    cached_path.parent.mkdir(parents=True)
    cached_path.write_text(json.dumps({
        "skip_reason": "fit_fail: RuntimeError: prior failure",
        "error_type": "fit_fail",
        "fit_params": {"c_grid": [0.01, 0.1, 1, 10, 100], "n_cv": 5, "seed": 42},
    }))

    # Bump the embed parquet mtimes to AFTER the cached result —
    # simulates the user just re-running embed extraction.
    time.sleep(0.05)
    future = time.time() + 60
    os.utime(emb / "train.parquet", (future, future))
    os.utime(emb / "test.parquet",  (future, future))

    proc = subprocess.run(
        [PY, str(REPO_ROOT / "scripts/run_downstream_classify.py"),
         "--embeddings-dir", str(out_phase2 / "embeddings"),
         "--out", str(out_phase2),
         "--no-retry-pipeline-errors"],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    # Mtime check fires BEFORE the pipeline-error opt-out, so even with
    # --no-retry-pipeline-errors the refit happens and produces a real AUC.
    assert "[refit]" in proc.stdout
    assert "embedding parquet newer" in proc.stdout
    new_result = json.loads(cached_path.read_text())
    assert "auc_test" in new_result


def test_classify_deleting_result_json_does_not_freeze_pair(tmp_path: Path) -> None:
    """Documents that the `pairs` loop in main() discovers (model, task)
    pairs from out_phase2/embeddings/, so deleting the cached
    result.json does NOT prevent the pair from being processed — it
    just re-routes through the fresh-fit path. This test pins that
    contract against the (wrong) "delete to freeze" workflow advice
    that earlier docstrings used."""
    out_phase2 = tmp_path / "out_phase2"
    emb = out_phase2 / "embeddings" / "modelDel" / "task1"
    _write_fixture_embed_parquet(emb / "train.parquet", n=200, n_dropped=0)
    _write_fixture_embed_parquet(emb / "test.parquet",  n=80,  n_dropped=0)

    cached_path = out_phase2 / "downstream" / "modelDel" / "task1" / "result.json"
    cached_path.parent.mkdir(parents=True)
    cached_path.write_text(json.dumps({
        "skip_reason": "load_fail: imagined prior",
        "error_type": "load_fail",
    }))

    # "Hard-freeze attempt": delete the cached result.json
    cached_path.unlink()

    proc = subprocess.run(
        [PY, str(REPO_ROOT / "scripts/run_downstream_classify.py"),
         "--embeddings-dir", str(out_phase2 / "embeddings"),
         "--out", str(out_phase2)],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    # The pair is re-discovered from embedding parquets and a fresh
    # fit runs — i.e. the delete did NOT freeze the pair.
    assert "[done]" in proc.stdout, (
        "deleting result.json should NOT freeze the pair; the embedding "
        "parquets are still discoverable so a fresh fit must run"
    )
    assert cached_path.exists(), "fresh fit should have re-created result.json"
    new_result = json.loads(cached_path.read_text())
    assert "auc_test" in new_result


def test_hf_mean_pool_skips_attention_mask_when_unsupported() -> None:
    """Mamba/SSM-based models (Caduceus, PlantCAD2) define forward()
    without an `attention_mask` parameter. The previous embed helper
    unconditionally passed attention_mask and crashed with TypeError on
    every sequence ('all sequences failed embedding'). Now the helper
    introspects forward signature and skips the kwarg when missing."""
    import inspect
    import torch
    import numpy as np
    from glmap.scoring.embeddings import _hf_mean_pool_last_hidden

    # Fake Mamba-style model with no attention_mask in forward()
    class _MambaForward(torch.nn.Module):
        def forward(self, input_ids, output_hidden_states=False):
            # Return a ModelOutput-shaped object with hidden_states
            B, T = input_ids.shape
            class _Out: pass
            o = _Out()
            o.hidden_states = (torch.randn(B, T, 16),)
            return o

    class _FakeTok:
        def __call__(self, s, **kwargs):
            ids = torch.tensor([[1, 2, 3, 4, 5]])
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    class _FakeLoader:
        device = torch.device("cpu")
        tokenizer = _FakeTok()
        model = _MambaForward()
        @property
        def device(self): return torch.device("cpu")

    loader = _FakeLoader()
    # If the gate is wrong, this raises TypeError("got unexpected kwarg 'attention_mask'")
    emb = _hf_mean_pool_last_hidden(loader, "ATCGA")
    assert emb.shape == (16,)
    assert np.isfinite(emb).all()


def test_hf_mean_pool_keeps_attention_mask_when_supported() -> None:
    """Sanity: when forward DOES accept attention_mask, it's still passed
    so the model can mask padding properly."""
    import torch
    import numpy as np
    from glmap.scoring.embeddings import _hf_mean_pool_last_hidden

    saw_mask: dict = {"value": None}

    class _StandardForward(torch.nn.Module):
        def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
            saw_mask["value"] = attention_mask is not None
            B, T = input_ids.shape
            class _Out: pass
            o = _Out()
            o.hidden_states = (torch.randn(B, T, 8),)
            return o

    class _FakeTok:
        def __call__(self, s, **kwargs):
            ids = torch.tensor([[1, 2, 3]])
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    class _FakeLoader:
        device = torch.device("cpu")
        tokenizer = _FakeTok()
        model = _StandardForward()
        @property
        def device(self): return torch.device("cpu")

    loader = _FakeLoader()
    emb = _hf_mean_pool_last_hidden(loader, "ATC")
    assert emb.shape == (8,)
    assert saw_mask["value"] is True, "standard model with attention_mask param must receive it"


def test_classify_no_retry_still_refits_on_param_change(tmp_path: Path) -> None:
    """Companion to the mtime test: changing --n-cv must STILL trigger
    a refit even on a pipeline-error cache with --no-retry-pipeline-errors.
    Soft opt-out: only the auto-retry-on-same-state branch is disabled."""
    out_phase2 = tmp_path / "out_phase2"
    emb = out_phase2 / "embeddings" / "modelN" / "task1"
    _write_fixture_embed_parquet(emb / "train.parquet", n=200, n_dropped=0)
    _write_fixture_embed_parquet(emb / "test.parquet",  n=80,  n_dropped=0)

    cached_path = out_phase2 / "downstream" / "modelN" / "task1" / "result.json"
    cached_path.parent.mkdir(parents=True)
    cached_path.write_text(json.dumps({
        "skip_reason": "load_fail: prior",
        "error_type": "load_fail",
        "fit_params": {"c_grid": [0.01, 0.1, 1, 10, 100], "n_cv": 5, "seed": 42},
    }))
    import os, time
    time.sleep(0.05)
    os.utime(cached_path, None)   # bump cache mtime so embeds aren't "newer"

    # Run with --n-cv 3 (vs cached n_cv=5) + --no-retry-pipeline-errors.
    proc = subprocess.run(
        [PY, str(REPO_ROOT / "scripts/run_downstream_classify.py"),
         "--embeddings-dir", str(out_phase2 / "embeddings"),
         "--out", str(out_phase2),
         "--n-cv", "3",
         "--no-retry-pipeline-errors"],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "[refit]" in proc.stdout
    assert "fit params changed" in proc.stdout


# ─── unknown error_type counted separately in aggregate meta ────────────

def test_aggregate_unknown_error_type_counted_separately(tmp_path: Path) -> None:
    """A skip record with an unknown / missing / typo'd error_type must
    NOT be silently bucketed as deliberate. The aggregate must put it in
    `n_unknown_skips` and emit a WARN so the user notices."""
    out_phase2 = tmp_path / "out_phase2"
    downstream = out_phase2 / "downstream"

    # Plant 4 records spanning all four buckets
    cases = [
        ("modelD", "task1", "tiny",        "tiny: ..."),
        ("modelD", "task2", "class_gate",  "min class < n_cv"),
        ("modelE", "task1", "load_fail",   "load_fail: ..."),
        ("modelE", "task2", "typo_gate",   "future error type"),    # unknown
    ]
    # Plus a record with no error_type at all
    cases.append(("modelF", "task1", None, "legacy skip, no error_type field"))

    for model, task, err_type, reason in cases:
        d = downstream / model / task
        d.mkdir(parents=True)
        payload = {"model": model, "task": task, "skip_reason": reason}
        if err_type is not None:
            payload["error_type"] = err_type
        (d / "result.json").write_text(json.dumps(payload))

    (out_phase2 / "embeddings").mkdir()
    proc = subprocess.run(
        [PY, str(REPO_ROOT / "scripts/run_downstream_classify.py"),
         "--embeddings-dir", str(out_phase2 / "embeddings"),
         "--out", str(out_phase2),
         "--hf-ids", "no-such-model"],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert "unknown/missing error_type" in proc.stdout
    meta = json.loads(
        (out_phase2 / "matrices" / "auc_matrix_meta.json").read_text()
    )
    assert meta["n_deliberate_skips"] == 2          # tiny + class_gate
    assert meta["n_pipeline_errors"] == 1           # load_fail
    assert meta["n_unknown_skips"] == 2             # typo_gate + missing
    assert meta["n_corrupt_skipped"] == 0
    # unknown_skip_examples surface the offending error_type names
    reasons = " ".join(e["reason"] for e in meta["unknown_skip_examples"])
    assert "typo_gate" in reasons
    assert "''" in reasons or "missing" in reasons  # the no-error_type case
