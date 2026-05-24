"""GLMap: Profiling genomic language models as individuals in a population.

GLMap is a training-free, architecture-agnostic framework for representing
and comparing genomic language models (GLMs) by their likelihood
responses over a fixed panel of DNA sequences. See the project home
page at https://github.com/ai4nucleome/GLMap.

Public API
==========

Matrix pipeline (paper notation: ``V`` raw, ``V_d`` double-centered):

* :func:`clip_lower` — lower-tail clipping of the raw response matrix.
* :func:`double_center` — row-then-column centering.
* :func:`pairwise_distances` — squared Euclidean distances over centered rows.
* :func:`fit_matrix` — clip + double-center pipeline, returning a dict with
  ``V_clipped`` / ``Vd`` / ``D`` and the calibration constants needed for
  :func:`project`.
* :func:`project` — project a new model's raw response row into an existing
  :math:`V_d` space.

Pre-built artefacts:

* :func:`load_panel` / :func:`load_control_panel` — load probe panel parquet.
* :func:`load_matrix` — load a pre-built ``V_AR`` / ``Vd_AR`` / ``D_AR`` /
  ``V_MLM`` / ``Vd_MLM`` / ``D_MLM`` numpy matrix.
* :func:`load_audit` — load the 123-model audit as ``list[dict]``.

Model scoring (lazy heavy imports):

* :class:`ModelSpec`, :func:`audit_entry_to_spec`, :func:`specs_from_audit`,
  :func:`get_loader` — construct loaders from the audit, dispatching to the
  appropriate per-family backend.
* :func:`score_sequence`, :func:`score_panel` — score one or all probes
  through a loaded loader.

Importing :mod:`glmap` does **not** trigger ``import torch`` or
``import transformers``. Heavy dependencies are loaded on demand inside
:func:`get_loader` (and only for the loader family being instantiated).
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # Matrix pipeline
    "clip_lower",
    "double_center",
    "pairwise_distances",
    "fit_matrix",
    "project",
    # Pre-built artefacts
    "load_panel",
    "load_control_panel",
    "load_matrix",
    "load_audit",
    # Model scoring
    "ModelSpec",
    "audit_entry_to_spec",
    "specs_from_audit",
    "get_loader",
    "score_sequence",
    "score_panel",
]


# ---------------------------------------------------------------------------
# Matrix pipeline (V / Vd public names; internal code in glmap.matrices.build
# still uses L / Q for backwards compatibility with the established
# implementation and test suite).
# ---------------------------------------------------------------------------

from glmap.matrices.build import (  # noqa: E402  (deliberate late import for layout)
    clip_lower,
    double_center,
    pairwise_squared_distance as pairwise_distances,
    build_L_Q_D as _build_L_Q_D,
)


def fit_matrix(V, clip_q: float = 0.02) -> dict:
    """Clip + double-center pipeline.

    Parameters
    ----------
    V : np.ndarray
        Raw response matrix, shape ``(n_models, n_probes)``. May contain
        ``NaN`` cells (legitimate mixed-modality un-scored probe classes);
        ``nanmean`` is used throughout so NaNs do not propagate into the
        centering.
    clip_q : float, default 0.02
        Lower-tail quantile threshold for :func:`clip_lower`.

    Returns
    -------
    dict
        Dictionary with keys:

        * ``V_clipped`` — same shape as ``V``, with cells below the
          ``clip_q`` quantile floored to the quantile threshold.
        * ``Vd`` — same shape as ``V``, double-centered (row mean
          subtracted first, then column mean of the row-centered matrix).
        * ``D`` — ``(n_models, n_models)`` pairwise squared Euclidean
          distances between rows of ``Vd``.
        * ``clip_threshold`` — the value at the ``clip_q`` quantile.
        * ``clip_quantile`` — ``clip_q`` (echoed back).
        * ``row_mean``, ``col_mean``, ``grand_mean`` — calibration
          constants. ``col_mean`` is the per-column mean of the
          *row-centered* matrix (not of the raw matrix); ``grand_mean``
          is the mean of all finite cells in the original ``V`` and is
          provided for reporting only — it is not used inside
          :func:`project`.
    """
    out = _build_L_Q_D(V, clip_q=clip_q)
    return {
        "V_clipped":      out["L_clipped"],
        "Vd":             out["Q"],
        "D":              out["D"],
        "clip_threshold": out["clip_threshold"],
        "clip_quantile":  out["clip_quantile"],
        "row_mean":       out["row_mean"],
        "col_mean":       out["col_mean"],
        "grand_mean":     out["grand_mean"],
    }


def project(V_new_row, fit_info: dict):
    """Project a new model's raw response row into an existing :math:`V_d` space.

    Given the calibration constants from a previous :func:`fit_matrix`
    call, project a new model's raw response row into the same
    double-centered representation without refitting the column means.

    The new model's row mean is computed from its own clipped row
    (:func:`numpy.nanmean`); the column mean and clip threshold are
    taken from ``fit_info``.

    Parameters
    ----------
    V_new_row : 1-D np.ndarray, shape ``(n_probes,)``
        Raw response vector for the new model on the same probe panel.
    fit_info : dict
        The dictionary returned by :func:`fit_matrix`.

    Returns
    -------
    1-D np.ndarray, shape ``(n_probes,)``
        The row in :math:`V_d` space, comparable to existing rows of
        ``fit_info["Vd"]``.

    Notes
    -----
    The centering subtracts the new row's mean and the fitted column
    mean. It does **not** add back ``grand_mean`` because
    :func:`double_center` does not use ``grand_mean`` in its centering;
    ``grand_mean`` is reported only for diagnostic purposes.
    """
    import numpy as np
    V_clipped = np.maximum(V_new_row, fit_info["clip_threshold"])
    row_mean = np.nanmean(V_clipped)
    return V_clipped - row_mean - fit_info["col_mean"]


# ---------------------------------------------------------------------------
# Pre-built artefact loaders. Backing files are located by
# :func:`glmap._data_resolver.resolve_data_path` which consults
# $GLMAP_DATA_DIR, package-bundled data, and the repo root in turn.
# ---------------------------------------------------------------------------

_PANEL_PATHS = {
    "main":                 "out_panel/main_panel.parquet",
    "control":              "out_panel/control_panel.parquet",
    "MLM_k1ablation_1000":  "out_panel/MLM_k1ablation_1000_main_panel.parquet",
}

_MATRIX_PATHS = {
    "V_AR":  "out_phase1/matrices/L_AR.npy",
    "Vd_AR": "out_phase1/matrices/Q_AR.npy",
    "D_AR":  "out_phase1/matrices/D_AR.npy",
    "V_MLM":  "out_phase1/matrices/L_MLM.npy",
    "Vd_MLM": "out_phase1/matrices/Q_MLM.npy",
    "D_MLM":  "out_phase1/matrices/D_MLM.npy",
}


def load_panel(name: str = "main"):
    """Load a pre-built probe panel as a :class:`pandas.DataFrame`.

    Parameters
    ----------
    name : str, default ``"main"``
        One of ``"main"`` (10,000-probe biological panel),
        ``"control"`` (10,000-probe synthetic control panel), or
        ``"MLM_k1ablation_1000"`` (1,000-probe stratified subset used for
        the k=1 vs k=6 stride PLL ablation).
    """
    import pandas as pd
    from glmap._data_resolver import resolve_data_path
    try:
        rel = _PANEL_PATHS[name]
    except KeyError:
        raise ValueError(
            f"Unknown panel name {name!r}; expected one of {list(_PANEL_PATHS)}"
        )
    return pd.read_parquet(resolve_data_path(rel))


def load_control_panel():
    """Convenience: same as ``load_panel("control")``."""
    return load_panel("control")


def load_matrix(name: str):
    """Load a pre-built matrix by name.

    Parameters
    ----------
    name : str
        One of ``"V_AR"``, ``"Vd_AR"``, ``"D_AR"``, ``"V_MLM"``,
        ``"Vd_MLM"``, ``"D_MLM"``. The internal storage uses ``L`` and
        ``Q`` filenames (``L_AR.npy``, ``Q_AR.npy``, …); the public
        ``V`` / ``Vd`` names are transparently mapped here.

    Returns
    -------
    np.ndarray
    """
    import numpy as np
    from glmap._data_resolver import resolve_data_path
    try:
        rel = _MATRIX_PATHS[name]
    except KeyError:
        raise ValueError(
            f"Unknown matrix name {name!r}; expected one of {list(_MATRIX_PATHS)}"
        )
    return np.load(resolve_data_path(rel))


def load_audit() -> list[dict]:
    """Load the 123-model audit (``data/audits/models.json``) as a list of dicts."""
    import json
    from glmap._data_resolver import resolve_data_path
    with open(resolve_data_path("data/audits/models.json")) as f:
        payload = json.load(f)
    return payload["models"]


# ---------------------------------------------------------------------------
# Loader dispatch (no heavy imports at module load time; lazy inside
# ``get_loader``).
# ---------------------------------------------------------------------------

from glmap.loaders.dispatch import (  # noqa: E402
    ModelSpec,
    audit_entry_to_spec,
    specs_from_audit,
    get_loader,
)


# ---------------------------------------------------------------------------
# Scoring wrappers. Thin convenience layer on top of each loader's
# ``.score_record()`` method; heavy imports happen inside the loader
# instance, not in this module.
# ---------------------------------------------------------------------------

def score_sequence(loader, sequence: str, stride: int = 6):
    """Score a single sequence with a loaded loader.

    Dispatches to the loader's ``.score_record()`` method, passing
    ``stride`` only for MLM loaders (AR loaders don't take a ``stride``
    argument and would raise ``TypeError``).
    """
    if getattr(loader, "branch", None) == "mlm":
        return loader.score_record(sequence, stride=stride)
    return loader.score_record(sequence)


def score_panel(loader, panel, stride: int = 6):
    """Score every probe in ``panel`` using ``loader``.

    Parameters
    ----------
    loader
        A loaded loader (call :func:`get_loader` or invoke ``.load()``
        on a loader instance first).
    panel : pandas.DataFrame
        Must contain a ``"sequence"`` column. All other columns are
        carried through to the output as per-probe metadata.
    stride : int, default 6
        Stride for MLM stride PLL scoring. Ignored for AR loaders.

    Returns
    -------
    pandas.DataFrame
        One row per probe. Columns: every non-``"sequence"`` column of
        the input panel, followed by the loader's score record fields
        (``sum_log_p``, ``ell_per_base``, ``bpb``, …).
    """
    from dataclasses import asdict, is_dataclass
    import pandas as pd

    is_mlm = getattr(loader, "branch", None) == "mlm"
    meta_columns = [c for c in panel.columns if c != "sequence"]
    rows: list[dict] = []
    for _, row in panel.iterrows():
        sequence = row["sequence"]
        rec = (loader.score_record(sequence, stride=stride)
               if is_mlm
               else loader.score_record(sequence))
        rec_dict = asdict(rec) if is_dataclass(rec) else dict(rec)
        meta = {k: row[k] for k in meta_columns}
        rows.append({**meta, **rec_dict})
    return pd.DataFrame(rows)
