"""Locate bundled / repo-local / env-var data artefacts.

Public helpers in :mod:`glmap` (``load_panel``, ``load_matrix``,
``load_audit``, …) need to find their backing parquet / npy / json
files. The artefacts can live in three places, checked in this order:

1. ``$GLMAP_DATA_DIR``  — explicit override (e.g. when the user has
   downloaded the artefacts to a custom path).
2. ``<package_install_dir>/data/<relative_subpath>`` — when the wheel
   was built with the artefacts bundled as package data.
3. Repo root — walking upward from this file until either the relative
   path resolves or a directory containing ``pyproject.toml`` / ``.git``
   is reached.

If none of the three resolve, ``FileNotFoundError`` is raised with a
message that names all the searched locations.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["resolve_data_path"]


def resolve_data_path(relative_subpath: str) -> Path:
    """Return the absolute :class:`pathlib.Path` to a data artefact.

    Parameters
    ----------
    relative_subpath
        Path **relative to the repo root**, e.g.
        ``"out_panel/main_panel.parquet"`` or
        ``"data/audits/models.json"``.

    Raises
    ------
    FileNotFoundError
        If the file is not found in any of the three searched locations.
    """
    candidates: list[str] = []

    # 1. GLMAP_DATA_DIR environment override
    env_root = os.environ.get("GLMAP_DATA_DIR")
    if env_root:
        candidate = Path(env_root) / relative_subpath
        candidates.append(str(candidate) + "  (from $GLMAP_DATA_DIR)")
        if candidate.exists():
            return candidate

    # 2. Package-bundled data (wheel ships with ``glmap/data/*``).
    package_data = Path(__file__).resolve().parent / "data" / relative_subpath
    candidates.append(str(package_data) + "  (package-bundled)")
    if package_data.exists():
        return package_data

    # 3. Repo-root fallback. Walk upward looking for a marker
    # (``pyproject.toml`` or ``.git``) and stop at the first one.
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / relative_subpath
        if candidate.exists():
            return candidate
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            candidates.append(str(parent / relative_subpath) + "  (repo root)")
            break  # don't keep walking past the repo

    msg_lines = [
        f"Could not locate {relative_subpath!r}.",
        "Searched:",
        *(f"  - {c}" for c in candidates),
        "",
        "Fixes:",
        "  - Set $GLMAP_DATA_DIR to the directory containing the artefacts.",
        "  - Or run from a checkout of the GLMap repository.",
        "  - Or install a wheel built with bundled data artefacts.",
    ]
    raise FileNotFoundError("\n".join(msg_lines))
