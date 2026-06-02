"""Probe panel utilities (library side).

To load the prebuilt panel (already shipped with this repository):

    >>> import glmap
    >>> panel = glmap.load_panel()      # → (10000, 11) DataFrame
"""

from __future__ import annotations

from glmap.panel.composition import (
    DINUC_INDEX,
    DINUC_ORDER,
    TRINUC_INDEX,
    TRINUC_ORDER,
    dinuc_vec,
    gc_fraction,
    gc_stratify_bin,
    trinuc_vec,
)

__all__ = [
    "DINUC_INDEX",
    "DINUC_ORDER",
    "TRINUC_INDEX",
    "TRINUC_ORDER",
    "dinuc_vec",
    "gc_fraction",
    "gc_stratify_bin",
    "trinuc_vec",
]
