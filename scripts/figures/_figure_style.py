#!/usr/bin/env python3
"""Shared figure styling for the GLMap figure suite.

Defines the common color palette, functional-element class order / colors,
per-branch model shades, and matplotlib rcParams imported by the
``fig*`` / ``figS*`` / ``panel_composition`` figure scripts. Import-only —
this module produces no figure of its own (hence the ``_`` prefix).

(Formerly ``phase1_main_figure.py``, which also built a legacy phase-1
summary figure off the now-retired ``run_phase1_analysis.py`` /
``glmap.analysis`` pipeline. Only the shared styling constants remain.)
"""

from __future__ import annotations


# ------------------------------ style ------------------------------ #

PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "blue_light": "#7CA9D6",
    "red_strong": "#B64342",
    "red_secondary": "#E9A6A1",
    "red_light": "#F6CFCB",
    "neutral": "#CFCECE",
    "highlight": "#FFD700",
    "teal": "#42949E",
    "violet": "#9A4D8E",
    "green_3": "#8BCF8B",
}

# 14 categorical colors for the 14 functional element_ids of the panel.
# Grouped by species_group so legends read cleanly: Human (blues/oranges)
# → Plant (greens) → Fungi (purples) → Virus (reds).
CLASS_ORDER = (
    # Human (4 elements)
    "promoter",
    "enhancer",
    "splice_donor",
    "splice_acceptor",
    # Plant (6 elements)
    "chromatin_access",
    "polyA",
    "lncRNA",
    "nascent_RNA",
    "splicing_plant_donor",
    "splicing_plant_acceptor",
    # Fungi (2 elements)
    "yeast_genome",
    "fungi_genome",
    # Virus (2 elements)
    "virus_variants",
    "virus_species",
)
CLASS_COLORS = {
    # Human — blue family
    "promoter":                  "#1f77b4",
    "enhancer":                  "#aec7e8",
    "splice_donor":              "#ff7f0e",
    "splice_acceptor":           "#ffbb78",
    # Plant — green family
    "chromatin_access":          "#2ca02c",
    "polyA":                     "#98df8a",
    "lncRNA":                    "#8c564b",
    "nascent_RNA":               "#c49c94",
    "splicing_plant_donor":      "#bcbd22",
    "splicing_plant_acceptor":   "#dbdb8d",
    # Fungi — purple family
    "yeast_genome":              "#9467bd",
    "fungi_genome":              "#c5b0d5",
    # Virus — red family
    "virus_variants":            "#d62728",
    "virus_species":             "#ff9896",
}

AR_MODEL_SHADES = [
    PALETTE["blue_light"], PALETTE["blue_secondary"], PALETTE["blue_main"],
    PALETTE["teal"], PALETTE["violet"],
]
MLM_MODEL_SHADES = [
    PALETTE["red_light"], PALETTE["red_secondary"], PALETTE["red_strong"],
    PALETTE["highlight"], PALETTE["green_3"],
]

RCPARAMS = {
    "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 11,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 1.5,
    "legend.frameon": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,           # TrueType so editors can edit text
    "ps.fonttype": 42,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
}
