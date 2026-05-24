"""Control panel construction (10K probes, three diagnostic subsets).

Three disjoint synthetic subsets, none of which enter the main sequence-
likelihood matrix:

  random_ACGT           3500  GC-stratified pseudo-random ACGT, 7 GC bins × 500
  dinucleotide_shuffled 3500  Altschul-Erickson shuffle drawn from each of the
                              14 main-panel functional elements (250 each)
  motif_spiked          3000  random ACGT backbone + 1 of 5 HOCOMOCO-style
                              core motifs inserted at center, 5 motifs × 600

Same ProbeRow schema as main_panel.ProbeRow:
  probe_id            "ctrl_<subset>_00001"
  sequence            uppercase ACGT
  length_bp           multi-tier in [200, 1024]
  functional_element  one of "ctrl_random_ACGT" / "ctrl_dinuc_shuffled" /
                      "ctrl_motif_spiked"
  species             "synthetic"
  source              descriptive provenance
  label_source        "control"
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from .composition import dinuc_vec, gc_fraction, trinuc_vec
from .main_panel import ProbeRow


# 5 widely-known core motifs (placeholder until HOCOMOCO PWMs are wired in).
# Each is a consensus core (forward strand); used to probe whether models
# pick up TF-binding motif signal beyond background composition.
DEFAULT_MOTIFS: tuple[str, ...] = (
    "TATAAAA",   # TATA-box-like (length 7)
    "CACGTG",    # E-box / MYC / USF (length 6)
    "CCAAT",     # CAAT-box / NF-Y (length 5)
    "GGGCGG",    # GC-box / SP1 (length 6)
    "CCCTC",     # CTCF-like core (length 5)
)

# Length tiers matching the main panel's range so control LL is comparable
# at each tier (rather than averaging across mismatched lengths).
DEFAULT_LENGTH_TIERS: tuple[int, ...] = (200, 300, 500, 1000)


# ─────────────────────── helpers ───────────────────────

def _make_row(
    probe_id: str,
    sequence: str,
    functional_element: str,
    source: str,
) -> ProbeRow:
    return ProbeRow(
        probe_id=probe_id,
        sequence=sequence,
        length_bp=len(sequence),
        functional_element=functional_element,
        species_group="synthetic",   # all 3 control subsets are synthetic
        species="synthetic",
        GC_content=gc_fraction(sequence),
        dinuc_vec=dinuc_vec(sequence),
        trinuc_vec=trinuc_vec(sequence),
        source=source,
        label_source="control",
    )


def _random_acgt_with_gc(length: int, gc: float, rng: random.Random) -> str:
    """Sample length bp i.i.d. with P(G or C) = gc."""
    p_gc_half = gc / 2.0
    p_at_half = (1.0 - gc) / 2.0
    bases = "ACGT"
    weights = [p_at_half, p_gc_half, p_gc_half, p_at_half]
    return "".join(rng.choices(bases, weights=weights, k=length))


# ─────────────────────── 1. random_ACGT ───────────────────────

def build_random_acgt(
    n: int = 3500,
    seed: int = 42,
    gc_bins: Sequence[float] = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80),
    length_tiers: Sequence[int] = DEFAULT_LENGTH_TIERS,
) -> list[ProbeRow]:
    """7 GC bins × ~500 probes each, lengths cycled across DEFAULT_LENGTH_TIERS.

    GC bins span 0.20-0.80 (mid-range plant + human + virus + yeast all covered).
    Each (GC bin × length tier) cell ends up with ~n/(7*4) = ~125 probes.
    """
    rng = random.Random(seed)
    rows: list[ProbeRow] = []
    per_bin = n // len(gc_bins)
    extra = n - per_bin * len(gc_bins)
    for bin_idx, gc in enumerate(gc_bins):
        n_this_bin = per_bin + (1 if bin_idx < extra else 0)
        for i in range(n_this_bin):
            length = length_tiers[i % len(length_tiers)]
            seq = _random_acgt_with_gc(length, gc, rng)
            probe_id = f"ctrl_random_ACGT_{len(rows) + 1:05d}"
            source = f"synthetic::gc={gc:.2f}::len={length}"
            rows.append(_make_row(probe_id, seq, "ctrl_random_ACGT", source))
    return rows


# ─────────────────────── 2. dinucleotide_shuffled ───────────────────────

def _ae_shuffle(seq: str, rng: random.Random) -> str:
    """Altschul-Erickson dinucleotide shuffle (exact preservation).

    Preserves the exact 16-D dinucleotide frequency vector while randomizing
    higher-order structure. Builds a directed multigraph where edges are
    dinucleotide transitions and finds an Eulerian trail starting at seq[0]
    and ending at seq[-1].

    Two-tier strategy:
      1. **Rejection sampling** (fast path): randomly permute outgoing edges
         at each node, walk greedily; on stuck-walk, retry. 200 attempts
         brings per-sequence failure probability below 1e-12 for typical DNA.
      2. **Saved-edge fallback** (guaranteed correctness): if rejection
         sampling somehow exhausts 200 attempts, fall through to the
         Fitch-Altschul-Erickson "save-last-edge" construction that
         deterministically yields an Eulerian trail.

    Either tier produces a sequence whose 16-D dinucleotide multiset
    matches the input exactly. The earlier character-shuffle fallback
    (which broke 2-mer preservation) has been removed.
    """
    if len(seq) < 2:
        return seq

    # Build adjacency: edges[base] = list of next bases (multiset)
    edges: dict[str, list[str]] = {b: [] for b in "ACGT"}
    for i in range(len(seq) - 1):
        edges[seq[i]].append(seq[i + 1])

    start_base = seq[0]
    end_base = seq[-1]

    # Tier 1: rejection sampling, 200 attempts
    for _ in range(200):
        adj = {b: list(es) for b, es in edges.items()}
        for b in adj:
            rng.shuffle(adj[b])
        out = [start_base]
        cur = start_base
        ok = True
        for _ in range(len(seq) - 1):
            if not adj[cur]:
                ok = False
                break
            nxt = adj[cur].pop()
            out.append(nxt)
            cur = nxt
        if ok and len(out) == len(seq):
            return "".join(out)

    # Tier 2: guaranteed-success save-last-edge construction
    return _ae_shuffle_saved_edge(seq, edges, start_base, end_base, rng)


def _ae_shuffle_saved_edge(
    seq: str,
    edges: dict[str, list[str]],
    start_base: str,
    end_base: str,
    rng: random.Random,
) -> str:
    """AE Eulerian-trail construction with explicit saved-edge guarantee.

    For each non-terminal node v ≠ end_base, save one outgoing edge that
    lies on a known path to end_base — this ensures the greedy walk can
    always reach end_base. Use reverse BFS from end_base to pick saved
    edges; among multiple successors that reach end_base, choose randomly.

    Returns a sequence whose 16-D dinucleotide multiset matches the input
    exactly. Raises RuntimeError only if the graph is genuinely Eulerian-
    trail-disconnected, which means the input sequence itself violates
    Eulerian connectivity (should not happen for ACGT sequences ≥ 2 bp).
    """
    # Reverse adjacency: reverse[v] = unique nodes u such that some edge u→v exists.
    reverse: dict[str, set[str]] = {b: set() for b in "ACGT"}
    for u in "ACGT":
        for v in edges[u]:
            reverse[v].add(u)

    # BFS from end_base backwards; reaches_end[v] = a chosen successor that
    # eventually reaches end_base (the saved-edge target for node v).
    reaches_end: dict[str, str | None] = {end_base: None}
    queue = [end_base]
    while queue:
        v = queue.pop(0)
        for u in reverse[v]:
            if u not in reaches_end and any(w == v for w in edges[u]):
                reaches_end[u] = v
                queue.append(u)

    # Build saved_edge: for each non-terminal node u with ≥ 1 outgoing edge,
    # the saved edge points to reaches_end[u]. We remove ONE occurrence
    # from edges[u] (multiset removal) and shuffle the rest.
    saved_edge: dict[str, str] = {}
    for u in "ACGT":
        if u == end_base or not edges[u]:
            continue
        target = reaches_end.get(u)
        if target is None:
            # u has outgoing edges but BFS didn't reach it from end_base —
            # this would mean u can't reach end at all. For a valid input
            # sequence this never happens.
            continue
        edges[u].remove(target)
        saved_edge[u] = target

    # Shuffle remaining (non-saved) edges at each node
    for u in "ACGT":
        rng.shuffle(edges[u])

    # Walk: at each node, prefer non-saved edges; use saved edge last.
    out = [start_base]
    cur = start_base
    for _ in range(len(seq) - 1):
        if edges[cur]:
            nxt = edges[cur].pop()
        elif cur in saved_edge:
            nxt = saved_edge.pop(cur)
        else:
            raise RuntimeError(
                f"AE shuffle: stuck at {cur} after {len(out)} of {len(seq)} bases; "
                "input sequence has Eulerian-disconnected dinucleotide graph"
            )
        out.append(nxt)
        cur = nxt

    if len(out) != len(seq):
        raise RuntimeError(
            f"AE shuffle: emitted {len(out)} bases, expected {len(seq)}"
        )
    return "".join(out)


def build_dinuc_shuffled(
    main_panel_df: pd.DataFrame,
    n_total: int = 3500,
    seed: int = 42,
) -> list[ProbeRow]:
    """Draw probes from each of the main-panel's 14 functional elements and
    Altschul-Erickson-shuffle each. Per-element budget = ceil(n_total /
    n_elements) so the control's length × element-class distribution matches
    the main panel within ±1.
    """
    rng = random.Random(seed)
    elements = sorted(main_panel_df["functional_element"].unique())
    n_per_elem = n_total // len(elements)
    extra = n_total - n_per_elem * len(elements)
    rows: list[ProbeRow] = []
    for ei, elem in enumerate(elements):
        budget = n_per_elem + (1 if ei < extra else 0)
        pool = main_panel_df[main_panel_df["functional_element"] == elem]
        if len(pool) == 0:
            continue
        if len(pool) >= budget:
            sample_idxs = rng.sample(list(pool.index), budget)
        else:
            # not enough in pool — take all and resample with replacement
            sample_idxs = list(pool.index) + rng.choices(list(pool.index), k=budget - len(pool))
        for idx in sample_idxs:
            src_row = main_panel_df.loc[idx]
            shuffled = _ae_shuffle(src_row["sequence"], rng)
            probe_id = f"ctrl_dinuc_shuffled_{len(rows) + 1:05d}"
            source = f"shuffled_from::{src_row['probe_id']}"
            rows.append(_make_row(probe_id, shuffled, "ctrl_dinuc_shuffled", source))
    return rows


# ─────────────────────── 3. motif_spiked ───────────────────────

def build_motif_spiked(
    n: int = 3000,
    seed: int = 42,
    motifs: Sequence[str] = DEFAULT_MOTIFS,
    length_tiers: Sequence[int] = DEFAULT_LENGTH_TIERS,
    background_gc: float = 0.5,
) -> list[ProbeRow]:
    """Random ACGT backbone (GC=0.5) with one motif inserted at the center.

    5 motifs × 4 length tiers × ~150 probes each = ~3000 probes.
    """
    rng = random.Random(seed)
    rows: list[ProbeRow] = []
    per_motif_per_tier = n // (len(motifs) * len(length_tiers))
    for motif in motifs:
        for length in length_tiers:
            if length < len(motif) + 4:
                # motif doesn't fit comfortably; skip this combination
                continue
            for _ in range(per_motif_per_tier):
                bg = _random_acgt_with_gc(length, background_gc, rng)
                mid = (length - len(motif)) // 2
                spiked = bg[:mid] + motif + bg[mid + len(motif):]
                probe_id = f"ctrl_motif_spiked_{len(rows) + 1:05d}"
                source = f"synthetic::motif={motif}::len={length}::bg_gc=0.5"
                rows.append(_make_row(probe_id, spiked, "ctrl_motif_spiked", source))
    return rows


# ─────────────────────── orchestration ───────────────────────

def build_control_panel(
    main_panel_df: pd.DataFrame,
    seed: int = 42,
    n_random_acgt: int = 3500,
    n_dinuc_shuffled: int = 3500,
    n_motif_spiked: int = 3000,
) -> pd.DataFrame:
    """Assemble the full 10K control panel.

    Three sub-builders are called with independent sub-seeds derived from
    the main seed so each subset is reproducible.
    """
    rng = random.Random(seed)
    seed_random = rng.randint(0, 2**31 - 1)
    seed_dinuc = rng.randint(0, 2**31 - 1)
    seed_motif = rng.randint(0, 2**31 - 1)
    rows: list[ProbeRow] = []
    rows.extend(build_random_acgt(n=n_random_acgt, seed=seed_random))
    rows.extend(build_dinuc_shuffled(main_panel_df, n_total=n_dinuc_shuffled, seed=seed_dinuc))
    rows.extend(build_motif_spiked(n=n_motif_spiked, seed=seed_motif))
    return pd.DataFrame([r.__dict__ for r in rows])


def write_control_outputs(out_dir: Path, ctrl_df: pd.DataFrame, seed: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ctrl_df.to_parquet(out_dir / "control_panel.parquet", index=False)
    manifest = {
        "seed": seed,
        "total_probes": len(ctrl_df),
        "subsets": {
            subset: int((ctrl_df["functional_element"] == subset).sum())
            for subset in ["ctrl_random_ACGT", "ctrl_dinuc_shuffled", "ctrl_motif_spiked"]
        },
        "length_distribution": ctrl_df.groupby("functional_element")["length_bp"]
            .describe().round(2).to_dict(orient="index"),
        "gc_by_subset": ctrl_df.groupby("functional_element")["GC_content"]
            .describe().round(3).to_dict(orient="index"),
    }
    with (out_dir / "control_manifest.json").open("w") as h:
        json.dump(manifest, h, indent=2)


__all__ = [
    "DEFAULT_MOTIFS",
    "DEFAULT_LENGTH_TIERS",
    "build_random_acgt",
    "build_dinuc_shuffled",
    "build_motif_spiked",
    "build_control_panel",
    "write_control_outputs",
]
