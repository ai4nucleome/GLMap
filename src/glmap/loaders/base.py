"""Loader protocol every gLM adapter must satisfy.

Defines the minimum interface phase 0 / phase 1 / phase 2 scoring code expects
from a model adapter. Implementations live alongside this file (one module per
non-HF model + a single huggingface.py for AutoModel-loadable ones).

The phase_1.md "打分协议" section is the source of truth for what a "scalar
score" means per branch (ModelMap convention: raw nats, no length norm,
no sign flip; cells are negative log-likelihoods entering L as-is):

  AR  branch: sum_t log p(x_t | x_<t)               (forward once)
  MLM branch: stride pseudo-log-likelihood          (primary stride k=6)

Adapters return the **raw native sum_log_p** per the convention above; the
matrix layer (`src/matrices/build.py`) applies floor-clip + double-center
without further normalization. `ell_per_base` and `bpb` are computed by
`src/scoring/` for per-probe diagnostic reporting only — they do not enter
the L / Q / D matrices.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

Branch = Literal["ar", "mlm"]


@runtime_checkable
class GLMLoader(Protocol):
    """Minimal contract every gLM adapter must satisfy."""

    #: Stable identifier matching models/download_models_list.txt (e.g.
    #: "lingxusb/megaDNA"). Used as the primary key across the audit
    #: manifests and the model_x_length_feasibility table.
    hf_id: str

    #: Either "ar" or "mlm"; must match
    #: data/audits/model_context_manifest.csv `branch`.
    branch: Branch

    #: Max input length in tokens. Must match
    #: data/audits/model_context_manifest.csv `context_limit_tokens` (the
    #: override-aware value, not the raw config field).
    context_tokens: int

    def load(self) -> None:
        """Load weights into memory; idempotent. Must be called before score()."""

    def score(self, sequence: str) -> float:
        """Return the native scalar per phase_1.md "打分协议".

        For AR loaders: log p(x) / n_bases (forward only).
        For MLM loaders: stride PLL(x) / n_bases.

        The caller (src/scoring/) is responsible for sign flipping to NLL and
        for any bpb conversion. Reverse-complement sanity check is done by
        calling score() twice (forward + RC) at the caller side, not here.
        """
