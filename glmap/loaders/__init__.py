"""Model loaders for HF and non-HF gLMs.

A loader is anything that returns an object satisfying GLMLoader (see base.py):
a scalar `score(sequence: str) -> float` per native scoring convention and a
fixed `context_tokens: int`. The audit feasibility table
(data/audits/model_x_length_feasibility.csv) is the contract for which
(model, bp) pairs each loader must support natively.

Non-HF models live here because they cannot be wrapped by AutoModel:
- megaDNA: torch.load .pt + custom 6-token vocab [**, A, T, C, G, #]
- PlasmidGPT: torch.load .pt + Addgene BPE tokenizer json
"""

from __future__ import annotations

__all__: list[str] = []
