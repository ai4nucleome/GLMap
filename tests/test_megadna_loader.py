"""Unit tests for src.loaders.megadna.

These cover the loader contract + the encode_sequence helper. Actually
loading the 580 MB megaDNA_phage_145M.pt weight (and pinning
MEGABYTE_pytorch==0.2.1 to deserialize it) is out of scope for fast CI —
treat the real load + score as a smoke test that runs manually.
"""

from __future__ import annotations

import pytest
import torch

from glmap.loaders.base import GLMLoader
from glmap.loaders.megadna import (
    MEGADNA_NUCLEOTIDE_TO_TOKEN,
    PAD_ID,
    MegaDNALoader,
    encode_sequence,
)


def test_protocol_conformance() -> None:
    loader = MegaDNALoader()
    assert isinstance(loader, GLMLoader)
    assert loader.hf_id == "lingxusb/megaDNA"
    assert loader.branch == "ar"
    assert loader.context_tokens == 131072


def test_encode_sequence_basic_acgt() -> None:
    out = encode_sequence("ACGT")
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 4)
    assert out.dtype == torch.long
    expected = torch.tensor(
        [[MEGADNA_NUCLEOTIDE_TO_TOKEN[b] for b in "ACGT"]], dtype=torch.long
    )
    assert torch.equal(out, expected)


def test_encode_sequence_uppercases() -> None:
    assert torch.equal(encode_sequence("acgt"), encode_sequence("ACGT"))


def test_encode_sequence_pads_unknown_default() -> None:
    out = encode_sequence("ACNGT")
    assert int(out[0, 2]) == PAD_ID


def test_encode_sequence_strict_raises_on_unknown() -> None:
    with pytest.raises(ValueError, match="non-ACGT"):
        encode_sequence("ACNGT", pad_unknown=False)


def test_score_record_requires_non_empty_sequence(tmp_path) -> None:
    loader = MegaDNALoader(weight_path=tmp_path / "missing.pt")
    with pytest.raises(ValueError, match="empty sequence"):
        loader.score_record("")


def test_load_raises_when_weight_file_missing(tmp_path, monkeypatch) -> None:
    # When the weight is absent, load() first tries to auto-download it from
    # HF. Force that fallback to fail so we deterministically exercise the
    # "weight file missing" path regardless of cache / network / whether the
    # megaDNA package is installed (otherwise a machine that can fetch the
    # weight would proceed to the package-import step instead).
    import huggingface_hub

    def _no_download(*args, **kwargs):
        raise RuntimeError("download disabled for test")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _no_download)
    loader = MegaDNALoader(weight_path=tmp_path / "absent.pt")
    with pytest.raises(FileNotFoundError, match="weight file missing"):
        loader.load()
