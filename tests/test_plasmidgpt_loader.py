"""Unit tests for src.loaders.plasmidgpt.

Cover the loader contract + the legacy-config rebuild helper. The actual
torch.load + forward roundtrip on the HF-cached pretrained_model.pt is a
manual smoke test (network + ~600 MB cache load); CI only checks the
contract.
"""

from __future__ import annotations

from glmap.loaders.base import GLMLoader
from glmap.loaders.plasmidgpt import (
    _GPT2_CONFIG_KEYS_TO_TRANSFER,
    PlasmidGPTLoader,
    _rebuild_gpt2_from_legacy,
)


def test_protocol_conformance() -> None:
    loader = PlasmidGPTLoader()
    assert isinstance(loader, GLMLoader)
    assert loader.hf_id == "lingxusb/PlasmidGPT"
    assert loader.branch == "ar"
    assert loader.context_tokens == 2048


def test_init_accepts_path_overrides(tmp_path) -> None:
    loader = PlasmidGPTLoader(
        weight_path=tmp_path / "weights.pt",
        tokenizer_path=tmp_path / "tok.json",
    )
    assert loader._weight_path_override.name == "weights.pt"
    assert loader._tokenizer_path_override.name == "tok.json"


def test_model_property_raises_before_load() -> None:
    loader = PlasmidGPTLoader()
    import pytest
    with pytest.raises(RuntimeError, match="call load"):
        _ = loader.model
    with pytest.raises(RuntimeError, match="call load"):
        _ = loader.tokenizer


def test_gpt2_config_keys_to_transfer_covers_core_topology() -> None:
    """The rebuild path must at least carry the parameters that shape the
    weight matrices; otherwise load_state_dict will report shape mismatches."""
    must_have = {"vocab_size", "n_positions", "n_embd", "n_layer", "n_head"}
    assert must_have.issubset(set(_GPT2_CONFIG_KEYS_TO_TRANSFER))


def test_rebuild_gpt2_from_legacy_constructs_valid_model() -> None:
    """Construct a small fresh GPT2LMHeadModel, copy state_dict, and verify
    the rebuild helper produces a model with matching state."""
    from transformers import GPT2Config, GPT2LMHeadModel

    class _FakeLegacy:
        pass

    config = GPT2Config(
        vocab_size=128,
        n_positions=64,
        n_embd=32,
        n_layer=2,
        n_head=4,
    )
    real = GPT2LMHeadModel(config)
    fake = _FakeLegacy()
    fake.config = config
    fake.state_dict = real.state_dict

    rebuilt = _rebuild_gpt2_from_legacy(fake)
    assert isinstance(rebuilt, GPT2LMHeadModel)
    assert rebuilt.config.vocab_size == 128
    assert rebuilt.config.n_layer == 2
    # Weight equality on a representative tensor.
    real_sd = real.state_dict()
    new_sd = rebuilt.state_dict()
    shared = set(real_sd) & set(new_sd)
    assert shared, "no shared parameters after rebuild"
    sample_key = next(iter(shared))
    import torch
    assert torch.equal(real_sd[sample_key], new_sd[sample_key])
