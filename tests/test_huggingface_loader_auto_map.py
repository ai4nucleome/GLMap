"""Regression test for the GENA-LM auto_map redirect bug.

Some HF model checkpoints (GENA-LM bert/bigbird variants) ship a config
whose `auto_map` ONLY maps `AutoModel` → a custom `*ForMaskedLM` or
`*ForPreTraining` class, with no explicit `AutoModelForMaskedLM` entry.
Loading via the standard `AutoModelForMaskedLM.from_pretrained(...)`
silently bypasses the custom modeling, instantiating HF's standard
`BertForMaskedLM` instead — which drops the checkpoint's pre-LN
LayerNorm weights and RANDOM-init's the standard LN slots. Logits
are then meaningless (verified empirically: per-token log p differed
by ~80 nats between the two paths on gena-lm-bert-base-t2t-multi).

`HFMaskedLMLoader.load()` now detects this case via
`_auto_map_redirects_to_mlm()` and falls back to `AutoModel` (which
honors `trust_remote_code` + custom modeling). This test pins the
detection contract by mocking `AutoConfig.from_pretrained`.
"""

from __future__ import annotations

from glmap.loaders.huggingface import _auto_map_redirects_to_mlm


class _FakeCfg:
    """Mocks `transformers.PretrainedConfig` with just `.auto_map`."""
    def __init__(self, auto_map):
        self.auto_map = auto_map


def _make_loader_detect(monkeypatch, fake_cfg):
    """Inject a fake AutoConfig.from_pretrained that returns fake_cfg."""
    from glmap.loaders import huggingface as hf_mod

    def fake_from_pretrained(hf_id, **kwargs):
        return fake_cfg
    monkeypatch.setattr(hf_mod.AutoConfig, "from_pretrained", fake_from_pretrained)


def test_redirects_when_automodel_is_for_masked_lm(monkeypatch) -> None:
    """The classic GENA-LM pattern: AutoModel=BertForMaskedLM, no
    AutoModelForMaskedLM entry → must redirect."""
    _make_loader_detect(monkeypatch, _FakeCfg({
        "AutoModel": "modeling_bert.BertForMaskedLM",
    }))
    assert _auto_map_redirects_to_mlm("fake/model", trust_remote_code=True) is True


def test_redirects_when_automodel_is_for_pretraining(monkeypatch) -> None:
    """Two GENA-LM variants (gena-lm-bert-base, gena-lm-bigbird-base-sparse)
    map AutoModel to BertForPreTraining (also wraps MLM head)."""
    _make_loader_detect(monkeypatch, _FakeCfg({
        "AutoModel": "modeling_bert.BertForPreTraining",
    }))
    assert _auto_map_redirects_to_mlm("fake/model", trust_remote_code=True) is True


def test_no_redirect_when_explicit_for_masked_lm_entry(monkeypatch) -> None:
    """If the config explicitly maps AutoModelForMaskedLM to a custom
    class, the standard AutoModelForMaskedLM path WILL honor it via
    trust_remote_code. No redirect needed."""
    _make_loader_detect(monkeypatch, _FakeCfg({
        "AutoModel": "modeling_bert.BertModel",
        "AutoModelForMaskedLM": "modeling_bert.BertForMaskedLM",
    }))
    assert _auto_map_redirects_to_mlm("fake/model", trust_remote_code=True) is False


def test_no_redirect_when_no_auto_map(monkeypatch) -> None:
    """Standard HF models without an auto_map (e.g., moderngena-base,
    nucleotide-transformer-*) should NOT redirect."""
    _make_loader_detect(monkeypatch, _FakeCfg(None))
    assert _auto_map_redirects_to_mlm("fake/model", trust_remote_code=True) is False


def test_no_redirect_when_automodel_is_a_base_class(monkeypatch) -> None:
    """auto_map.AutoModel pointing to a base encoder (BertModel,
    BigBirdModel, …) does NOT carry an MLM head; standard
    AutoModelForMaskedLM is correct."""
    _make_loader_detect(monkeypatch, _FakeCfg({
        "AutoModel": "modeling_bert.BertModel",
    }))
    assert _auto_map_redirects_to_mlm("fake/model", trust_remote_code=True) is False


def test_robust_to_missing_config(monkeypatch) -> None:
    """If AutoConfig.from_pretrained raises (network down, no config.json,
    custom hub error), default to False so we still try the standard path."""
    from glmap.loaders import huggingface as hf_mod

    def boom(hf_id, **kwargs):
        raise OSError("simulated network failure")
    monkeypatch.setattr(hf_mod.AutoConfig, "from_pretrained", boom)
    assert _auto_map_redirects_to_mlm("fake/model", trust_remote_code=True) is False
