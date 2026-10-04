"""
Unit Tests for RazorEngine & Model Patching
===========================================
"""

import pytest
import torch
from razorkv.config import RazorConfig
from razorkv.cache import RazorKVCache
from razorkv.engine import RazorEngine


class DummyModelConfig:
    def __init__(self, num_hidden_layers=12):
        self.num_hidden_layers = num_hidden_layers


class DummyHFModel:
    def __init__(self, num_layers=12):
        self.config = DummyModelConfig(num_hidden_layers=num_layers)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        return {"input_ids": input_ids, "past_key_values": past_key_values}


def test_engine_create_cache():
    model = DummyHFModel(num_layers=16)
    cache = RazorEngine.create_cache(model)
    assert isinstance(cache, RazorKVCache)
    assert cache.num_layers == 16


def test_engine_patch_and_unpatch():
    model = DummyHFModel(num_layers=8)
    dummy_input = torch.tensor([[1, 2, 3]])

    # Before patch: past_key_values is None if not passed
    res_before = model.prepare_inputs_for_generation(dummy_input)
    assert res_before["past_key_values"] is None

    # Patch model
    RazorEngine.patch_model(model, config=RazorConfig(compression_ratio=0.35))
    res_patched = model.prepare_inputs_for_generation(dummy_input)
    assert isinstance(res_patched["past_key_values"], RazorKVCache)
    assert res_patched["past_key_values"].config.compression_ratio == 0.35

    # Unpatch model
    RazorEngine.unpatch_model(model)
    res_unpatched = model.prepare_inputs_for_generation(dummy_input)
    assert res_unpatched["past_key_values"] is None


def test_engine_vram_savings_calculator():
    savings = RazorEngine.estimate_vram_savings(
        context_len=65536,
        num_layers=32,
        num_kv_heads=8,
        head_dim=128,
        dtype_bytes=2,
        compression_ratio=0.30,
    )
    assert savings["savings_percentage"] == 70.0
    assert savings["dense_mb"] > 0
    assert savings["razorkv_mb"] < savings["dense_mb"]
    assert savings["saved_mb"] == savings["dense_mb"] - savings["razorkv_mb"]
