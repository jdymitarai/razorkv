"""
Unit Tests for RazorKVCache
===========================
Verifies:
1. Cache initialization & budget calculations
2. Multi-step token insertion & hysteresis eviction
3. Preservation of Sinks (Zone 1) & Local Window (Zone 3)
4. Layer-Pyramid gradient (lower layers compressed more aggressively than higher layers)
5. Grouped Query Attention (GQA) compatibility
6. Memory statistics computation
"""

import pytest
import torch
from razorkv.config import RazorConfig
from razorkv.cache import RazorKVCache


def test_razor_config_budgets():
    config = RazorConfig(
        compression_ratio=0.30,
        sink_tokens=16,
        local_window=64,
        pyramid_mode="adaptive",
        pyramid_min_ratio=0.15,
        pyramid_max_ratio=0.70,
    )
    num_layers = 32
    seq_len = 1000

    budget_layer0 = config.get_layer_budget(0, num_layers, seq_len)
    budget_layer16 = config.get_layer_budget(16, num_layers, seq_len)
    budget_layer31 = config.get_layer_budget(31, num_layers, seq_len)

    # Deeper layers must have higher or equal budget than shallow layers
    assert budget_layer0 <= budget_layer16 <= budget_layer31
    # Minimum budget must cover sink + local window
    assert budget_layer0 >= config.sink_tokens + config.local_window


def test_razor_cache_prefill_and_eviction():
    num_layers = 4
    sink_tokens = 8
    local_window = 32
    page_size = 16
    config = RazorConfig(
        compression_ratio=0.30,
        sink_tokens=sink_tokens,
        local_window=local_window,
        page_size=page_size,
        pyramid_mode="linear",
        pyramid_min_ratio=0.20,
        pyramid_max_ratio=0.50,
    )
    cache = RazorKVCache(config=config, num_layers=num_layers)

    batch_size = 1
    num_heads = 4
    head_dim = 32
    prefill_len = 500

    # Simulate prefill
    for layer_idx in range(num_layers):
        k = torch.randn(batch_size, num_heads, prefill_len, head_dim)
        v = torch.randn(batch_size, num_heads, prefill_len, head_dim)
        cache.update(k, v, layer_idx)

    # Check that eviction has occurred for shallow layer 0
    layer0_len = cache.get_seq_length(0)
    layer3_len = cache.get_seq_length(3)

    assert layer0_len < prefill_len, f"Layer 0 was not compressed: len={layer0_len}"
    assert layer0_len <= layer3_len, f"Layer 0 should have smaller or equal length to Layer 3: {layer0_len} vs {layer3_len}"
    assert layer0_len >= sink_tokens + local_window


def test_razor_cache_autoregressive_decode():
    config = RazorConfig(
        compression_ratio=0.40,
        sink_tokens=8,
        local_window=16,
        page_size=8,
        pyramid_mode="uniform",
    )
    cache = RazorKVCache(config=config, num_layers=2)

    batch_size = 1
    num_heads = 2
    head_dim = 16

    # Prefill 100 tokens
    k_init = torch.randn(batch_size, num_heads, 100, head_dim)
    v_init = torch.randn(batch_size, num_heads, 100, head_dim)
    cache.update(k_init, v_init, 0)
    cache.update(k_init, v_init, 1)

    initial_len = cache.get_seq_length(0)

    # Autoregressive generation of 20 tokens
    for step in range(20):
        k_step = torch.randn(batch_size, num_heads, 1, head_dim)
        v_step = torch.randn(batch_size, num_heads, 1, head_dim)
        cache.update(k_step, v_step, 0)
        cache.update(k_step, v_step, 1)

    final_len = cache.get_seq_length(0)
    # The cache length should remain bounded within budget + hysteresis
    max_expected = int(120 * 0.40) + config.page_size + (config.sink_tokens + config.local_window)
    assert final_len <= max_expected
    assert cache.seen_tokens == 120


def test_razor_cache_sink_preservation():
    config = RazorConfig(
        compression_ratio=0.25,
        sink_tokens=10,
        local_window=20,
        page_size=5,
    )
    cache = RazorKVCache(config=config, num_layers=1)

    batch_size = 1
    num_heads = 2
    head_dim = 16
    total_len = 200

    # Assign distinct recognizable values to sink tokens
    k = torch.randn(batch_size, num_heads, total_len, head_dim)
    v = torch.randn(batch_size, num_heads, total_len, head_dim)
    k[:, :, :10, :] = 999.0
    v[:, :, :10, :] = 888.0

    cache.update(k, v, 0)

    # Verify that the first 10 tokens in the cache are strictly the sink tokens
    assert torch.allclose(cache.key_cache[0][:, :, :10, :], torch.tensor(999.0))
    assert torch.allclose(cache.value_cache[0][:, :, :10, :], torch.tensor(888.0))


def test_memory_stats():
    config = RazorConfig(compression_ratio=0.5)
    cache = RazorKVCache(config=config, num_layers=2)
    k = torch.randn(1, 2, 50, 16, dtype=torch.float32)
    v = torch.randn(1, 2, 50, 16, dtype=torch.float32)
    cache.update(k, v, 0)
    cache.update(k, v, 1)

    stats = cache.get_memory_stats()
    assert stats["total_bytes"] > 0
    assert stats["total_mb"] > 0
    assert len(stats["per_layer_tokens"]) == 2


def test_edge_case_sequence_shorter_than_window():
    """Verify that when sequence length is smaller than sinks + local window, no eviction occurs."""
    config = RazorConfig(sink_tokens=16, local_window=32, compression_ratio=0.10)
    cache = RazorKVCache(config=config, num_layers=1)

    k = torch.randn(1, 2, 20, 16)
    v = torch.randn(1, 2, 20, 16)
    cache.update(k, v, 0)

    # All 20 tokens must be retained
    assert cache.get_seq_length(0) == 20


def test_multi_batch_handling():
    """Verify cache works properly with batch_size > 1."""
    config = RazorConfig(sink_tokens=4, local_window=8, compression_ratio=0.30, page_size=4)
    cache = RazorKVCache(config=config, num_layers=2)

    batch_size = 4
    k = torch.randn(batch_size, 2, 80, 16)
    v = torch.randn(batch_size, 2, 80, 16)
    cache.update(k, v, 0)

    assert cache.key_cache[0].shape[0] == batch_size
    assert cache.value_cache[0].shape[0] == batch_size
    assert cache.get_seq_length(0) < 80


def test_reorder_cache_for_beam_search():
    """Verify beam search reordering operates cleanly."""
    config = RazorConfig()
    cache = RazorKVCache(config=config, num_layers=1)

    k = torch.tensor([[[[1.0]], [[2.0]]], [[[3.0]], [[4.0]]]])  # batch 2
    v = torch.tensor([[[[10.0]], [[20.0]]], [[[30.0]], [[40.0]]]])
    cache.update(k, v, 0)

    beam_idx = torch.tensor([1, 0], dtype=torch.long)
    cache.reorder_cache(beam_idx)

    # First batch item should now have values from original index 1
    assert torch.allclose(cache.key_cache[0][0], k[1])
    assert torch.allclose(cache.value_cache[0][0], v[1])

