"""
Unit Tests for RazorKV Attention Kernels
========================================
"""

import pytest
import torch
import torch.nn.functional as F
from razorkv.kernels.torch_sparse import compacted_paged_sdpa, compute_page_salience_pool


def test_compacted_paged_sdpa_correctness():
    batch_size = 2
    num_heads = 4
    num_kv_heads = 2  # GQA
    q_len = 1
    kv_len = 128
    head_dim = 64

    q = torch.randn(batch_size, num_heads, q_len, head_dim)
    k = torch.randn(batch_size, num_kv_heads, kv_len, head_dim)
    v = torch.randn(batch_size, num_kv_heads, kv_len, head_dim)

    # Output from our kernel
    out = compacted_paged_sdpa(q, k, v)

    # Reference manual SDPA
    k_expanded = k.repeat_interleave(num_heads // num_kv_heads, dim=1)
    v_expanded = v.repeat_interleave(num_heads // num_kv_heads, dim=1)
    ref_out = F.scaled_dot_product_attention(q, k_expanded, v_expanded, is_causal=False)

    assert torch.allclose(out, ref_out, atol=1e-5, rtol=1e-5)


def test_page_salience_pooling():
    salience = torch.tensor([[1.0, 5.0, 2.0, 3.0, 8.0, 2.0, 1.0, 4.0]])  # seq_len = 8
    page_size = 4

    # Max pooling: Page 0 is [1, 5, 2, 3] -> max 5; Page 1 is [8, 2, 1, 4] -> max 8
    paged_max, num_pages = compute_page_salience_pool(salience, page_size=page_size, pool_type="max")
    assert num_pages == 2
    assert torch.allclose(paged_max, torch.tensor([[5.0, 8.0]]))

    # Mean pooling: Page 0 -> 11/4 = 2.75; Page 1 -> 15/4 = 3.75
    paged_mean, num_pages = compute_page_salience_pool(salience, page_size=page_size, pool_type="mean")
    assert torch.allclose(paged_mean, torch.tensor([[2.75, 3.75]]))
