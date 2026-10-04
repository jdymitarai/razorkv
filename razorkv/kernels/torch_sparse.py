"""
RazorKV PyTorch Optimized Attention Kernels
===========================================
Highly vectorized pure PyTorch kernels for block-sparse attention and compacted cache forward.
Leverages torch.nn.functional.scaled_dot_product_attention (cuDNN / FlashAttention under the hood)
for zero-overhead execution on both CUDA and CPU.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn.functional as F


def compacted_paged_sdpa(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    scaling: Optional[float] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
) -> torch.Tensor:
    """Executes scaled dot-product attention over compacted/evicted KV cache tensors.

    Args:
        query_states: [batch_size, num_heads, q_len, head_dim]
        key_states:   [batch_size, num_kv_heads, kv_len, head_dim]
        value_states: [batch_size, num_kv_heads, kv_len, head_dim]
        attention_mask: Optional attention mask
        scaling: Optional scaling factor (default: 1.0 / sqrt(head_dim))
        dropout_p: Dropout probability
        is_causal: Whether causal masking applies (Note: for single-token decode, is_causal=False is exact)

    Returns:
        attn_output: [batch_size, num_heads, q_len, head_dim]
    """
    batch_size, num_heads, q_len, head_dim = query_states.shape
    _, num_kv_heads, kv_len, _ = key_states.shape

    # Handle Grouped Query Attention (GQA) head repeat
    if num_heads != num_kv_heads:
        group_size = num_heads // num_kv_heads
        key_states = key_states.repeat_interleave(group_size, dim=1)
        value_states = value_states.repeat_interleave(group_size, dim=1)

    # For autoregressive single-token decode (q_len == 1), all cached keys are strictly in the past,
    # so is_causal must be False.
    effective_is_causal = is_causal if q_len > 1 else False

    # Execute PyTorch native FlashAttention / SDPA kernel
    output = F.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attention_mask,
        dropout_p=dropout_p,
        is_causal=effective_is_causal,
        scale=scaling,
    )
    return output


def compute_page_salience_pool(
    salience_scores: torch.Tensor,
    page_size: int,
    pool_type: str = "max",
) -> Tuple[torch.Tensor, int]:
    """Pools token salience into page-level scores.

    Args:
        salience_scores: [batch_size, seq_len]
        page_size: number of tokens per page
        pool_type: 'max' or 'mean'

    Returns:
        paged_scores: [batch_size, num_pages]
        num_pages: int
    """
    batch_size, seq_len = salience_scores.shape
    num_pages = seq_len // page_size
    if num_pages == 0:
        return salience_scores.amax(dim=-1, keepdim=True), 1

    aligned_len = num_pages * page_size
    reshaped = salience_scores[:, :aligned_len].view(batch_size, num_pages, page_size)

    if pool_type == "max":
        return reshaped.amax(dim=-1), num_pages
    elif pool_type == "mean":
        return reshaped.mean(dim=-1), num_pages
    else:
        raise ValueError(f"Unknown pool_type: {pool_type}")
