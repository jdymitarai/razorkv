"""
RazorKV Triton Block-Sparse Paged Attention Kernel
==================================================
Fused GPU kernel for block-sparse decoding over paged memory.
Computes online softmax with running maximum and sum across active memory pages.
Provides transparent fallback to PyTorch SDPA when Triton is unavailable.
"""

from typing import Optional
import torch

_TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    pass


if _TRITON_AVAILABLE:
    @triton.jit
    def _triton_paged_decode_kernel(
        Q_ptr,             # [B, H, D]
        K_ptr,             # [B, H_kv, MAX_PAGES, PAGE_SIZE, D]
        V_ptr,             # [B, H_kv, MAX_PAGES, PAGE_SIZE, D]
        PageTable_ptr,     # [B, NUM_ACTIVE_PAGES]
        Out_ptr,           # [B, H, D]
        sm_scale,          # float: 1.0 / sqrt(D)
        stride_qb, stride_qh, stride_qd,
        stride_kb, stride_kh, stride_kp, stride_kt, stride_kd,
        stride_vb, stride_vh, stride_vp, stride_vt, stride_vd,
        stride_ptb, stride_ptp,
        stride_ob, stride_oh, stride_od,
        num_active_pages,
        group_size: tl.constexpr,   # H // H_kv for GQA
        PAGE_SIZE: tl.constexpr,    # e.g. 16
        HEAD_DIM: tl.constexpr,     # e.g. 64 or 128
    ):
        """Triton kernel for fused online softmax over active paged blocks."""
        batch_id = tl.program_id(0)
        head_id = tl.program_id(1)
        kv_head_id = head_id // group_size

        # Offsets for Q
        offs_d = tl.arange(0, HEAD_DIM)
        q_offset = batch_id * stride_qb + head_id * stride_qh + offs_d * stride_qd
        q = tl.load(Q_ptr + q_offset)

        # Running online softmax statistics
        m_i = -float("inf")
        l_i = 0.0
        acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

        # Loop over active pages
        offs_page_t = tl.arange(0, PAGE_SIZE)
        for p_idx in range(num_active_pages):
            # Load physical page index from page table
            page_tbl_offset = batch_id * stride_ptb + p_idx * stride_ptp
            phys_page = tl.load(PageTable_ptr + page_tbl_offset)

            # Pointer to keys in this physical page
            k_base = (
                batch_id * stride_kb
                + kv_head_id * stride_kh
                + phys_page * stride_kp
                + offs_page_t[:, None] * stride_kt
                + offs_d[None, :] * stride_kd
            )
            k = tl.load(K_ptr + k_base)

            # Compute QK^T: [PAGE_SIZE]
            qk = tl.sum(q[None, :] * k, axis=1) * sm_scale

            # Online softmax update
            m_curr = tl.max(qk, axis=0)
            m_next = tl.maximum(m_i, m_curr)
            alpha = tl.exp(m_i - m_next)
            p = tl.exp(qk - m_next)

            # Update running denominator
            l_i = l_i * alpha + tl.sum(p, axis=0)

            # Load values in this physical page
            v_base = (
                batch_id * stride_vb
                + kv_head_id * stride_vh
                + phys_page * stride_vp
                + offs_page_t[:, None] * stride_vt
                + offs_d[None, :] * stride_vd
            )
            v = tl.load(V_ptr + v_base)

            # Accumulate output: [HEAD_DIM]
            p_cast = tl.cast(p, v.dtype)
            acc = acc * alpha + tl.sum(p_cast[:, None] * v, axis=0)
            m_i = m_next

        # Normalize and store final output
        out = acc / l_i
        out_offset = batch_id * stride_ob + head_id * stride_oh + offs_d * stride_od
        tl.store(Out_ptr + out_offset, tl.cast(out, q.dtype))


def triton_paged_sparse_attention(
    query: torch.Tensor,
    key_paged: torch.Tensor,
    value_paged: torch.Tensor,
    page_table: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Executes fused Triton block-sparse decode attention over paged memory.

    Args:
        query: [batch_size, num_heads, 1, head_dim]
        key_paged: [batch_size, num_kv_heads, max_pages, page_size, head_dim]
        value_paged: [batch_size, num_kv_heads, max_pages, page_size, head_dim]
        page_table: [batch_size, num_active_pages] containing indices of active pages
        scale: Scaling factor (default 1.0 / sqrt(head_dim))

    Returns:
        output: [batch_size, num_heads, 1, head_dim]
    """
    if not _TRITON_AVAILABLE or not query.is_cuda:
        # Fallback to PyTorch gather + SDPA
        from razorkv.kernels.torch_sparse import compacted_paged_sdpa
        batch_size, num_heads, _, head_dim = query.shape
        _, num_kv_heads, _, page_size, _ = key_paged.shape

        # Gather active pages into a compacted tensor
        # page_table: [batch, num_active_pages]
        num_active = page_table.shape[1]
        active_tokens = num_active * page_size
        gathered_k = []
        gathered_v = []
        for b in range(batch_size):
            p_indices = page_table[b]
            k_b = key_paged[b, :, p_indices, :, :].reshape(num_kv_heads, active_tokens, head_dim)
            v_b = value_paged[b, :, p_indices, :, :].reshape(num_kv_heads, active_tokens, head_dim)
            gathered_k.append(k_b)
            gathered_v.append(v_b)

        k_compact = torch.stack(gathered_k, dim=0)
        v_compact = torch.stack(gathered_v, dim=0)
        return compacted_paged_sdpa(query, k_compact, v_compact, scaling=scale)

    # Triton Execution Path
    batch_size, num_heads, q_len, head_dim = query.shape
    assert q_len == 1, "Triton paged decode kernel is optimized for single-token autoregressive decoding (q_len == 1)."

    _, num_kv_heads, max_pages, page_size, _ = key_paged.shape
    num_active_pages = page_table.shape[1]
    group_size = num_heads // num_kv_heads

    if scale is None:
        scale = 1.0 / (head_dim ** 0.5)

    q_squeezed = query.squeeze(2).contiguous()  # [batch, num_heads, head_dim]
    out = torch.empty_like(q_squeezed)

    grid = (batch_size, num_heads)
    _triton_paged_decode_kernel[grid](
        q_squeezed,
        key_paged,
        value_paged,
        page_table,
        out,
        scale,
        q_squeezed.stride(0), q_squeezed.stride(1), q_squeezed.stride(2),
        key_paged.stride(0), key_paged.stride(1), key_paged.stride(2), key_paged.stride(3), key_paged.stride(4),
        value_paged.stride(0), value_paged.stride(1), value_paged.stride(2), value_paged.stride(3), value_paged.stride(4),
        page_table.stride(0), page_table.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        num_active_pages=num_active_pages,
        group_size=group_size,
        PAGE_SIZE=page_size,
        HEAD_DIM=head_dim,
    )

    return out.unsqueeze(2)


def is_triton_available() -> bool:
    """Returns True if Triton is installed and supported on the current device."""
    return _TRITON_AVAILABLE and torch.cuda.is_available()
