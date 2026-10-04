"""
RazorKVCache: Dynamic Layer-Pyramid Paged Key-Value Cache Engine
================================================================
Drop-in replacement for HuggingFace Transformers Cache / DynamicCache.
Cross-compatible with Transformers 4.x, 5.x, and standalone PyTorch.

Implements:
1. Tri-Zone memory partitioning (Sinks + Dynamic Haystack + Local Window)
2. Layer-pyramid budget distribution (85% reduction in shallow layers, high retention in deep layers)
3. Page-level chunked eviction to maintain memory coalescing and match vLLM BlockManager
4. Query-Key Resonance tracking with exponential decay
5. Amortized hysteresis eviction for zero-overhead autoregressive decode
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import torch

try:
    from transformers.cache_utils import DynamicCache, Cache
    BaseCacheClass = DynamicCache
except ImportError:
    try:
        from transformers.cache_utils import Cache
        BaseCacheClass = Cache
    except ImportError:
        class BaseCacheClass:
            """Fallback base class when transformers is not installed."""
            pass

from razorkv.config import RazorConfig


class RazorLayerState:
    """Lightweight wrapper matching Transformers 5.x DynamicLayer interface."""

    is_compileable: bool = False
    is_sliding: bool = False
    is_croppable: bool = True

    def __init__(
        self,
        keys: Optional[torch.Tensor] = None,
        values: Optional[torch.Tensor] = None,
        cumulative_length: int = 0,
    ):
        self.keys = keys
        self.values = values
        self.cumulative_length = cumulative_length

    def get_seq_length(self) -> int:
        """Returns the logical cumulative sequence length for position embedding computation."""
        return self.cumulative_length

    def get_mask_sizes(self, query_length: Union[int, torch.Tensor]) -> Tuple[int, int]:
        """Returns (kv_length, kv_offset) for attention masking."""
        if isinstance(query_length, torch.Tensor):
            q_len = query_length.shape[0] if query_length.ndim > 0 else int(query_length.item())
        else:
            q_len = int(query_length)
        k_len = self.keys.shape[-2] if self.keys is not None else 0
        return k_len + q_len, 0

    def crop(self, tokens_to_remove: int = 0) -> None:
        """Removes the most recent tokens from this layer's cache."""
        if tokens_to_remove > 0 and self.keys is not None:
            self.keys = self.keys[:, :, :-tokens_to_remove, :]
            self.values = self.values[:, :, :-tokens_to_remove, :]
            self.cumulative_length = max(0, self.cumulative_length - tokens_to_remove)


class RazorKVCache(BaseCacheClass):
    """Production-grade Dynamic KV Cache with Layer-Pyramid Paged Eviction.

    Fully compatible with HuggingFace `generate(..., past_key_values=cache)`
    across Transformers 4.x and 5.x, as well as standalone PyTorch attention modules.
    """

    def __init__(self, config: Optional[RazorConfig] = None, num_layers: int = 32):
        try:
            super().__init__()
        except Exception:
            pass

        self.config = config or RazorConfig()
        self.num_layers = num_layers

        # Internal per-layer storage (Transformers 4.x & PyTorch standard)
        self.key_cache: List[Optional[torch.Tensor]] = []
        self.value_cache: List[Optional[torch.Tensor]] = []
        self.salience_scores: List[Optional[torch.Tensor]] = []  # Shape: [batch_size, seq_len]

        # Transformers 5.x compatibility layer
        self.layers: List[RazorLayerState] = []

        # Tracking total tokens seen across generation
        self._seen_tokens: int = 0

    @property
    def seen_tokens(self) -> int:
        return self._seen_tokens

    @property
    def is_compileable(self) -> bool:
        return False

    @property
    def is_sliding(self) -> List[bool]:
        return [False] * len(self.layers)

    @property
    def is_croppable(self) -> bool:
        return True

    @property
    def is_initialized(self) -> bool:
        return len(self.key_cache) > 0 and self.key_cache[0] is not None

    @property
    def batch_size(self) -> int:
        if self.is_initialized:
            return self.key_cache[0].shape[0]
        return 0

    def __len__(self) -> int:
        """Returns the logical sequence length seen across generation."""
        return self._seen_tokens

    def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Provides tuple-style indexing: (key, value) = cache[layer_idx]."""
        if layer_idx < len(self.key_cache) and self.key_cache[layer_idx] is not None:
            return self.key_cache[layer_idx], self.value_cache[layer_idx]
        raise IndexError(f"Layer index {layer_idx} out of range ({len(self.key_cache)} layers allocated).")

    def __iter__(self):
        for k, v in zip(self.key_cache, self.value_cache):
            if k is not None and v is not None:
                yield (k, v)

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns logical cumulative sequence length (used by RoPE position embedding alignment)."""
        return self._seen_tokens

    def get_cached_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the physical token count currently stored in memory for a given layer."""
        if layer_idx is None:
            layer_idx = 0
        if layer_idx >= len(self.key_cache) or self.key_cache[layer_idx] is None:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_length(self) -> Optional[int]:
        """Returns maximum length if known."""
        return None

    def get_max_cache_shape(self) -> Optional[int]:
        """Compatibility method for Transformers 5.x."""
        return None

    def get_usable_length(self, new_seq_len: int, layer_idx: Optional[int] = 0) -> int:
        """Returns the usable sequence length for position computation."""
        return self._seen_tokens

    def get_mask_sizes(self, query_length: Union[int, torch.Tensor], layer_idx: int = 0) -> Tuple[int, int]:
        """Returns (kv_length, kv_offset) for Transformers attention masking."""
        if isinstance(query_length, torch.Tensor):
            q_len = query_length.shape[0] if query_length.ndim > 0 else int(query_length.item())
        else:
            q_len = int(query_length)
        cached_len = self.get_cached_seq_length(layer_idx)
        return cached_len + q_len, 0

    def get_query_offset(self, layer_idx: int = 0) -> int:
        """Returns current query offset matching logical seen tokens."""
        return self._seen_tokens

    def crop(self, tokens_to_remove: int = 0) -> None:
        """Crops the most recent tokens from the cache across all layers."""
        if tokens_to_remove <= 0:
            return
        for layer_idx in range(len(self.key_cache)):
            if self.key_cache[layer_idx] is not None:
                self.key_cache[layer_idx] = self.key_cache[layer_idx][:, :, :-tokens_to_remove, :]
                self.value_cache[layer_idx] = self.value_cache[layer_idx][:, :, :-tokens_to_remove, :]
                if layer_idx < len(self.salience_scores) and self.salience_scores[layer_idx] is not None:
                    self.salience_scores[layer_idx] = self.salience_scores[layer_idx][:, :-tokens_to_remove]
                self._sync_layers(layer_idx)
        self._seen_tokens = max(0, self._seen_tokens - tokens_to_remove)

    def reset(self) -> None:
        """Resets all internal cache buffers."""
        self.key_cache.clear()
        self.value_cache.clear()
        self.salience_scores.clear()
        self.layers.clear()
        self._seen_tokens = 0

    def batch_repeat_interleave(self, repeats: int) -> None:
        """Repeats cache across the batch dimension (used in beam search / num_return_sequences)."""
        for layer_idx in range(len(self.key_cache)):
            if self.key_cache[layer_idx] is not None:
                self.key_cache[layer_idx] = self.key_cache[layer_idx].repeat_interleave(repeats, dim=0)
                self.value_cache[layer_idx] = self.value_cache[layer_idx].repeat_interleave(repeats, dim=0)
                if layer_idx < len(self.salience_scores) and self.salience_scores[layer_idx] is not None:
                    self.salience_scores[layer_idx] = self.salience_scores[layer_idx].repeat_interleave(repeats, dim=0)
                self._sync_layers(layer_idx)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        """Filters cache along batch dimension given indices."""
        for layer_idx in range(len(self.key_cache)):
            if self.key_cache[layer_idx] is not None:
                dev = self.key_cache[layer_idx].device
                idx = indices.to(dev)
                self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, idx)
                self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, idx)
                if layer_idx < len(self.salience_scores) and self.salience_scores[layer_idx] is not None:
                    self.salience_scores[layer_idx] = self.salience_scores[layer_idx].index_select(0, idx)
                self._sync_layers(layer_idx)

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search generation."""
        for layer_idx in range(len(self.key_cache)):
            if self.key_cache[layer_idx] is not None:
                device = self.key_cache[layer_idx].device
                self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
                self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))
                if layer_idx < len(self.salience_scores) and self.salience_scores[layer_idx] is not None:
                    self.salience_scores[layer_idx] = self.salience_scores[layer_idx].index_select(0, beam_idx.to(device))
                self._sync_layers(layer_idx)

    def _sync_layers(self, layer_idx: int):
        """Maintains Transformers 5.x self.layers compatibility."""
        while len(self.layers) <= layer_idx:
            self.layers.append(RazorLayerState())
        self.layers[layer_idx].keys = self.key_cache[layer_idx]
        self.layers[layer_idx].values = self.value_cache[layer_idx]
        self.layers[layer_idx].cumulative_length = self._seen_tokens

    def _init_salience(self, key_states: torch.Tensor) -> torch.Tensor:
        """Initializes salience scores for new keys based on L2-norm magnitude."""
        if self.config.salience_init == "norm":
            norm = torch.linalg.vector_norm(key_states.float(), ord=2, dim=-1)  # [batch, heads, seq]
            salience = norm.mean(dim=1)  # Average over heads -> [batch, seq]
        else:
            batch_size, _, seq_len, _ = key_states.shape
            salience = torch.ones((batch_size, seq_len), device=key_states.device, dtype=torch.float32)
        return salience

    def update_salience_with_query(
        self,
        query_states: torch.Tensor,
        layer_idx: int,
    ):
        """Updates query-key resonance scores using incoming query states.

        Args:
            query_states: [batch_size, num_heads, q_len, head_dim]
            layer_idx: Layer index to update
        """
        if layer_idx >= len(self.key_cache) or self.key_cache[layer_idx] is None:
            return

        keys = self.key_cache[layer_idx]  # [batch, num_kv_heads, seq_len, head_dim]
        b, q_heads, q_len, d = query_states.shape
        _, kv_heads, seq_len, _ = keys.shape

        # Expand KV heads if Grouped Query Attention (GQA)
        if q_heads != kv_heads:
            group_size = q_heads // kv_heads
            keys_expanded = keys.repeat_interleave(group_size, dim=1)
        else:
            keys_expanded = keys

        scale = 1.0 / math.sqrt(d)
        q_dev = keys.device
        q_matched = query_states.to(device=q_dev, dtype=torch.float32)
        k_matched = keys_expanded.to(device=q_dev, dtype=torch.float32)

        scores = torch.matmul(q_matched * scale, k_matched.transpose(-1, -2))
        max_resonance = scores.amax(dim=(1, 2))  # [batch, seq_len]

        decay = self.config.salience_decay
        if layer_idx < len(self.salience_scores) and self.salience_scores[layer_idx] is not None:
            old_salience = self.salience_scores[layer_idx].to(device=q_dev)
            self.salience_scores[layer_idx] = decay * old_salience + (1.0 - decay) * max_resonance
        else:
            self.salience_scores[layer_idx] = max_resonance

    def _evict_layer(self, layer_idx: int, target_budget: int):
        """Performs Block-Paged Layer Eviction to compress cache to target_budget tokens."""
        keys = self.key_cache[layer_idx]
        values = self.value_cache[layer_idx]
        salience = self.salience_scores[layer_idx]

        seq_len = keys.shape[-2]
        if seq_len <= target_budget:
            return

        batch_size = keys.shape[0]
        sink_len = min(self.config.sink_tokens, seq_len)
        local_len = min(self.config.local_window, seq_len - sink_len)

        haystack_start = sink_len
        haystack_end = seq_len - local_len
        haystack_len = haystack_end - haystack_start

        if haystack_len <= 0:
            return

        haystack_budget = target_budget - sink_len - local_len
        if haystack_budget <= 0:
            keep_indices = torch.cat([
                torch.arange(0, sink_len, device=keys.device, dtype=torch.long),
                torch.arange(haystack_end, seq_len, device=keys.device, dtype=torch.long),
            ])
            self.key_cache[layer_idx] = keys[:, :, keep_indices, :].contiguous()
            self.value_cache[layer_idx] = values[:, :, keep_indices, :].contiguous()
            self.salience_scores[layer_idx] = salience[:, keep_indices].contiguous()
            self._sync_layers(layer_idx)
            return

        page_size = self.config.page_size
        num_pages = haystack_len // page_size

        if num_pages <= 0:
            return

        aligned_haystack_len = num_pages * page_size
        haystack_salience = salience[:, haystack_start : haystack_start + aligned_haystack_len]
        paged_salience = haystack_salience.view(batch_size, num_pages, page_size)
        page_scores = paged_salience.amax(dim=-1)

        pages_to_keep = max(1, min(num_pages, haystack_budget // page_size))

        if batch_size > 1:
            avg_page_scores = page_scores.mean(dim=0)
        else:
            avg_page_scores = page_scores[0]

        _, top_page_indices = torch.topk(avg_page_scores, k=pages_to_keep, largest=True, sorted=False)
        top_page_indices, _ = torch.sort(top_page_indices)

        # Vectorized on-device token index generation (avoids Python loops and CPU-GPU sync)
        page_offsets = top_page_indices.unsqueeze(1) * page_size + torch.arange(page_size, device=keys.device, dtype=torch.long).unsqueeze(0)
        kept_haystack_tensor = (haystack_start + page_offsets).flatten()

        if aligned_haystack_len < haystack_len:
            remainder_indices = torch.arange(haystack_start + aligned_haystack_len, haystack_end, device=keys.device, dtype=torch.long)
            kept_haystack_tensor = torch.cat([kept_haystack_tensor, remainder_indices])

        sink_indices = torch.arange(0, sink_len, device=keys.device, dtype=torch.long)
        local_indices = torch.arange(haystack_end, seq_len, device=keys.device, dtype=torch.long)

        final_indices = torch.cat([sink_indices, kept_haystack_tensor, local_indices])

        self.key_cache[layer_idx] = keys[:, :, final_indices, :].contiguous()
        self.value_cache[layer_idx] = values[:, :, final_indices, :].contiguous()
        self.salience_scores[layer_idx] = salience[:, final_indices].contiguous()
        self._sync_layers(layer_idx)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Appends new key and value states, updates resonance, and executes paged eviction if needed."""
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(None)
            self.value_cache.append(None)
            self.salience_scores.append(None)

        new_salience = self._init_salience(key_states)
        is_prefill = key_states.shape[-2] > 1

        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states.contiguous()
            self.value_cache[layer_idx] = value_states.contiguous()
            self.salience_scores[layer_idx] = new_salience.contiguous()
            if layer_idx == 0:
                self._seen_tokens = key_states.shape[-2]
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2).contiguous()
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2).contiguous()
            self.salience_scores[layer_idx] = torch.cat([self.salience_scores[layer_idx], new_salience], dim=-1).contiguous()
            if layer_idx == 0:
                self._seen_tokens += key_states.shape[-2]

        self._sync_layers(layer_idx)

        if cache_kwargs is not None and "query_states" in cache_kwargs:
            self.update_salience_with_query(cache_kwargs["query_states"], layer_idx)

        current_len = self.key_cache[layer_idx].shape[-2]
        num_layers = max(self.num_layers, len(self.key_cache))
        total_seq_len = max(self._seen_tokens, current_len)
        target_budget = self.config.get_layer_budget(layer_idx, num_layers, total_seq_len)

        if is_prefill:
            # For prefill (q_len > 1), return full states so causal attention over prompt
            # tokens is mathematically exact, then evict internal stored cache to target budget.
            ret_k = self.key_cache[layer_idx]
            ret_v = self.value_cache[layer_idx]
            if current_len > target_budget:
                self._evict_layer(layer_idx, target_budget)
            return ret_k, ret_v

        # Autoregressive single-token decode path with hysteresis compaction
        hysteresis_threshold = target_budget + self.config.page_size
        if current_len > hysteresis_threshold:
            self._evict_layer(layer_idx, target_budget)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_memory_stats(self) -> Dict[str, Any]:
        """Calculates total allocated VRAM for key/value cache across all layers."""
        total_elements = 0
        total_bytes = 0
        per_layer_tokens = []

        for k, v in zip(self.key_cache, self.value_cache):
            if k is not None and v is not None:
                layer_elems = k.numel() + v.numel()
                layer_bytes = layer_elems * k.element_size()
                total_elements += layer_elems
                total_bytes += layer_bytes
                per_layer_tokens.append(k.shape[-2])

        return {
            "total_bytes": total_bytes,
            "total_mb": total_bytes / (1024 * 1024),
            "total_elements": total_elements,
            "per_layer_tokens": per_layer_tokens,
            "seen_tokens": self._seen_tokens,
        }
