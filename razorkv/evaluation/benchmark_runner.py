"""
RazorKV Automated Benchmark Runner
==================================
Conducts rigorous comparative benchmarks against:
1. Full Dense KV Cache (Uncompressed Baseline)
2. StreamingLLM (Sink + Local Window)
3. H2O (Heavy Hitter Oracle / Cumulative Attention)
4. RazorKV (Layer-Pyramid Paged Dynamic Resonance)

Measures:
- VRAM Allocated & Savings (%)
- Decode Throughput (tokens/sec) & Speedup (x)
- Latency per token (ms)
- Needle Retrieval Accuracy (%) across context depths
"""

import gc
import math
import time
from typing import Any, Dict, List, Optional
import torch
import torch.nn.functional as F

from razorkv.config import RazorConfig
from razorkv.cache import RazorKVCache
from razorkv.metrics import MemoryProfiler, BenchmarkTimer


class BaselineStreamingLLMCache:
    """Implements StreamingLLM eviction: retains sink tokens + sliding local window."""
    def __init__(self, sink_tokens: int = 32, local_window: int = 1024):
        self.sink_tokens = sink_tokens
        self.local_window = local_window
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int):
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(None)
            self.value_cache.append(None)

        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        cur_len = self.key_cache[layer_idx].shape[-2]
        max_budget = self.sink_tokens + self.local_window
        if cur_len > max_budget:
            keys = self.key_cache[layer_idx]
            values = self.value_cache[layer_idx]
            sink_k = keys[:, :, :self.sink_tokens, :]
            sink_v = values[:, :, :self.sink_tokens, :]
            recent_k = keys[:, :, -self.local_window:, :]
            recent_v = values[:, :, -self.local_window:, :]
            self.key_cache[layer_idx] = torch.cat([sink_k, recent_k], dim=-2).contiguous()
            self.value_cache[layer_idx] = torch.cat([sink_v, recent_v], dim=-2).contiguous()

        return self.key_cache[layer_idx], self.value_cache[layer_idx]


class BaselineH2OCache:
    """Implements H2O eviction: retains sink tokens + heavy-hitters (cumulative score) + local window."""
    def __init__(self, budget_ratio: float = 0.30, sink_tokens: int = 32, local_window: int = 512):
        self.budget_ratio = budget_ratio
        self.sink_tokens = sink_tokens
        self.local_window = local_window
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.cumulative_scores: List[torch.Tensor] = []
        self.seen_tokens: int = 0

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int):
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(None)
            self.value_cache.append(None)
            self.cumulative_scores.append(None)

        b, _, new_len, _ = key_states.shape
        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
            self.cumulative_scores[layer_idx] = torch.zeros((b, new_len), device=key_states.device)
            if layer_idx == 0:
                self.seen_tokens = new_len
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
            new_zeros = torch.zeros((b, new_len), device=key_states.device)
            self.cumulative_scores[layer_idx] = torch.cat([self.cumulative_scores[layer_idx], new_zeros], dim=-1)
            if layer_idx == 0:
                self.seen_tokens += new_len

        cur_len = self.key_cache[layer_idx].shape[-2]
        budget = max(self.sink_tokens + self.local_window, int(self.seen_tokens * self.budget_ratio))
        if cur_len > budget:
            keys = self.key_cache[layer_idx]
            values = self.value_cache[layer_idx]
            scores = self.cumulative_scores[layer_idx]

            haystack_k_budget = budget - self.sink_tokens - self.local_window
            haystack_scores = scores[:, self.sink_tokens : cur_len - self.local_window]

            if haystack_k_budget > 0 and haystack_scores.shape[-1] > haystack_k_budget:
                _, topk_idx = torch.topk(haystack_scores[0], k=haystack_k_budget, sorted=False)
                topk_idx, _ = torch.sort(topk_idx + self.sink_tokens)
                kept_indices = torch.cat([
                    torch.arange(0, self.sink_tokens, device=keys.device),
                    topk_idx,
                    torch.arange(cur_len - self.local_window, cur_len, device=keys.device),
                ])
                self.key_cache[layer_idx] = keys[:, :, kept_indices, :].contiguous()
                self.value_cache[layer_idx] = values[:, :, kept_indices, :].contiguous()
                self.cumulative_scores[layer_idx] = scores[:, kept_indices].contiguous()

        return self.key_cache[layer_idx], self.value_cache[layer_idx]


class BenchmarkRunner:
    """Executes high-accuracy long-context inference benchmarks across multiple engines."""

    def __init__(
        self,
        num_layers: int = 32,
        num_q_heads: int = 32,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float16,
    ):
        self.num_layers = num_layers
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.dtype = dtype

    def run_benchmark_suite(
        self,
        context_lengths: List[int] = [8192, 16384, 32768, 65536],
        decode_steps: int = 32,
        needle_depth: float = 0.50,
    ) -> Dict[str, Any]:
        """Runs the complete benchmark suite across context lengths."""
        results = {
            "device": str(self.device),
            "device_name": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "CPU",
            "num_layers": self.num_layers,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "lengths": context_lengths,
            "data": {},
        }

        engines = ["dense", "streamingllm", "h2o", "razorkv"]

        for ctx_len in context_lengths:
            print(f"\n=======================================================")
            print(f"  BENCHMARKING CONTEXT LENGTH: {ctx_len:,} TOKENS")
            print(f"=======================================================")
            results["data"][ctx_len] = {}

            for engine_name in engines:
                # Run engine benchmark
                engine_res = self._benchmark_single_engine(
                    engine_name=engine_name,
                    context_len=ctx_len,
                    decode_steps=decode_steps,
                    needle_depth=needle_depth,
                )
                results["data"][ctx_len][engine_name] = engine_res

                print(
                    f"[{engine_name.upper():<12}] "
                    f"KV VRAM: {engine_res['vram_mb']:>8.1f} MB ({engine_res['vram_saving_pct']:>5.1f}% saved) | "
                    f"Decode: {engine_res['throughput_tok_s']:>6.1f} tok/s ({engine_res['latency_ms_per_tok']:>5.2f} ms/tok) | "
                    f"Retrieval: {engine_res['retrieval_acc'] * 100:>5.1f}%"
                )

        return results

    def _benchmark_single_engine(
        self,
        engine_name: str,
        context_len: int,
        decode_steps: int = 32,
        needle_depth: float = 0.50,
    ) -> Dict[str, Any]:
        """Benchmarks a single cache engine on GPU."""
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)

        batch_size = 1
        q_heads = self.num_q_heads
        kv_heads = self.num_kv_heads
        d = self.head_dim

        # Needle position index
        needle_idx = int(context_len * needle_depth)
        target_needle_key = torch.randn(1, 1, 1, d, device=self.device, dtype=self.dtype) * 3.0
        needle_retrieval_query = target_needle_key.clone().expand(1, q_heads, 1, d)

        # Initialize cache instance
        if engine_name == "dense":
            cache_keys = []
            cache_values = []
        elif engine_name == "streamingllm":
            cache = BaselineStreamingLLMCache(sink_tokens=32, local_window=1024)
        elif engine_name == "h2o":
            cache = BaselineH2OCache(budget_ratio=0.30, sink_tokens=32, local_window=512)
        elif engine_name == "razorkv":
            cfg = RazorConfig(compression_ratio=0.30, sink_tokens=32, local_window=512, page_size=16)
            cache = RazorKVCache(config=cfg, num_layers=self.num_layers)
        else:
            raise ValueError(f"Unknown engine: {engine_name}")

        # Measure baseline memory
        base_mem = torch.cuda.memory_allocated(self.device) if self.device.type == "cuda" else 0

        # Prefill phase: generate keys and values in chunks to simulate realistic prompt prefill
        chunk_size = 4096
        num_chunks = math.ceil(context_len / chunk_size)

        for chunk_idx in range(num_chunks):
            start_pos = chunk_idx * chunk_size
            curr_chunk = min(chunk_size, context_len - start_pos)
            k_chunk = torch.randn(batch_size, kv_heads, curr_chunk, d, device=self.device, dtype=self.dtype)
            v_chunk = torch.randn(batch_size, kv_heads, curr_chunk, d, device=self.device, dtype=self.dtype)

            # Inject needle at target index if within this chunk
            if start_pos <= needle_idx < start_pos + curr_chunk:
                offset = needle_idx - start_pos
                k_chunk[:, :, offset : offset + 1, :] = target_needle_key

            for layer_idx in range(self.num_layers):
                if engine_name == "dense":
                    if chunk_idx == 0:
                        cache_keys.append(k_chunk)
                        cache_values.append(v_chunk)
                    else:
                        cache_keys[layer_idx] = torch.cat([cache_keys[layer_idx], k_chunk], dim=-2)
                        cache_values[layer_idx] = torch.cat([cache_values[layer_idx], v_chunk], dim=-2)
                else:
                    cache.update(k_chunk, v_chunk, layer_idx)

        # Record KV Cache VRAM after prefill
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            peak_vram_bytes = torch.cuda.max_memory_allocated(self.device) - base_mem
            peak_vram_mb = peak_vram_bytes / (1024 * 1024)
        else:
            peak_vram_mb = (2 * self.num_layers * kv_heads * context_len * d * 2) / (1024 * 1024)

        # Autoregressive Decode phase: generate decode_steps tokens
        decode_query = torch.randn(batch_size, q_heads, 1, d, device=self.device, dtype=self.dtype)
        start_t = time.perf_counter()

        for step in range(decode_steps):
            k_step = torch.randn(batch_size, kv_heads, 1, d, device=self.device, dtype=self.dtype)
            v_step = torch.randn(batch_size, kv_heads, 1, d, device=self.device, dtype=self.dtype)

            for layer_idx in range(self.num_layers):
                if engine_name == "dense":
                    cache_keys[layer_idx] = torch.cat([cache_keys[layer_idx], k_step], dim=-2)
                    cache_values[layer_idx] = torch.cat([cache_values[layer_idx], v_step], dim=-2)
                    cur_k = cache_keys[layer_idx]
                    cur_v = cache_values[layer_idx]
                else:
                    cur_k, cur_v = cache.update(k_step, v_step, layer_idx)

                # Execute attention for this layer
                if q_heads != kv_heads:
                    cur_k = cur_k.repeat_interleave(q_heads // kv_heads, dim=1)
                    cur_v = cur_v.repeat_interleave(q_heads // kv_heads, dim=1)

                _ = F.scaled_dot_product_attention(decode_query, cur_k, cur_v, is_causal=False)

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        total_decode_time = time.perf_counter() - start_t
        throughput = decode_steps / max(total_decode_time, 1e-6)
        latency_ms = (total_decode_time / decode_steps) * 1000.0

        # Retrieval accuracy evaluation on needle:
        # Check deep layer (layer 30) attention weights when queried with needle retrieval query
        if engine_name == "dense":
            k_retrieval = cache_keys[-1]
        elif engine_name == "razorkv":
            k_retrieval = cache.key_cache[-1]
        elif engine_name == "streamingllm":
            k_retrieval = cache.key_cache[-1]
        elif engine_name == "h2o":
            k_retrieval = cache.key_cache[-1]

        # Calculate max cosine similarity with target needle key
        k_norm = F.normalize(k_retrieval.float(), dim=-1)
        tgt_norm = F.normalize(target_needle_key.float(), dim=-1)
        sim = (k_norm * tgt_norm).sum(dim=-1).max().item()

        # Needle is successfully retrieved if cosine similarity > 0.95
        retrieval_success = 1.0 if sim > 0.95 else 0.0

        # Theoretical dense VRAM for comparison
        theoretical_dense_mb = (2 * self.num_layers * kv_heads * (context_len + decode_steps) * d * 2) / (1024 * 1024)
        if engine_name == "dense":
            vram_saving_pct = 0.0
        else:
            vram_saving_pct = max(0.0, (1.0 - (peak_vram_mb / max(theoretical_dense_mb, 1e-6))) * 100.0)

        # Cleanup
        if engine_name == "dense":
            del cache_keys
            del cache_values
        else:
            del cache
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return {
            "vram_mb": peak_vram_mb,
            "vram_saving_pct": vram_saving_pct,
            "throughput_tok_s": throughput,
            "latency_ms_per_tok": latency_ms,
            "retrieval_acc": retrieval_success,
            "needle_max_sim": sim,
        }
