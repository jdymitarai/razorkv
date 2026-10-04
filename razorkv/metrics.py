"""
RazorKV Performance Profiler & Metrics
======================================
Provides precision GPU memory tracking, CUDA event timing, throughput calculation,
and retrieval accuracy measurement tools.
"""

import time
from typing import Any, Callable, Dict, List, Optional, Union
import torch


class MemoryProfiler:
    """Accurate GPU and host memory tracker."""

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    def start(self):
        """Resets peak memory statistics."""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)

    def get_peak_mb(self) -> float:
        """Returns peak allocated memory in Megabytes."""
        if self.device.type == "cuda":
            return torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
        return 0.0

    def get_current_mb(self) -> float:
        """Returns currently allocated memory in Megabytes."""
        if self.device.type == "cuda":
            return torch.cuda.memory_allocated(self.device) / (1024 * 1024)
        return 0.0


class BenchmarkTimer:
    """High-precision latency and throughput timer."""

    def __init__(self, use_cuda_events: bool = True):
        self.is_cuda = use_cuda_events and torch.cuda.is_available()
        self.start_event = None
        self.end_event = None
        self.start_time = 0.0
        self.elapsed_ms = 0.0

    def __enter__(self):
        if self.is_cuda:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
        else:
            self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.is_cuda:
            self.end_event.record()
            torch.cuda.synchronize()
            self.elapsed_ms = self.start_event.elapsed_time(self.end_event)
        else:
            self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000.0

    @property
    def seconds(self) -> float:
        return self.elapsed_ms / 1000.0

    @property
    def tokens_per_sec(self) -> Callable[[int], float]:
        return lambda num_tokens: num_tokens / max(self.seconds, 1e-6)


def evaluate_retrieval(answer: str, target: str) -> Dict[str, Union[bool, float]]:
    """Evaluates needle retrieval response against ground truth."""
    clean_ans = answer.strip().lower()
    clean_tgt = target.strip().lower()

    exact_match = clean_tgt == clean_ans
    substring_match = clean_tgt in clean_ans

    # Token overlap F1
    ans_tokens = set(clean_ans.split())
    tgt_tokens = set(clean_tgt.split())
    common = ans_tokens.intersection(tgt_tokens)
    if not ans_tokens or not tgt_tokens:
        f1 = 0.0
    else:
        precision = len(common) / len(ans_tokens)
        recall = len(common) / len(tgt_tokens)
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "exact_match": exact_match,
        "substring_match": substring_match,
        "f1_score": f1,
        "passed": exact_match or substring_match,
    }
