"""
RazorKV: Extreme Dynamic KV Cache Sparsification & Layer-Pyramid Eviction Engine
================================================================================
A plug-and-play, training-free inference engine for ultra-long-context LLMs.
Slashes KV Cache VRAM by 65-75% while preserving >98% needle retrieval accuracy.
"""

from razorkv.config import RazorConfig
from razorkv.cache import RazorKVCache
from razorkv.engine import RazorEngine
from razorkv.metrics import MemoryProfiler, BenchmarkTimer, evaluate_retrieval

__version__ = "0.1.0"

__all__ = [
    "RazorConfig",
    "RazorKVCache",
    "RazorEngine",
    "MemoryProfiler",
    "BenchmarkTimer",
    "evaluate_retrieval",
    "__version__",
]
