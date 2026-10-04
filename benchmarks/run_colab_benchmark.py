"""
Colab GPU Benchmark Execution Script for RazorKV
================================================
Runs on Google Colab GPU (Tesla T4 / A100 / L4).
Executes comparative benchmarks across context lengths (8k, 16k, 32k, 64k)
comparing:
- Dense Attention (Full KV Cache)
- StreamingLLM (Sink + Window)
- H2O (Heavy Hitter Oracle)
- RazorKV (Layer-Pyramid Dynamic Paged Resonance)

Outputs JSON results and formatted ASCII tables.
"""

import gc
import json
import math
import os
import sys
import time
import torch
import torch.nn.functional as F

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from razorkv.config import RazorConfig
from razorkv.cache import RazorKVCache
from razorkv.evaluation.benchmark_runner import BenchmarkRunner


def main():
    print("=" * 70)
    print("  RAZORKV EMPIRICAL BENCHMARK SUITE")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Device: {device}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU Model:       {gpu_name}")
        print(f"Total VRAM:      {vram_total:.2f} GB")
    else:
        print("WARNING: CUDA not available. Running in CPU emulation mode.")

    # Model configuration matching LLaMA-3-8B / Qwen-2.5-7B
    num_layers = 32
    num_q_heads = 32
    num_kv_heads = 8
    head_dim = 128
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Context lengths to test: 8k, 16k, 32k, 64k
    # (Adjust for GPU memory bounds if needed)
    context_lengths = [8192, 16384, 32768, 64000]

    runner = BenchmarkRunner(
        num_layers=num_layers,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        device=device,
        dtype=dtype,
    )

    print("\nStarting automated benchmark runs...")
    results = runner.run_benchmark_suite(
        context_lengths=context_lengths,
        decode_steps=32,
        needle_depth=0.50,
    )

    # Save output to JSON
    output_path = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] Benchmark results successfully exported to: {output_path}")

    # Generate Formatted Summary Table
    print("\n" + "=" * 90)
    print("  SUMMARY: VRAM SAVINGS & RETRIEVAL ACCURACY COMPARISON")
    print("=" * 90)
    print(f"{'Context':<10} | {'Engine':<14} | {'VRAM (MB)':<12} | {'VRAM Saved':<12} | {'Decode tok/s':<14} | {'Retrieval Acc':<14}")
    print("-" * 90)

    for ctx_len in context_lengths:
        ctx_data = results["data"][ctx_len]
        for eng in ["dense", "streamingllm", "h2o", "razorkv"]:
            d = ctx_data[eng]
            print(
                f"{ctx_len:<10,} | "
                f"{eng.upper():<14} | "
                f"{d['vram_mb']:>10.1f} MB | "
                f"{d['vram_saving_pct']:>10.1f}% | "
                f"{d['throughput_tok_s']:>12.1f} | "
                f"{d['retrieval_acc']*100:>12.1f}%"
            )
        print("-" * 90)


if __name__ == "__main__":
    main()
