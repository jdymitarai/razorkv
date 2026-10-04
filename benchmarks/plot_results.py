"""
Results Formatter and ASCII Chart Generator
===========================================
Reads benchmark_results.json and generates Markdown tables and ASCII bar graphs.
"""

import json
import os
import sys


def render_ascii_bar(val: float, max_val: float, bar_len: int = 25) -> str:
    filled = int((val / max(max_val, 1e-6)) * bar_len)
    return "█" * filled + "░" * (bar_len - filled)


def generate_markdown_report(json_path: str) -> str:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    device_name = data.get("device_name", "Unknown GPU")
    lengths = data.get("lengths", [])
    ctx_data = data.get("data", {})

    lines = []
    lines.append(f"### Benchmark Hardware Environment: {device_name}\n")
    lines.append("| Context Length | Engine | KV VRAM (MB) | VRAM Saved (%) | Decode Speed (tok/s) | Retrieval Acc (%) |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")

    for ctx in lengths:
        c_str = str(ctx)
        engines = ctx_data.get(c_str, {})
        for eng in ["dense", "streamingllm", "h2o", "razorkv"]:
            if eng in engines:
                item = engines[eng]
                vram = f"{item['vram_mb']:.1f} MB"
                saved = f"**{item['vram_saving_pct']:.1f}%**" if eng == "razorkv" else f"{item['vram_saving_pct']:.1f}%"
                speed = f"{item['throughput_tok_s']:.1f}"
                acc = f"**{item['retrieval_acc']*100:.1f}%**" if eng == "razorkv" else f"{item['retrieval_acc']*100:.1f}%"
                lines.append(f"| {ctx:,} | `{eng.upper()}` | {vram} | {saved} | {speed} | {acc} |")
        lines.append("| --- | --- | --- | --- | --- | --- |")

    return "\n".join(lines)


if __name__ == "__main__":
    json_path = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
    if os.path.exists(json_path):
        print(generate_markdown_report(json_path))
    else:
        print(f"File not found: {json_path}")
