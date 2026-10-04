"""
RazorKV Configuration
=====================
Defines hyper-parameters, layer-pyramid budget distribution schedules,
and memory block specifications for dynamic KV cache compression.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union
import torch


@dataclass
class RazorConfig:
    """Configuration class for RazorKV dynamic compression engine.

    Attributes:
        compression_ratio: Target fraction of KV cache to retain across the model.
                           e.g., 0.30 means retaining 30% KV cache (70% VRAM reduction).
        sink_tokens: Number of initial tokens permanently retained as attention sinks (Zone 1).
        local_window: Number of recent tokens permanently retained in dense sliding window (Zone 3).
        page_size: Size of contiguous token blocks for paged memory allocation (default: 16).
        pyramid_mode: Budget scheduling across transformer layers:
                      - 'adaptive': Steeper pyramid allocating low budget to shallow layers
                                    and high budget to deep retrieval layers.
                      - 'linear': Linear gradient from pyramid_min_ratio to pyramid_max_ratio.
                      - 'uniform': Uniform budget across all layers (flat compression).
                      - 'custom': User-specified per-layer retention ratios via custom_layer_ratios.
        pyramid_min_ratio: Minimum retention ratio for shallowest layers in adaptive/linear modes.
        pyramid_max_ratio: Maximum retention ratio for deepest layers in adaptive/linear modes.
        salience_decay: Exponential moving average decay rate for query-key resonance (gamma in (0, 1)).
        salience_init: Strategy to initialize key salience ('norm' or 'uniform').
        custom_layer_ratios: Optional list of explicit retention ratios per layer.
        use_triton: Whether to use high-performance Triton sparse kernels if available.
        device: Target torch device ('cuda', 'cpu', etc.).
        dtype: Data type for key/value tensors (e.g. torch.float16, torch.bfloat16).
    """

    compression_ratio: float = 0.30
    sink_tokens: int = 32
    local_window: int = 512
    page_size: int = 16
    pyramid_mode: str = "adaptive"
    pyramid_min_ratio: float = 0.15
    pyramid_max_ratio: float = 0.70
    salience_decay: float = 0.98
    salience_init: str = "norm"
    custom_layer_ratios: Optional[List[float]] = None
    use_triton: bool = False
    device: Optional[Union[str, torch.device]] = None
    dtype: torch.dtype = torch.float16

    def __post_init__(self):
        if not (0.0 < self.compression_ratio <= 1.0):
            raise ValueError(f"compression_ratio must be in (0, 1], got {self.compression_ratio}")
        if self.sink_tokens < 0:
            raise ValueError(f"sink_tokens must be >= 0, got {self.sink_tokens}")
        if self.local_window < 0:
            raise ValueError(f"local_window must be >= 0, got {self.local_window}")
        if self.page_size < 1:
            raise ValueError(f"page_size must be >= 1, got {self.page_size}")
        if not (0.0 < self.salience_decay <= 1.0):
            raise ValueError(f"salience_decay must be in (0, 1], got {self.salience_decay}")

    def get_layer_ratio(self, layer_idx: int, num_layers: int) -> float:
        """Computes the target retention ratio for a specific layer index."""
        if num_layers <= 1 or self.pyramid_mode == "uniform":
            return self.compression_ratio

        if self.pyramid_mode == "custom" and self.custom_layer_ratios is not None:
            if layer_idx < len(self.custom_layer_ratios):
                return self.custom_layer_ratios[layer_idx]
            return self.compression_ratio

        progress = layer_idx / (num_layers - 1)  # 0.0 at layer 0, 1.0 at layer N-1

        if self.pyramid_mode == "linear":
            # Linear ramp from min_ratio to max_ratio
            ratio = self.pyramid_min_ratio + progress * (self.pyramid_max_ratio - self.pyramid_min_ratio)
        elif self.pyramid_mode == "adaptive":
            # Quadratic / exponential concentration: bottom layers are ultra-compressed,
            # top layers maintain high resolution for needle retrieval and logits synthesis.
            # Scale curve so that average across layers matches target compression_ratio
            curve = progress ** 1.8  # concave up: low in bottom half, climbs in top half
            raw_ratios = [
                self.pyramid_min_ratio + (i / (num_layers - 1)) ** 1.8 * (self.pyramid_max_ratio - self.pyramid_min_ratio)
                for i in range(num_layers)
            ]
            avg_raw = sum(raw_ratios) / len(raw_ratios)
            scale = self.compression_ratio / max(avg_raw, 1e-6)
            scaled_ratio = (self.pyramid_min_ratio + curve * (self.pyramid_max_ratio - self.pyramid_min_ratio)) * scale
            ratio = max(0.05, min(1.0, scaled_ratio))
        else:
            ratio = self.compression_ratio

        return ratio

    def get_layer_budget(self, layer_idx: int, num_layers: int, total_seq_len: int) -> int:
        """Computes the maximum number of KV tokens to retain in the given layer.

        Guarantees that at minimum, sink_tokens + local_window tokens are preserved.
        """
        min_preserved = self.sink_tokens + self.local_window
        if total_seq_len <= min_preserved:
            return total_seq_len

        ratio = self.get_layer_ratio(layer_idx, num_layers)
        target_budget = int(total_seq_len * ratio)

        # Budget must at least cover sink and local window
        budget = max(min_preserved, target_budget)
        return min(total_seq_len, budget)
