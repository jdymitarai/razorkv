"""
RazorKV Inference Engine & Model Patcher
========================================
Integrates RazorKV dynamically with HuggingFace AutoModelForCausalLM models
(LLaMA-3, Qwen-2.5, Mistral, DeepSeek-R1, etc.) for plug-and-play inference.
"""

from typing import Any, Dict, Optional, Union
import torch
from razorkv.config import RazorConfig
from razorkv.cache import RazorKVCache


class RazorEngine:
    """High-level engine to manage, patch, and execute LLM inference with RazorKV."""

    @staticmethod
    def create_cache(
        model_or_num_layers: Union[int, Any],
        config: Optional[RazorConfig] = None,
    ) -> RazorKVCache:
        """Creates a RazorKVCache instance sized for the given model or layer count."""
        if isinstance(model_or_num_layers, int):
            num_layers = model_or_num_layers
        elif hasattr(model_or_num_layers, "config"):
            num_layers = getattr(
                model_or_num_layers.config,
                "num_hidden_layers",
                getattr(model_or_num_layers.config, "n_layer", 32),
            )
        else:
            num_layers = 32

        return RazorKVCache(config=config, num_layers=num_layers)

    @classmethod
    def patch_model(
        cls,
        model: Any,
        config: Optional[RazorConfig] = None,
    ) -> Any:
        """Patches a HuggingFace PreTrainedModel so that .generate() automatically uses RazorKV.

        Zero fine-tuning, training-free, non-destructive patch.
        Stores original methods to allow clean unpatching via `unpatch_model`.
        """
        razor_cfg = config or RazorConfig()
        num_layers = getattr(
            model.config,
            "num_hidden_layers",
            getattr(model.config, "n_layer", 32),
        )

        # Preserve original method if not already patched
        if not hasattr(model, "_orig_prepare_inputs_for_generation"):
            model._orig_prepare_inputs_for_generation = model.prepare_inputs_for_generation
            model._is_razorkv_patched = True

        orig_prepare = model._orig_prepare_inputs_for_generation

        def razor_prepare_inputs_for_generation(input_ids, past_key_values=None, **kwargs):
            if past_key_values is None:
                past_key_values = RazorKVCache(config=razor_cfg, num_layers=num_layers)
            return orig_prepare(input_ids, past_key_values=past_key_values, **kwargs)

        model.prepare_inputs_for_generation = razor_prepare_inputs_for_generation
        model._razorkv_config = razor_cfg

        return model

    @classmethod
    def unpatch_model(cls, model: Any) -> Any:
        """Restores original HuggingFace model generation behavior."""
        if hasattr(model, "_orig_prepare_inputs_for_generation"):
            model.prepare_inputs_for_generation = model._orig_prepare_inputs_for_generation
            del model._orig_prepare_inputs_for_generation
            if hasattr(model, "_is_razorkv_patched"):
                del model._is_razorkv_patched
            if hasattr(model, "_razorkv_config"):
                del model._razorkv_config
        return model

    @staticmethod
    def estimate_vram_savings(
        context_len: int,
        num_layers: int = 32,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        dtype_bytes: int = 2,  # FP16 / BF16
        compression_ratio: float = 0.30,
    ) -> Dict[str, float]:
        """Calculates theoretical VRAM consumption and savings in Megabytes."""
        bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * dtype_bytes
        dense_bytes = bytes_per_token * context_len
        razor_bytes = dense_bytes * compression_ratio
        saved_bytes = dense_bytes - razor_bytes

        return {
            "dense_mb": dense_bytes / (1024 * 1024),
            "razorkv_mb": razor_bytes / (1024 * 1024),
            "saved_mb": saved_bytes / (1024 * 1024),
            "savings_percentage": (1.0 - compression_ratio) * 100.0,
        }
