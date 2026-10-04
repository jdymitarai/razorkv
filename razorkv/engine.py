"""
RazorKV Inference Engine & Model Patcher
========================================
Integrates RazorKV dynamically with HuggingFace AutoModelForCausalLM models
(LLaMA-3, Qwen-2.5, Mistral, DeepSeek-R1, etc.) for plug-and-play inference.
"""

import contextvars
from typing import Any, Dict, List, Optional, Union
import torch
from razorkv.config import RazorConfig
from razorkv.cache import RazorKVCache


class RazorEngine:
    """High-level engine to manage, patch, and execute LLM inference with RazorKV."""

    _active_cache: contextvars.ContextVar = contextvars.ContextVar("active_razorkv_cache", default=None)

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
        Stores original methods and hook handles to allow clean unpatching via `unpatch_model`.
        """
        razor_cfg = config or RazorConfig()
        num_layers = getattr(
            getattr(model, "config", None),
            "num_hidden_layers",
            getattr(getattr(model, "config", None), "n_layer", 32),
        )

        # Preserve original methods if not already patched
        if not hasattr(model, "_orig_generate"):
            model._orig_generate = getattr(model, "generate", None)
            model._orig_prepare_inputs_for_generation = getattr(model, "prepare_inputs_for_generation", None)
            model._orig_forward = getattr(model, "forward", None)
            model._razorkv_hook_handles = []
            model._is_razorkv_patched = True

        orig_gen = model._orig_generate
        orig_prepare = model._orig_prepare_inputs_for_generation
        orig_forward = model._orig_forward

        # 1. Patch generate to auto-inject RazorKVCache
        if orig_gen is not None:
            def razor_generate(*args, **kwargs):
                if kwargs.get("past_key_values") is None:
                    kwargs["past_key_values"] = RazorKVCache(config=razor_cfg, num_layers=num_layers)
                return orig_gen(*args, **kwargs)

            model.generate = razor_generate

        # 2. Patch prepare_inputs_for_generation for direct caller compatibility
        if orig_prepare is not None:
            def razor_prepare_inputs_for_generation(input_ids, past_key_values=None, **kwargs):
                if past_key_values is None:
                    past_key_values = RazorKVCache(config=razor_cfg, num_layers=num_layers)
                return orig_prepare(input_ids, past_key_values=past_key_values, **kwargs)

            model.prepare_inputs_for_generation = razor_prepare_inputs_for_generation

        # 3. Patch forward to set active cache context for query resonance hooks
        if orig_forward is not None:
            def razor_forward(*args, **kwargs):
                pkv = kwargs.get("past_key_values", None)
                if pkv is not None and isinstance(pkv, RazorKVCache):
                    token = cls._active_cache.set(pkv)
                    try:
                        return orig_forward(*args, **kwargs)
                    finally:
                        cls._active_cache.reset(token)
                return orig_forward(*args, **kwargs)

            model.forward = razor_forward

        # 4. Attach forward hooks on attention query projections for dynamic resonance tracking
        layers = getattr(model, "layers", None)
        if layers is None and hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
        elif layers is None and hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            layers = model.transformer.h

        if layers is not None and hasattr(model, "config"):
            num_q_heads = getattr(model.config, "num_attention_heads", 32)
            head_dim = getattr(
                model.config,
                "head_dim",
                getattr(model.config, "hidden_size", 4096) // max(1, num_q_heads),
            )

            for layer_idx, layer in enumerate(layers):
                attn = getattr(layer, "self_attn", getattr(layer, "attn", None))
                if attn is not None and hasattr(attn, "q_proj"):
                    def _make_q_hook(l_idx, n_heads, h_dim):
                        def _q_hook(module, inp, out):
                            cache = cls._active_cache.get()
                            if cache is not None and isinstance(cache, RazorKVCache):
                                b = out.shape[0]
                                q_len = out.shape[1]
                                q = out.view(b, q_len, n_heads, h_dim).transpose(1, 2)
                                cache.update_salience_with_query(q, l_idx)
                        return _q_hook

                    handle = attn.q_proj.register_forward_hook(_make_q_hook(layer_idx, num_q_heads, head_dim))
                    model._razorkv_hook_handles.append(handle)

        model._razorkv_config = razor_cfg
        return model

    @classmethod
    def unpatch_model(cls, model: Any) -> Any:
        """Restores original HuggingFace model generation behavior and removes hooks."""
        if hasattr(model, "_orig_generate") and model._orig_generate is not None:
            model.generate = model._orig_generate
            del model._orig_generate
        if hasattr(model, "_orig_prepare_inputs_for_generation") and model._orig_prepare_inputs_for_generation is not None:
            model.prepare_inputs_for_generation = model._orig_prepare_inputs_for_generation
            del model._orig_prepare_inputs_for_generation
        if hasattr(model, "_orig_forward") and model._orig_forward is not None:
            model.forward = model._orig_forward
            del model._orig_forward
        if hasattr(model, "_razorkv_hook_handles"):
            for h in model._razorkv_hook_handles:
                h.remove()
            model._razorkv_hook_handles.clear()
            del model._razorkv_hook_handles
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
