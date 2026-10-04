"""
Needle-In-A-Haystack (NIAH) Benchmark Generator & Evaluator
===========================================================
Generates long-context evaluation prompts with secret needles placed at parameterized depths
(e.g., 10%, 25%, 50%, 75%, 90%) across sequence lengths from 4k to 64k+ tokens.
"""

import math
import random
from typing import Any, Dict, List, Optional, Tuple


HAYSTACK_ESSAY = """The exploration of complex distributed computing architectures reveals significant trade-offs
between latency, throughput, and state persistence. When designing high-performance systems,
engineers frequently encounter the challenge of state synchronization across heterogeneous compute nodes.
In modern distributed training and inference clusters, interconnect bandwidth frequently forms the primary bottleneck.
Hardware acceleration units such as tensor processing cores and high-bandwidth memory (HBM) modules require
carefully orchestrated data pipelines to prevent compute starvation. As memory hierarchies continue to evolve,
the distinction between local cache and remote pooled memory becomes increasingly fluid. Communication primitives,
including all-reduce, all-gather, and scatter-gather operations, must be scheduled with microsecond precision.
Fault tolerance strategies must gracefully handle transient node dropouts without incurring prohibitive checkpointing overhead.
Furthermore, asynchronous execution graphs enable overlapping of communication with compute kernels, thereby maximizing
hardware utilization rates across thousands of parallel processor cores."""


class NeedleHaystackBenchmark:
    """Generates and evaluates Needle-In-A-Haystack retrieval experiments."""

    def __init__(
        self,
        needle: str = "The secret passkey for the DeepSpace Project is 938472.",
        retrieval_query: str = "What is the secret passkey for the DeepSpace Project? Answer:",
        expected_answer: str = "938472",
        seed: int = 42,
    ):
        self.needle = needle
        self.retrieval_query = retrieval_query
        self.expected_answer = expected_answer
        self.seed = seed
        random.seed(seed)

    def generate_prompt(self, target_token_len: int, depth_ratio: float, tokenizer: Any = None) -> Tuple[str, int]:
        """Generates a text prompt of approximately `target_token_len` tokens with the needle at `depth_ratio`.

        Args:
            target_token_len: Approximate target token length (e.g. 8192, 16384, 32768, 65536)
            depth_ratio: Float in [0.0, 1.0] specifying where the needle is inserted.
            tokenizer: Optional HuggingFace tokenizer. If None, uses word approximation (~1.3 tokens/word).

        Returns:
            Tuple of (full_prompt_text, actual_token_count)
        """
        # Estimate words needed
        words_per_token = 0.75  # 1 token ~ 0.75 words
        target_words = int(target_token_len * words_per_token)

        base_words = HAYSTACK_ESSAY.split()
        repetitions = math.ceil(target_words / len(base_words)) + 1
        full_haystack = (base_words * repetitions)[:target_words]

        # Insert needle at specified depth
        insert_idx = int(len(full_haystack) * depth_ratio)
        needle_words = self.needle.split()
        haystack_with_needle = full_haystack[:insert_idx] + needle_words + full_haystack[insert_idx:]

        # Append query prompt
        query_words = self.retrieval_query.split()
        final_words = haystack_with_needle + query_words
        prompt_text = " ".join(final_words)

        if tokenizer is not None:
            tokens = tokenizer.encode(prompt_text)
            actual_count = len(tokens)
        else:
            actual_count = int(len(final_words) / words_per_token)

        return prompt_text, actual_count

    def check_answer(self, generated_text: str) -> bool:
        """Checks if the generated text contains the correct needle passkey."""
        return self.expected_answer.lower() in generated_text.lower()
