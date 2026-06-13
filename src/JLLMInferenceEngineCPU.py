"""
JLLM CPU Inference Entry Point
==============================
CPU-only inference (no CUDA dependency).
"""

import os
import sys
import torch
from transformers import AutoTokenizer

from JLLMLoader import JLLMLoader
from JLLMInferenceEngine import JLLMInferenceEngine


def main():
    """Main entry point for CPU inference."""

    # Configuration
    JLLM_FILE = "qwen_7b.jllm"
    tokenizer_name = "Qwen/Qwen2.5-7B-Instruct"

    # Allow override via command line
    if len(sys.argv) > 1:
        JLLM_FILE = sys.argv[1]
    if len(sys.argv) > 2:
        tokenizer_name = sys.argv[2]

    if not os.path.exists(JLLM_FILE):
        print(f"[Error] Model file not found: {JLLM_FILE}")
        print("Usage: python JLLMInferenceEngineCPU.py <model.jllm> [tokenizer]")
        return

    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    # Create engine on CPU
    engine = JLLMInferenceEngine(
        model_path=JLLM_FILE,
        tokenizer=tokenizer,
        device="cpu",
        cache_mode="no_cache"  # CPU doesn't need GPU cache modes
    )

    prompt = "IT technology trend is"
    engine.generate_stream(prompt=prompt)


if __name__ == "__main__":
    main()
