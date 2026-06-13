"""
JLLM GPU Inference Entry Point
==============================
GPU-accelerated inference using CUDA.
"""

import os
import sys
import numpy as np
import torch
from transformers import AutoTokenizer

# Import engine components
from JLLMLoader import JLLMLoader
from JLLMInferenceEngine import *
from JLLMFileCacheConverter import JLLMFileCacheConverter


def save_all_layers(loader, num_layers):
    """Save all layers to individual layer cache files.

    Args:
        loader: JLLMLoader instance
        num_layers: Total number of layers to save
    """
    for layer_idx in range(num_layers):
        cache_path = loader.tensorManager._layer_cache_path(layer_idx)
        bin_path = cache_path + ".bin"
        meta_path = cache_path + ".meta.pt"

        if os.path.exists(bin_path) and os.path.exists(meta_path):
            print(f"[JTensorManager] Layer {layer_idx} cache already exists, skipping.")
            continue

        print(f"[JTensorManager] Saving layer {layer_idx}/{num_layers-1}...")
        loader.tensorManager.save_layer_cache(layer_idx, loader)

    print(f"[JTensorManager] All {num_layers} layers saved.")


def main():
    """Main entry point for GPU inference."""

    # Configuration
    tokenizer_name = r"C:\Users\j_lam\OneDrive\桌面\JProjects\JCUDA\src V5\src\Models\qwen"
    JLLM_FILE = "jqwen.jllm"

    # Check if model file exists
    if not os.path.exists(JLLM_FILE):
        print(f"[Error] Model file not found: {JLLM_FILE}")
        print("Usage: python JLLMInferenceEngineGPU.py wait<model.jllm> [tokenizer]")
        return

    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    # Determine cache mode
    cache_mode = sys.argv[3] if len(sys.argv) > 3 else "full_gpu"
    print(f"Cache mode: {cache_mode}")

    # Create engine
    engine = JLLMInferenceEngine(
        model_path=JLLM_FILE,
        tokenizer=tokenizer,
        cache_mode="full_gpu",
        determinism_mode="fast"
        #determinism_mode="strict"
        #determinism_mode="deterministic"
    )

    # Optional: Save all layers to cache files
    # Usage: python JLLMInferenceEngineGPU.py save_all_layers
    if len(sys.argv) > 1 and sys.argv[1] == "save_all_layers":
        save_all_layers(engine.loader, engine.num_layers)
        return

    # Default prompt
    prompt = "IT technology trend is"

    # Run inference
    engine.generate_stream(prompt=prompt)


if __name__ == "__main__":
    main()
