"""
JLLM Data Extractor
====================
Converts HuggingFace Safetensors to custom .jllm format.
"""

import os
import sys
import json
import torch
import gc
from safetensors import safe_open


def auto_detect_model_type(model_dir):
    """Auto-detect model type from config.json."""
    config_paths = [
        os.path.join(model_dir, "config.json"),
        os.path.join(model_dir, "model.safetensors.index.json"),
    ]

    for config_path in config_paths:
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)

                model_type = config.get("model_type") or config.get("architectures", [None])[0]
                if model_type:
                    print(f"[Auto-Detect] model_type: {model_type}")
                    return model_type, config
            except Exception as e:
                print(f"[Auto-Detect] Warning: Could not read {config_path}: {e}")

    print("[Auto-Detect] Warning: Could not auto-detect, using 'Unknown'")
    return "Unknown", {}


def extract_architecture_from_config(config):
    """Extract architecture parameters from config.json."""
    arch = {}
    arch['hidden_size'] = config.get('hidden_size', 4096)
    arch['num_hidden_layers'] = config.get('num_hidden_layers', 32)
    arch['num_attention_heads'] = config.get('num_attention_heads', 32)
    arch['num_key_value_heads'] = config.get('num_key_value_heads', 8)
    arch['vocab_size'] = config.get('vocab_size', 32000)
    arch['rope_theta'] = config.get('rope_theta', 10000.0)
    arch['intermediate_size'] = config.get('intermediate_size', arch['hidden_size'] * 4)
    
    # Dynamically calculate head_dim if not explicitly stated, rather than hardcoding 128
    arch['head_dim'] = config.get('head_dim', arch['hidden_size'] // arch['num_attention_heads'])

    print(f"[Auto-Detect] hidden_size: {arch['hidden_size']}")
    print(f"[Auto-Detect] num_layers: {arch['num_hidden_layers']}")
    print(f"[Auto-Detect] num_heads: {arch['num_attention_heads']}")
    print(f"[Auto-Detect] num_kv_heads: {arch['num_key_value_heads']}")
    print(f"[Auto-Detect] head_dim: {arch['head_dim']}")
    print(f"[Auto-Detect] vocab_size: {arch['vocab_size']}")
    print(f"[Auto-Detect] rope_theta: {arch['rope_theta']}")

    return arch


def extract_weights(model_dir, output_file):
    """
    Extract weights from Safetensors to custom .jllm format.

    Args:
        model_dir: Directory containing .safetensors files
        output_file: Output .jllm file path
    """
    print(f"[Extractor] Scanning model directory: {model_dir}")

    # Find all .safetensors files
    tensor_files = [f for f in os.listdir(model_dir) if f.endswith(".safetensors")]
    if not tensor_files:
        raise FileNotFoundError(
            f"[Extractor] Error: No .safetensors files found in {model_dir}"
        )
    print(f"[Extractor] Found {len(tensor_files)} weight file(s)")

    # Auto-detect model type and architecture
    model_type, config = auto_detect_model_type(model_dir)
    arch = extract_architecture_from_config(config)

    # Initialize header
    jllm_header = {
        "model_type": model_type,
        "architecture": arch,
        "tensors": {}
    }

    # Create output file
    HEADER_SIZE = 1024 * 1024  # 1MB for header
    current_offset = HEADER_SIZE

    with open(output_file, "wb") as out_f:
        # Reserve header space
        out_f.write(b"\x00" * HEADER_SIZE)

        # Process each safetensors file
        for file_name in tensor_files:
            file_path = os.path.join(model_dir, file_name)
            print(f"[Extractor] Processing: {file_name}...")

            with safe_open(file_path, framework="pt", device="cpu") as f:
                for tensor_name in f.keys():
                    
                    tensor = f.get_tensor(tensor_name)

                    # Convert to float16 (most common for inference)
                    tensor_data = tensor.to(torch.float16).numpy().tobytes()
                    tensor_size = len(tensor_data)

                    # 1. FIX: Safer name cleaning. 
                    # replace("model.", "") would corrupt names like "layers.0.model_proj" 
                    clean_name = tensor_name
                    if clean_name.startswith("model."):
                        clean_name = clean_name[6:]

                    # 2. FIX: Memory Alignment for zero-copy mmap.
                    # GPUs prefer memory accesses aligned to 64-byte boundaries. 
                    # If we don't pad, unaligned tensors can cause massive slowdowns or segfaults during inference.
                    alignment_padding = (64 - (current_offset % 64)) % 64
                    if alignment_padding > 0:
                        out_f.write(b"\x00" * alignment_padding)
                        current_offset += alignment_padding

                    # Record in header
                    jllm_header["tensors"][clean_name] = {
                        "shape": list(tensor.shape),
                        "offset": current_offset,
                        "size": tensor_size,
                        "dtype": "float16" # Explicitly define dtype for JLLMLoader
                    }

                    # Write to file
                    out_f.write(tensor_data)
                    current_offset += tensor_size

                    # Clean up
                    del tensor, tensor_data

            gc.collect()

        # Write header at the beginning
        out_f.seek(0)
        header_bytes = json.dumps(jllm_header, ensure_ascii=False).encode('utf-8')

        if len(header_bytes) > HEADER_SIZE:
            raise RuntimeError(
                f"[Extractor] Error: Header size ({len(header_bytes)} bytes) "
                f"exceeds reserved space ({HEADER_SIZE} bytes)"
            )

        # Write header, pad with null bytes
        out_f.write(header_bytes)
        # Note: We don't overwrite the entire 1MB block here, just up to HEADER_SIZE
        # because the initial reservation handles the rest.

    print(f"\n[Extractor] Success! Output: {output_file}")
    print(f"[Extractor] Total tensors extracted: {len(jllm_header['tensors'])}")


if __name__ == "__main__":
    # Configuration - MODIFY THESE for your model
    MODEL_PATH = r"C:\Users\j_lam\OneDrive\桌面\JProjects\JCUDA\src V5\src\Models\qwen"  # Directory with .safetensors
    OUTPUT_JLLM = "jqwen.jllm"

    if len(sys.argv) >= 3:
        MODEL_PATH = sys.argv[1]
        OUTPUT_JLLM = sys.argv[2]

    if os.path.exists(MODEL_PATH):
        extract_weights(MODEL_PATH, OUTPUT_JLLM)
    else:
        print(f"[Extractor] Error: Model path not found: {MODEL_PATH}")
        print("Usage: python JLLMDataExtractor.py <model_dir> <output.jllm>")