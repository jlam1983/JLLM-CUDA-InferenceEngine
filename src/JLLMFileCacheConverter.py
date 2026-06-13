"""
JLLM KV Cache Converter
=======================
Handles KV cache compression and decompression with disk storage.
"""

import os
import numpy as np
import torch
from mmap import mmap, ACCESS_READ


class JLLMFileCacheConverter:
    """
    Manages compressed KV cache storage to disk.

    Storage Format:
        cache_store/
        ├── layer_0_k_block.jbin    # Key blocks
        ├── layer_0_k_inv.jbin      # Key indices
        ├── layer_0_v_block.jbin    # Value blocks
        ├── layer_0_v_inv.jbin      # Value indices
        ...

    The converter uses a simple block-wise compression with inverse indices
    for random access during retrieval.
    """

    MAX_WINDOWS_PER_LAYER = 256

    def __init__(self, storage_dir="./cache_store", run_device="cuda"):
        self.storage_dir = storage_dir
        self.device = run_device
        self.interval = 2  # Compression interval
        self.read_only = False

        os.makedirs(self.storage_dir, exist_ok=True)

        # File registry: {layer_idx: {path_k_block, path_k_inv, ...}}
        self.file_registry = {"global": None, "layers": {}}

    def save_global_root_cache(self, k_tensor, v_tensor):
        """Save root KV cache (for system prompt)."""
        pass  # Not implemented yet

    def save_layer_compressed_cache(self, layer_idx, key_tensor, value_tensor, single_token=False):
        """
        Save compressed KV cache for a layer.

        Args:
            layer_idx: Layer index
            key_tensor: Key tensor [batch, seq, num_kv_heads, head_dim]
            value_tensor: Value tensor [batch, seq, num_kv_heads, head_dim]
            single_token: If True, save single token instead of block
        """
        if self.read_only:
            return

        path_prefix = os.path.join(self.storage_dir, f"layer_{layer_idx}")
        paths = {
            "k_block": f"{path_prefix}_k_block.jbin",
            "k_inv": f"{path_prefix}_k_inv.jbin",
            "v_block": f"{path_prefix}_v_block.jbin",
            "v_inv": f"{path_prefix}_v_inv.jbin"
        }

        # Convert to numpy (float16)
        k_cpu = key_tensor.half().cpu().numpy()
        v_cpu = value_tensor.half().cpu().numpy()

        k_flat = k_cpu.flatten()
        v_flat = v_cpu.flatten()

        # Number of windows
        n_windows = 1 if single_token else k_flat.shape[0] // self.interval
        if n_windows == 0:
            return

        # Get current count for index offset
        layer_info = self.file_registry["layers"].get(layer_idx, {})
        current_total = layer_info.get("total_windows", 0)

        # Create inverse indices (offset by current total)
        inv_indices = np.arange(current_total, current_total + n_windows, dtype=np.int32)

        # Convert to bytes
        k_block_bytes = k_flat.tobytes()
        v_block_bytes = v_flat.tobytes()
        k_inv_bytes = inv_indices.tobytes()
        v_inv_bytes = inv_indices.tobytes()

        data_map = {
            "k_block": k_block_bytes,
            "k_inv": k_inv_bytes,
            "v_block": v_block_bytes,
            "v_inv": v_inv_bytes
        }

        if layer_idx in self.file_registry["layers"]:
            # Close existing mmaps if cached
            for key in ["mmap_k_block", "mmap_k_inv", "mmap_v_block", "mmap_v_inv"]:
                if key in layer_info and hasattr(layer_info[key], '_mmap'):
                    layer_info[key]._mmap.close()
                    del layer_info[key]

            # Check for context overflow
            if current_total + n_windows > self.MAX_WINDOWS_PER_LAYER:
                print(f"[Cache] Layer {layer_idx} exceeded MAX_WINDOWS. Wiping.")
                for key, data in data_map.items():
                    with open(paths[key], "wb") as f:
                        f.write(data)
                layer_info["total_windows"] = n_windows
            else:
                # Append to existing files
                for key, data in data_map.items():
                    with open(paths[key], "ab") as f:
                        f.write(data)
                layer_info["total_windows"] += n_windows

            layer_info["is_dirty"] = True
        else:
            # First time for this layer
            for key, data in data_map.items():
                with open(paths[key], "wb") as f:
                    f.write(data)

            self.file_registry["layers"][layer_idx] = {
                "total_windows": n_windows,
                "is_dirty": False,
                "path_k_block": paths["k_block"],
                "path_k_inv": paths["k_inv"],
                "path_v_block": paths["v_block"],
                "path_v_inv": paths["v_inv"]
            }

    def get_incremental_prefill_decompressed(self, layer_idx, window_start, window_end):
        """
        Load and decompress KV cache for a layer.

        Args:
            layer_idx: Layer index
            window_start: Start window index
            window_end: End window index

        Returns:
            Tuple of (k_tensor, v_tensor) or (None, None) if not available
        """
        if layer_idx not in self.file_registry["layers"]:
            return None, None

        layer_files = self.file_registry["layers"][layer_idx]
        total_windows = layer_files.get("total_windows", 0)

        # Adjust bounds
        effective_start = max(0, total_windows - self.MAX_WINDOWS_PER_LAYER)
        effective_end = total_windows

        n_requested = effective_end - effective_start
        if n_requested <= 0:
            return None, None

        # Open file descriptors
        fd_k_inv = os.open(layer_files["path_k_inv"], os.O_RDONLY)
        fd_v_inv = os.open(layer_files["path_v_inv"], os.O_RDONLY)
        fd_k_block = os.open(layer_files["path_k_block"], os.O_RDONLY)
        fd_v_block = os.open(layer_files["path_v_block"], os.O_RDONLY)

        try:
            # Memory map files
            mm_k_inv = mmap(fd_k_inv, 0, access=ACCESS_READ)
            mm_v_inv = mmap(fd_v_inv, 0, access=ACCESS_READ)
            mm_k_block = mmap(fd_k_block, 0, access=ACCESS_READ)
            mm_v_block = mmap(fd_v_block, 0, access=ACCESS_READ)

            # Read inverse indices
            inv_k_view = np.frombuffer(mm_k_inv, dtype=np.int32)[effective_start:effective_end]
            inv_v_view = np.frombuffer(mm_v_inv, dtype=np.int32)[effective_start:effective_end]

            # Convert to PyTorch tensors
            inv_k_t = torch.from_numpy(inv_k_view).long()
            inv_v_t = torch.from_numpy(inv_v_view).long()

            # Read blocks
            block_k_view = torch.frombuffer(mm_k_block, dtype=torch.float16)
            block_v_view = torch.frombuffer(mm_v_block, dtype=torch.float16)

            # Calculate window size
            window_size = block_k_view.numel() // max(1, total_windows)

            # Reshape to 2D
            block_k_2d = block_k_view.view(total_windows, window_size)
            block_v_2d = block_v_view.view(total_windows, window_size)

            # Reconstruct via advanced indexing
            recon_k = block_k_2d[inv_k_t].cuda(non_blocking=True)
            recon_v = block_v_2d[inv_v_t].cuda(non_blocking=True)

            # Synchronize
            torch.cuda.current_stream().synchronize()

        finally:
            # Close mmaps and file descriptors
            mm_k_inv.close()
            mm_v_inv.close()
            mm_k_block.close()
            mm_v_block.close()
            os.close(fd_k_inv)
            os.close(fd_v_inv)
            os.close(fd_k_block)
            os.close(fd_v_block)

        return recon_k.flatten(), recon_v.flatten()
