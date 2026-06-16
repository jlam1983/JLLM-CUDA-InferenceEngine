from __future__ import annotations

import gc
import json
import mmap
import os
from pathlib import Path
from typing import Iterator

import numpy as np


class JLLMReader:
    """
    Memory-mapped JLLM loader — zero-copy header, safer tensor reconstruction.
    """

    HEADER_SIZE = 2 * 1024 * 1024  # 2 MB

    def __init__(self, path: str | Path):
        """
        Summary: Initialize the JLLMReader with a file path, opening and memory-mapping the file.
        理論描述: 開啟指定的 JLLM 檔案並建立記憶體映射，解析表頭元數據以供後續讀取使用。
        """
        self.path = Path(path)
        self._file = None
        self._mmap = None
        self._header: dict | None = None
        self._tensors_meta: dict[str, dict] = {}
        self._model_type: str | None = None
        self._architecture: dict | None = None
        self._open()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read_tensor(self, name: str, device: str = "cpu", *, as_numpy: bool = False):
        """
        Summary: Read and reconstruct a tensor by name, supporting both CPU and GPU reconstruction paths.
        理論描述: 根據張量名稱讀取並重建張量，自動處理離群值與常規值的分離重建邏輯。
        """
        meta = self._tensors_meta.get(name)
        if meta is None:
            raise KeyError(f"Tensor '{name}' not found. Available: {list(self._tensors_meta.keys())}")

        shape = meta["shape"]
        offset = meta["data_offset"]
        sub = meta["sub_offsets"]

        numel = int(np.prod(shape))
        outlier_count = (sub["outliers"][1] - sub["outliers"][0]) // 2

        if device == "cuda":
            return self._read_tensor_gpu(numel, outlier_count, offset, sub, shape, as_numpy=as_numpy)
        return self._read_tensor_cpu(numel, outlier_count, offset, sub, shape)

    # ------------------------------------------------------------------
    # CPU reconstruction (numpy)
    # ------------------------------------------------------------------

    def _read_tensor_cpu(self, numel: int, outlier_count: int, base_offset: int, sub: dict, shape: list):
        """
        Summary: Reconstruct a tensor on CPU using NumPy, unmixing outliers and quantized normal values.
        理論描述: 在 CPU 上重建張量，透過 mask 分离離群值與常規值，並使用量化映射表還原常規數值。
        """
        # Safer buffer extraction
        def get_buffer(offset: int, length: int):
            return self._mmap[base_offset + offset : base_offset + offset + length]

        # Outliers (float16)
        outliers = np.frombuffer(
            get_buffer(*sub["outliers"]), dtype=np.float16
        ).copy()

        # Mapping (float16)
        mapping = np.frombuffer(
            get_buffer(*sub["mapping"]), dtype=np.float16
        ).copy()

        # Normal indices (uint8)
        normal_indices = np.frombuffer(
            get_buffer(*sub["normal"]), dtype=np.uint8
        ).copy()

        # Mask
        mask_bytes = get_buffer(*sub["mask"])
        full_mask = np.unpackbits(np.frombuffer(mask_bytes, dtype=np.uint8), bitorder='big')
        full_mask = full_mask[:numel].astype(np.bool_)

        # Validation
        if len(full_mask) < numel or outlier_count < 0 or outlier_count > numel:
            raise ValueError(f"Data length mismatch for tensor (numel={numel}, outlier_count={outlier_count})")

        # Reconstruct
        sort_idx = np.argsort(full_mask, kind="stable")
        normal_positions = sort_idx[:numel - outlier_count]
        outlier_positions = sort_idx[numel - outlier_count:]

        result = np.empty(numel, dtype=np.float16)
        result[normal_positions] = mapping[normal_indices]
        result[outlier_positions] = outliers

        return result.reshape(shape)

    # ------------------------------------------------------------------
    # GPU reconstruction (cupy)
    # ------------------------------------------------------------------

    def _read_tensor_gpu(self, numel: int, outlier_count: int, base_offset: int, sub: dict, shape: list, as_numpy: bool):
        """
        Summary: Reconstruct a tensor on GPU using CuPy, enabling faster reconstruction for large tensors.
        理論描述: 在 GPU 上使用 CuPy 重建張量，透過記憶體映射直接讀取資料並利用 GPU 並行運算加速還原。
        """
        try:
            import cupy as cp
        except ImportError:
            raise ImportError("cupy is required for GPU reconstruction")

        def get_buffer(offset: int, length: int):
            return self._mmap[base_offset + offset : base_offset + offset + length]

        outliers = cp.frombuffer(get_buffer(*sub["outliers"]), dtype=np.float16).copy()
        mapping = cp.frombuffer(get_buffer(*sub["mapping"]), dtype=np.float16).copy()
        normal_indices = cp.frombuffer(get_buffer(*sub["normal"]), dtype=np.uint8).copy()

        mask_bytes = get_buffer(*sub["mask"])
        full_mask = cp.unpackbits(
            cp.frombuffer(mask_bytes, dtype=np.uint8), bitorder='big'
        )[:numel].astype(cp.bool_)

        if len(full_mask) < numel or outlier_count < 0 or outlier_count > numel:
            raise ValueError(f"Data length mismatch for tensor (numel={numel}, outlier_count={outlier_count})")

        sort_idx = cp.argsort(full_mask, kind="stable")
        normal_positions = sort_idx[:numel - outlier_count]
        outlier_positions = sort_idx[numel - outlier_count:]

        result = cp.empty(numel, dtype=np.float16)
        result[normal_positions] = mapping[normal_indices]
        result[outlier_positions] = outliers

        if as_numpy:
            return cp.asnumpy(result.reshape(shape))
        return result.reshape(shape)

    # ------------------------------------------------------------------
    # Raw access
    # ------------------------------------------------------------------

    def read_tensor_raw(self, name: str) -> dict:
        """
        Summary: Return raw uncompressed data for a tensor without reconstructing the full values.
        理論描述: 回傳未重建的原始壓縮資料，包含離群值、遮罩、索引及量化映射表，供高階應用使用。
        """
        meta = self._tensors_meta.get(name)
        if meta is None:
            raise KeyError(f"Tensor '{name}' not found.")

        offset = meta["data_offset"]
        sub = meta["sub_offsets"]
        numel = int(np.prod(meta["shape"]))

        mask_bytes = self._mmap[offset + sub["mask"][0]: offset + sub["mask"][1]]
        full_mask = np.unpackbits(np.frombuffer(mask_bytes, dtype=np.uint8), bitorder='big')[:numel].astype(np.bool_)

        outliers_buf = self._mmap[offset + sub["outliers"][0]: offset + sub["outliers"][1]]
        outliers = np.frombuffer(outliers_buf, dtype=np.float16).copy() if len(outliers_buf) > 0 else np.array([], dtype=np.float16)

        return {
            "shape": meta["shape"],
            "outliers": outliers,
            "outlier_mask": full_mask,
            "normal_indices": np.frombuffer(
                self._mmap[offset + sub["normal"][0]: offset + sub["normal"][1]],
                dtype=np.uint8).copy(),
            "mapping": np.frombuffer(
                self._mmap[offset + sub["mapping"][0]: offset + sub["mapping"][1]],
                dtype=np.float16).copy(),
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def iter_tensors(self, names=None) -> Iterator[tuple[str, np.ndarray]]:
        """
        Summary: Iterate over tensors by name, yielding (name, tensor) pairs with memory cleanup.
        理論描述: 逐一讀取指定名稱的張量並回傳，每次讀取後主動釋放記憶體以避免累積。
        """
        targets = names if names is not None else list(self._tensors_meta.keys())
        for name in targets:
            tensor = self.read_tensor(name)
            yield name, tensor
            del tensor
            gc.collect()

    def tensor_shape(self, name: str) -> list[int]:
        """
        Summary: Return the shape of a tensor by name.
        理論描述: 查詢指定張量的維度形狀資訊。
        """
        return self._tensors_meta[name]["shape"]

    def architecture(self) -> dict:
        """
        Summary: Return a copy of the model architecture metadata.
        理論描述: 回傳模型架構資訊的副本，包含隱藏層大小、層數、注意力頭數等。
        """
        return dict(self._architecture)

    def model_type(self) -> str:
        """
        Summary: Return the model type string (e.g., "Qwen2").
        理論描述: 回傳模型類型識別字串。
        """
        return self._model_type

    def tensor_names(self) -> list[str]:
        """
        Summary: Return a list of all tensor names stored in the file.
        理論描述: 回傳檔案中所有已儲存張量的名稱列表。
        """
        return list(self._tensors_meta.keys())

    def close(self):
        """
        Summary: Close the memory map and file handle, releasing all resources.
        理論描述: 關閉記憶體映射與檔案控制代碼，釋放所有相關資源。
        """
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if self._file is not None:
            self._file.close()
            self._file = None
        self._header = None
        self._tensors_meta = {}
        self._model_type = None
        self._architecture = None

    def __enter__(self):
        """
        Summary: Context manager entry, returns self for `with` statement usage.
        理論描述: 上下文管理器入口，回傳 self 以支援 `with` 語法。
        """
        return self

    def __exit__(self, *args):
        """
        Summary: Context manager exit, ensures resources are released on block exit.
        理論描述: 上下文管理器出口，區塊結束時自動呼叫 close 釋放資源。
        """
        self.close()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _open(self):
        """
        Summary: Internal method to open the file, memory-map it, and parse the JSON header.
        理論描述: 內部方法，負責開啟檔案、建立記憶體映射並解析 JSON 表頭以讀取元數據。
        """
        if not self.path.exists():
            raise FileNotFoundError(f"JLLM file not found: {self.path}")

        size = os.path.getsize(self.path)
        if size < self.HEADER_SIZE:
            raise ValueError(f"File too small ({size} bytes)")

        self._file = open(self.path, "rb")
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

        header_bytes = self._mmap[:self.HEADER_SIZE].rstrip(b"\x00")
        self._header = json.loads(header_bytes.decode("utf-8"))

        self._tensors_meta = self._header.get("tensors", {})
        if not self._tensors_meta:
            raise ValueError("No tensors in JLLM header.")

        self._model_type = self._header.get("model_type", "Unknown")
        self._architecture = dict(self._header.get("architecture", {}))
        self._header = None   # Free memory