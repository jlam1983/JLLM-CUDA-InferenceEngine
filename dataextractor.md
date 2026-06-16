# JLLMReader Data Flow Documentation
# JLLMReader 資料流文檔

## Overview | 概述

```
Memory-mapped JLLM loader — zero-copy header, safer tensor reconstruction.
記憶體映射 JLLM 載入器 — 零拷貝表頭，更安全的張量重建。
```

**理論描述: JLLMReader 負責記憶體映射讀取 .jllm 檔案格式，解析表頭元數據並提供張量的讀取與重建功能。**

---

## 1. Initialization | 初始化

### 1.1 Class Structure | 類別結構

```
class JLLMReader:
    """
    Memory-mapped JLLM loader — zero-copy header, safer tensor reconstruction.
    """
```

### 1.2 Open File | 開啟檔案

```
# ------------------------------------------------------------------
#  Initialization
# ------------------------------------------------------------------

def _open(self):
    """
    Summary: Internal method to open the file, memory-map it, and parse the JSON header.
    理論描述: 內部方法，負責開啟檔案、建立記憶體映射並解析 JSON 表頭以讀取元數據。
    """
    # 開啟檔案、建立 mmap、解析 JSON 表頭
```

**理論描述: _open 方法負責開啟指定的 JLLM 檔案並建立記憶體映射，解析表頭元數據以供後續讀取使用。**

### 1.3 Header Parsing | 表頭解析

```
HEADER_SIZE = 2 * 1024 * 1024  # 2 MB

# 表頭解析流程:
# 1. 開啟檔案控制代碼
# 2. 建立記憶體映射 (mmap)
# 3. 讀取並解析 JSON 表頭
# 4. 提取張量元數據 (tensors_meta)
# 5. 提取模型類型 (model_type)
# 6. 提取架構資訊 (architecture)
```

---

## 2. Public API | 公開 API

### 2.1 read_tensor | 讀取張量

```
# ------------------------------------------------------------------
#  Public API
# ------------------------------------------------------------------

def read_tensor(self, name: str, device: str = "cpu", *, as_numpy: bool = False):
    """
    Summary: Read and reconstruct a tensor by name, supporting both CPU and GPU reconstruction paths.
    理論描述: 根據張量名稱讀取並重建張量，自動處理離群值與常規值的分離重建邏輯。
    """
```

**理論描述: read_tensor 根據張量名稱讀取並重建張量，自動處理離群值與常規值的分離重建邏輯。**

**資料流:**
```
name ──▶ 查詢 tensors_meta ──▶ 取得 shape, offset, sub_offsets
                            │
                            ▼
                   ┌────────────────────┐
                   │ 計算 numel,        │
                   │ outlier_count      │
                   └────────────────────┘
                            │
                            ▼
                   ┌────────────────────┐
                   │ device == "cuda"? │
                   └────────────────────┘
                        │         │
                       Yes        No
                        │         │
                        ▼         ▼
              _read_tensor_gpu   _read_tensor_cpu
```

### 2.2 read_tensor_raw | 原始資料讀取

```
# ------------------------------------------------------------------
#  Raw access
# ------------------------------------------------------------------

def read_tensor_raw(self, name: str) -> dict:
    """
    Summary: Return raw uncompressed data for a tensor without reconstructing the full values.
    理論描述: 回傳未重建的原始壓縮資料，包含離群值、遮罩、索引及量化映射表，供高階應用使用。
    """
```

**理論描述: read_tensor_raw 回傳未重建的原始壓縮資料，包含離群值、遮罩、索引及量化映射表，供高階應用使用。**

**回傳結構:**
```python
{
    "shape": meta["shape"],
    "outliers": outliers,           # float16 離群值
    "outlier_mask": full_mask,      # bool 遮罩陣列
    "normal_indices": ...,           # uint8 常規索引
    "mapping": ...,                  # float16 量化映射表
}
```

### 2.3 Utility Methods | 工具方法

```
def iter_tensors(self, names=None) -> Iterator[tuple[str, np.ndarray]]:
    """
    Summary: Iterate over tensors by name, yielding (name, tensor) pairs with memory cleanup.
    理論描述: 逐一讀取指定名稱的張量並回傳，每次讀取後主動釋放記憶體以避免累積。
    """

def tensor_shape(self, name: str) -> list[int]:
    """
    Summary: Return the shape of a tensor by name.
    理論描述: 查詢指定張量的維度形狀資訊。
    """

def architecture(self) -> dict:
    """
    Summary: Return a copy of the model architecture metadata.
    理論描述: 回傳模型架構資訊的副本，包含隱藏層大小、層數、注意力頭數等。
    """

def model_type(self) -> str:
    """
    Summary: Return the model type string (e.g., "Qwen2").
    理論描述: 回傳模型類型識別字串。
    """

def tensor_names(self) -> list[str]:
    """
    Summary: Return a list of all tensor names stored in the file.
    理論描述: 回傳檔案中所有已儲存張量的名稱列表。
    """

def close(self):
    """
    Summary: Close the memory map and file handle, releasing all resources.
    理論描述: 關閉記憶體映射與檔案控制代碼，釋放所有相關資源。
    """
```

---

## 3. Tensor Reconstruction | 張量重建

### 3.1 CPU Reconstruction | CPU 重建

```
# ------------------------------------------------------------------
#  CPU reconstruction (numpy)
# ------------------------------------------------------------------

def _read_tensor_cpu(self, numel: int, outlier_count: int, base_offset: int, sub: dict, shape: list):
    """
    Summary: Reconstruct a tensor on CPU using NumPy, unmixing outliers and quantized normal values.
    理論描述: 在 CPU 上重建張量，透過 mask 分離離群值與常規值，並使用量化映射表還原常規數值。
    """
```

**理論描述: 在 CPU 上重建張量，透過 mask 分離離群值與常規值，並使用量化映射表還原常規數值。**

**重建流程:**
```
1. 從 mmap 讀取 outliers (float16)
2. 從 mmap 讀取 mapping (float16)
3. 從 mmap 讀取 normal_indices (uint8)
4. 從 mmap 讀取 mask 位元組
5. 解壓 mask 為 full_mask (bool 陣列)
6. 驗證資料長度
7. 排序 full_mask 分離 normal/outlier 位置
8. 重建結果: result[normal_positions] = mapping[normal_indices]
                     result[outlier_positions] = outliers
9. reshape 為原始形狀
```

### 3.2 GPU Reconstruction | GPU 重建

```
# ------------------------------------------------------------------
#  GPU reconstruction (cupy)
# ------------------------------------------------------------------

def _read_tensor_gpu(self, numel: int, outlier_count: int, base_offset: int, sub: dict, shape: list, as_numpy: bool):
    """
    Summary: Reconstruct a tensor on GPU using CuPy, enabling faster reconstruction for large tensors.
    理論描述: 在 GPU 上使用 CuPy 重建張量，透過記憶體映射直接讀取資料並利用 GPU 並行運算加速還原。
    """
```

**理論描述: 在 GPU 上使用 CuPy 重建張量，透過記憶體映射直接讀取資料並利用 GPU 並行運算加速還原。**

**重建流程:**
```
1. 從 mmap 讀取 outliers (GPU float16)
2. 從 mmap 讀取 mapping (GPU float16)
3. 從 mmap 讀取 normal_indices (GPU uint8)
4. 從 mmap 讀取 mask 位元組
5. 解壓 mask 為 full_mask (GPU bool 陣列)
6. 驗證資料長度
7. 排序 full_mask 分離 normal/outlier 位置
8. 重建結果: result[normal_positions] = mapping[normal_indices]
                     result[outlier_positions] = outliers
9. reshape 為原始形狀
10. 若 as_numpy=True，轉換為 numpy 陣列後回傳
```

---

## 4. Context Manager | 上下文管理器

```
def __enter__(self):
    """
    Summary: Context manager entry, returns self for `with` statement usage.
    理論描述: 上下文管理器入口，回傳 self 以支援 `with` 語法。
    """

def __exit__(self, *args):
    """
    Summary: Context manager exit, ensures resources are released on block exit.
    理論描述: 上下文管理器出口，區塊結束時自動呼叫 close 釋放資源。
    """
```

---

## Data Flow Diagram | 資料流示意圖

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           JLLMReader Data Flow                               │
│                           JLLMReader 資料流                                  │
└─────────────────────────────────────────────────────────────────────────────┘

1. INITIALIZATION | 初始化
   ┌─────────────┐    ┌──────────────┐    ┌───────────────┐    ┌────────────┐
   │ Open File   │───▶│ Memory Map   │───▶│ Parse JSON   │───▶│ Extract    │
   │ (開啟檔案)   │    │ (記憶體映射)  │    │ (解析表頭)    │    │ Metadata   │
   └─────────────┘    └──────────────┘    └───────────────┘    │ (提取元數據) │
                                                                    └────────────┘
                                                                            │
                                                                            ▼
                                                               ┌─────────────────────┐
                                                               │ tensors_meta        │
                                                               │ model_type         │
                                                               │ architecture       │
                                                               └─────────────────────┘

2. READ_TENSOR | 讀取張量
   ┌─────────────┐    ┌──────────────┐    ┌─────────────────────┐
   │ read_tensor │───▶│ Query Meta    │───▶│ Compute numel,     │
   │ (讀取張量)   │    │ (查詢元數據)   │    │ outlier_count       │
   └─────────────┘    └──────────────┘    └─────────────────────┘
                                                     │
                                                     ▼
                                            ┌─────────────────────┐
                                            │ device == "cuda"?   │
                                            └─────────────────────┘
                                                 │         │
                                                Yes        No
                                                 │         │
                                                 ▼         ▼
                                    ┌────────────────┐  ┌────────────────┐
                                    │ _read_tensor_   │  │ _read_tensor_  │
                                    │ gpu (CuPy)      │  │ cpu (NumPy)     │
                                    └────────────────┘  └────────────────┘

3. RAW ACCESS | 原始存取
   ┌─────────────────┐    ┌──────────────────────────────────────────┐
   │ read_tensor_raw │───▶│ Return: outliers, mask, indices, mapping │
   │ (原始資料讀取)   │    │ (回傳: 離群值、遮罩、索引、映射表)        │
   └─────────────────┘    └──────────────────────────────────────────┘

4. CLEANUP | 清理
   ┌─────────────┐    ┌─────────────────────────────────────────────┐
   │ close()     │───▶│ Close mmap → Close file → Clear metadata   │
   │ (關閉)       │    │ (關閉 mmap → 關閉檔案 → 清除元數據)          │
   └─────────────┘    └─────────────────────────────────────────────┘
```

---

## JLLM File Format | JLLM 檔案格式

```
┌─────────────────────────────────────────────────────────────────┐
│                     JLLM File Structure                          │
│                     JLLM 檔案結構                                 │
├─────────────────────────────────────────────────────────────────┤
│  Header (2MB)                                                    │
│  表頭 (2MB)                                                      │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ JSON Metadata                                               │ │
│  │ {                                                           │ │
│  │   "model_type": "Qwen2",                                   │ │
│  │   "architecture": { hidden_size, num_layers, ... },        │ │
│  │   "tensors": {                                              │ │
│  │     "tensor_name": {                                        │ │
│  │       "shape": [...],                                        │ │
│  │       "data_offset": ...,                                   │ │
│  │       "sub_offsets": {                                      │ │
│  │         "outliers": [start, end],                           │ │
│  │         "mapping": [start, end],                            │ │
│  │         "normal": [start, end],                             │ │
│  │         "mask": [start, end]                               │ │
│  │       }                                                     │ │
│  │     }                                                       │ │
│  │   }                                                         │ │
│  │ }                                                           │ │
│  └─────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  Tensor Data                                                    │
│  張量資料                                                        │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ [tensor1_data] [tensor2_data] ... [tensorN_data]            │ │
│  │                                                               │ │
│  │ Each tensor data contains:                                   │ │
│  │   - outliers (float16)                                       │ │
│  │   - mapping (float16)                                         │ │
│  │   - normal_indices (uint8)                                   │ │
│  │   - mask (bits, unpacked to bool array)                      │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quantization Theory | 量化理論

```
JLLM 量化格式基於以下原理:

1. 大多數權重值可以用 uint8 映射表表示 (normal values)
2. 少數偏離正常範圍的值直接儲存為 float16 (outliers)
3. mask 陣列標記每個位置是 normal 還是 outlier

重建演算法:
- 根據 mask 排序，分離 normal 和 outlier 位置
- normal 位置: result = mapping[normal_indices]
- outlier 位置: result = outliers
- 合併為完整 float16 陣列

這種格式可以:
- 節省約 50% 儲存空間
- 保持模型精度
- 支援快速重建
```
