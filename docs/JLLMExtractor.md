# JLLMExtractor 🪓

`JLLMExtractor` 是專案的權重離線預處理工具。它充當「手術刀」的角色，負責將 HuggingFace 官方下載的巨大分散分片（`.safetensors`）解剖，精準過濾並重組為底層引擎專屬的單一連續二進位權重庫（`.jllm`）。

## ⚙️ 核心機制 (Core Mechanisms)

1. **零拷貝過濾 (Zero-Copy Filtering)**：
   利用 `safetensors.safe_open` 算子，直接讀取磁碟張量的數據映射，在不耗費額外 CPU 記憶體的情況下，精準攔截含有 `.weight` 結尾的核心特徵，並剔除不必要的 Bias 偏置。
2. **半精度對齊 (FP16 Alignment)**：
   全自動將原始權重矩陣強制轉換為 `torch.float16` 高速半精度，並以原始二進位數字流（Raw Bytes）順序打入硬碟，確保下游 CUDA/CuPy 載入時的物理連續性。
3. **動態絕對座標映射**：
   在檔案最前端預留固定的 `1MB` 空間，動態寫入由 JSON 格式封裝的導航索引表頭（Index Header），精確記錄每個特徵矩陣在新檔案中的 `offset`（絕對偏移量）與 `size`，供 `JLLMLoader` 進行動態指針尋址。