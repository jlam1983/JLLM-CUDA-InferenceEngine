# JLLMCacher 🧠

`JLLMCacher` 是 JLLM 引擎的「核心大腦」，掌管著整個推理管線中最硬核的兩大超能力：**1D 矩陣哈希去重**與**流式滑動視窗（Lazy Loading）動態調度**。

## ⚙️ 核心技術演進 (Technical Deep-Dive)

### 1. 1D 降維雜湊去重
為了榨乾顯存空間，Cacher 在 GPU 顯存內部將連續的 FP16 權重以 `interval=2` 切塊，乘以 $2^{12}$ 黃金縮放因子整數化，並發動純原生的 CUDA 雜湊算子：
$$\text{Hash}(v_1, v_2) = (v_1 \times 1000003) \oplus v_2$$
利用 GPU 處理 1D 長整數向量去重的超高並行優化，在顯存內部一瞬間產出唯一的 `hashed_keys` 身份代碼與 `inverse_map_gpu` 反向地圖。

### 2. GPU 內物理反向膨脹還原 (`get`)
當外部引擎前向傳播索取權重時，Cacher 拒絕執行傳統緩慢的 CPU 字典解析。它利用儲存在顯存內部的 `inverse_map_gpu` 作為硬體指針，在 0 毫秒內將去重後的短矩陣完成物理級的拉伸與平鋪（`cp.ravel`），瞬間還原成 Llama 3 原始規格的權重向量。

### 3. 流式滑動視窗調度 (`cache_single_layer` & `clear_single_layer`)
* **`cache_single_layer(layer_idx)`**：具備智慧型關鍵字模糊匹配防線（自動搜出帶有 `model.` 或 `transformer.` 前綴的真實全名）。在解碼器即將進入該層的前一毫秒，才動態載入並雜湊該層的 7 個核心權重。
* **`clear_single_layer(layer_idx)`**：當該層計算完成、接力棒交出的瞬間，立刻執行物理級清空。強行斬斷字典指針引用，並迫使 CuPy 記憶體池（Memory Pool）交還 VRAM 顯存，**讓 12GB 顯示卡始終維持在只裝有 1 層模型的健康水位**。