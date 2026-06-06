# JLLM-CUDA-InferenceEngine 🚀

一個基於 **Python + Pure CUDA (CuPy & PyTorch)** 打造的輕量化、工業級大模型（LLM）動態流式推理引擎。專為消費級顯示卡（如 NVIDIA RTX 3060 12GB）進行極致優化，支援 Llama 3 與 DeepSeek 等主流的 SwiGLU / GQA 架構模型。

本專案的核心突破在於**消滅了全量模型預載的顯存高壓點**，透過 **1D 矩陣哈希去重**與**流式動態滑動視窗（Lazy On-Demand Loading）**技術，實現了「計算哪層、加載哪層、用完即丟」的極致記憶體複用，讓 12GB 顯存也能穩健跑通深層大模型推理。

---

## 🌟 核心硬體級技術亮點 (Technical Highlights)

### 1. 0 複製跨框架記憶體共享 (Zero-Copy DLPack Bridge)
拒絕傳統在 CPU 與 GPU 之間昂貴的 PCIe 頻寬搬運。本引擎利用 `DLPack` 協議，在 PyTorch 的高能幾何流（殘差相加、Attention 前向）與 CuPy 的高效底層算子（哈希、去重、激活函數）之間，實現了 **0 複製的 GPU 顯存內部指針共享**，將框架切換開銷壓縮至 0 毫秒。

### 2. 1D 降維哈希與 GPU 內去重 (CUDA-Native 1D Hash Deduplication)
大模型的特徵矩陣通常極其巨大。本引擎利用 $2^{12}$ (4096) 黃金縮放因子將 FP16 浮點數安全整數化，並在 GPU 內部萬箭齊發地執行一維哈希壓縮算子：
$$\text{Hash}(v_1, v_2) = (v_1 \times 1000003) \oplus v_2$$
配合 `inverse_map_gpu` 反向還原地圖，實現了極速的特徵矩陣去重與 0 毫秒物理膨脹拉伸，徹底杜絕了 `MemoryError: bad allocation` 溢出。

### 3. 流式滑動視窗動態置換 (Sliding Window On-Demand Caching)
告別開機一次性塞滿顯存的自殺式預載。本引擎引入了動態垃圾清道夫與記憶體置換策略：
* **進攻型加載**：在解碼迴圈（Decoding Loop）執行到第 $N$ 層的前一毫秒，Cacher 才動態讀取該層權重。
* **毀滅性清理**：該層接力賽跑完的瞬間，立刻呼叫核心就地正法，強行驅逐記憶體池扣留，**讓 VRAM 始終保持在單層模型的健康水位**！

### 4. 智慧型幾何自適應注意力算子 (Adaptive MHA / GQA Defense)
拋棄死板的硬編碼尺寸。算子能夠在還原投影矩陣的一瞬間，自動根據快取陣列的實際體積進行 **Dynamic Shape Inference**，完美、全自動地相容標準 MHA 或是 Llama 3 官方最核心的 **GQA（Grouped-Query Attention）群組查詢注意力機制**。

---

## 🛠️ 架構拓撲與模組職責 (Architecture)

整個專案由四大核心齒輪天衣無縫地嚙合而成：

* **`JLLMLoader`**：負責從磁碟安全掛載二進位原始權重庫，解析 Header 檔案頭。
* **`JLLMCacher`**：本引擎的核心大腦。負責純 GPU 內的 1D 雜湊壓縮、建立 `inverse_map_gpu` 還原地圖，並實作單層快取的 `cache_single_layer` 與 `clear_single_layer` 動態調度。
* **`JMultiHeadAttention`**：純原生的自適應多頭注意力算子。內部整合了 GQA 廣播拉伸（`cp.repeat`）與智慧型解碼遮罩分流（Decode 階段自動跳過 Causal Mask 計算）。
* **`JLLMInferenceEngine`**：負責掌管整個自迴歸解碼管線（Autoregressive Decoding Pipeline），手動接力完成跨 Transformer 層的 SwiGLU 激活與前向傳播。

---

## ⚡ 性能表現與數值防線

* **數值防漏牆**：MLP 區塊內部的 SwiGLU 門控採用純原生、0 依賴的手動熔煉安全算子，配合 `cp.clip(..., -15.0, 15.0)` 限制極端擾動，徹底絕育了 FP16 指數運算引發的 `NaN` 或 `Inf` 數值血崩。
* **混合精度優化**：在 RMSNorm 算子中，計算平方和均值時強行在顯存內部升級到 `float32` 確保絕不溢出，算完再降回大模型標準的 `float16` 進行廣播相乘，兼顧了硬體速度與幾何精確度。

---

## 🚀 快速開始 (Quick Start)

### 1. 依賴環境
確保你的環境中安裝了支援 CUDA 的 PyTorch 與相應版本的 CuPy：
```bash
pip install torch
pip install cupy-cuda12x  # 根據你的 CUDA 版本選擇（如 11x 或 12x）
pip install transformers