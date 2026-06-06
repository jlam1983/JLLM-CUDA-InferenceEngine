# JLLMInferenceEngine 🚀

`JLLMInferenceEngine` 是整個推理專案的指揮官與調度總管。它負責掌管整個自迴歸解碼管線（Autoregressive Decoding Pipeline），並手動接力完成跨 Transformer 層的高速傳播。

## ⚙️ 核心技術演進 (Technical Deep-Dive)

### 1. 0 複製跨框架 DLPack 雙向橋樑
解碼引擎內部完美融合了 PyTorch（掌管外層自迴歸、Attention 幾何流）與 CuPy（掌管 Cacher 雜湊、MLP 高速矩陣相乘）。引擎透過內建的 `from_dlpack` 橋樑方法，實現了兩個龐大框架在 **GPU 顯存內部的 0 複製共享指針通訊**。

### 2. 0 毫秒迴圈內物件開銷
拒絕在自迴歸迴圈內部動態建立類別。引擎在開機點火的一瞬間，就會一次性預建並編譯全模型 32 層的 `JMultiHeadAttention` 算子實例存入清單。進入解碼迴圈後直接指針複用，將 Python 的垃圾回收（GC）延遲降到了絕對的物理零點。

### 3. 數值穩定的純原生 SwiGLU 激活體
在處理大模型最核心的 MLP 區塊時，引擎拋棄了對複雜外置子模組（如 `cupy.scipy`）的玄學依賴。直接利用純原生 cp 算子手動熔煉出數值穩定的 SwiGLU 門控融合公式：
$$\text{SwiGLU}(X) = \big( X \cdot \text{Gate}^T \times \text{Sigmoid}(X \cdot \text{Gate}^T) \big) \cdot \text{Up}^T$$
並引入了 **`cp.clip(..., -15.0, 15.0)`** 構築數值防火牆，徹底絕育了 FP16 指數極端運算引發的 `NaN` 或 `Inf` 信號雪崩。