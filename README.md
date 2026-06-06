# JLLM-CUDA-InferenceEngine
一個基於 **Python + Pure CUDA (CuPy &amp; PyTorch)** 打造的輕量化、工業級大模型（LLM）動態流式推理引擎。專為消費級顯示卡（如 NVIDIA RTX 3060 12GB）進行極致優化，支援 Llama 3 與 DeepSeek 等主流的 SwiGLU / GQA 架構模型。本專案的核心突破在於**消滅了全量模型預載的顯存高壓點**，透過 **1D 矩陣哈希去重**與**流式動態滑動視窗（Lazy On-Demand Loading）**技術，實現了「計算哪層、加載哪層、用完即丟」的極致記憶體複用，讓 12GB 顯存也能穩健跑通深層大模型推理。
