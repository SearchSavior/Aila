# Qwen3 推理引擎修复总结 (Walkthrough)

我们成功解决了 Qwen3-0.6B 推理引擎输出乱码和空行的问题。主要修复包括架构对齐、权重加载修正以及位置编码格式更正。

## 关键修复 (Key Fixes)

### 1. 实现 QK-Norm (Per-head RMSNorm)
Qwen3 架构要求在 Q 和 K 向量进入 Attention 和 RoPE 之前，对其进行按头 (Per-head) 的 RMSNorm。我们新增了 `ops::head_rms_norm` 并在 `Qwen3Model::forward` 中正确集成。

### 2. 修正 RoPE 格式 (Interleaved to Split-half)
这是解决乱码最关键的一步。之前的 RoPE 实现采用了交错式 (Interleaved) 旋转，而 Qwen3 (以及大多数现代模型) 使用的是分半式 (Split-half) 旋转。
- **之前**: 旋转相邻对 `(2d, 2d+1)`。
- **现在**: 旋转 `(d, d + head_dim/2)` 对。

### 3. 修正 lm_head 权重加载
尽管 `config.json` 中 `tie_word_embeddings` 为 true，但 Qwen3 的 `safetensors` 文件中包含独立的 `lm_head.weight`。我们修正了加载逻辑，优先使用独立权重。

### 4. 绕过 "Thinking" 模式
Qwen3-0.6B-Thinking 模型默认会生成大量思考文本。我们在 ChatML 模板中注入了 `<think>\n</think>\n`，引导模型跳过冗长的内部思考，直接输出或在文本中思考。

## 验证结果 (Verification)

模型现在可以生成连贯的文本。例如，针对 "What is 2+3?" 的提问，模型输出了清晰的思考过程（Interior Monologue）并最终得出正确答案。

> [!TIP]
> 目前模型在生成时会打印 `[TokenID]` 诊断信息。如果需要更简洁的输出，可以随时修改 `Engine.hpp` 移除 `std::cout << "[" << next_token << "]" << std::flush;`。

## 代码链接
- [Ops.cpp (RoPE & QK-Norm)](file:///e:/RiderProjects/Aila/src/ops/Ops.cpp)
- [Qwen3.cpp (Forward Loop)](file:///e:/RiderProjects/Aila/src/models/Qwen3.cpp)
- [Engine.hpp (Bypass & Logging)](file:///e:/RiderProjects/Aila/include/engine/Engine.hpp)
