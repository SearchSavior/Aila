# Phase 11 Qwen3.5 Vision Optimization Handoff

## Summary

以 `2026-04-30`、当前分支 `main`、提交基线 `1a8505d` 为起点，这一轮工作的目标分成两段：

1. 让 **Qwen3.5 quantized vision** 在 Aila 里真正跑通，而不是只支持 text-only quantized path
2. 在此基础上，先稳定并优化 **vision path**，暂时不碰 text-path decode 优化

这轮已经可以比较明确地说：

- 本地 offline BnB NF4 导出链路已经能保留 multimodal 结构
- Aila 现在可以加载“**语言侧 NF4，vision tower 保持 dense**”的 Qwen3.5 multimodal checkpoint
- Qwen3.5 vision encoder 已经迁到 GPU 主路径，token cap 不再是唯一性能手段
- 第一刀 vision 性能优化已经落在 `attention_bidi(...)`，并且在 1024 vision token case 上拿到了可重复的 block-time 下降

## Current State

这份 handoff 写入时，工作区包含以下几类变更，并准备一起提交：

- quantized multimodal export / offline validation 脚本
- bitsandbytes 4-bit safetensors / ModelSpec / backend 支持
- vision embedding override / mRoPE 接口下沉到 `IModelBackend`
- GPU 版 `Qwen35VisionEncoder`
- vision 专用 ops（patchify / position / mRoPE / bidirectional attention 等）
- 本轮新增的 `attention_bidi(...)` tiled streaming softmax 优化

不建议提交的本地杂项：

- `.claude/worktrees/`
- `__pycache__/`

## What Was Implemented

### 1. Quantized multimodal export now preserves vision

主文件：

- `test_bnb_nf4_xpu_export.py`
- `test_bnb_nf4_xpu_offline.py`

关键调整：

- 对 Qwen3.5 multimodal config 不再走 `AutoModelForCausalLM`
- 改为基于 `AutoConfig` 自动选择：
  - text-only -> `AutoModelForCausalLM`
  - multimodal -> `AutoModelForImageTextToText`
- 默认保留 `model.visual` 为 dense 精度
- 通过 `BitsAndBytesConfig.llm_int8_skip_modules=['model.visual']` 复用 Transformers 4-bit quantizer 的 skip 逻辑
- 导出后保留 processor 相关配置文件，避免 multimodal 元数据丢失

结果：

- 本地导出的 Qwen3.5 NF4 artifact 不再退化成 text-only checkpoint
- Aila 可以正确识别 `model_type=qwen3_5`、`vision_config`、`architectures=["Qwen3_5ForConditionalGeneration"]`

### 2. Aila runtime now accepts quantized Qwen3.5 multimodal checkpoints

主文件：

- `include/engine/Engine.hpp`
- `src/utils/ModelSpec.cpp`
- `src/utils/SafeTensors.cpp`
- `src/utils/SafeTensors.hpp`
- `src/models/Qwen35HybridBnb4Backend.cpp`
- `src/ops/Bnb4BitLinear.cpp`
- `src/ops/Bnb4BitLinear.hpp`

关键调整：

- `ModelSpec` / `SafeTensors` 能解析并校验 bitsandbytes NF4 side tensors / quant_state
- quantized path 明确要求：
  - `quant_method = bitsandbytes`
  - `load_in_4bit = true`
  - `bnb_4bit_quant_type = nf4`
  - `bnb_4bit_quant_storage = uint8`
  - `bnb_4bit_compute_dtype = float16`
- Qwen3.5 hybrid quantized backend 不再因为 vision enabled 而整体拒绝加载
- 保持策略：
  - **语言侧 backend 量化**
  - **vision encoder 仍吃 dense vision 权重**

### 3. Vision embedding override / multimodal injection interface was lowered into `IModelBackend`

主文件：

- `src/models/IModelBackend.hpp`
- `src/models/Qwen35HybridTextBackend.hpp`
- `src/models/Qwen35HybridTextBackend.cpp`
- `include/engine/Engine.hpp`

关键调整：

- `supports_vision_embedding_override()`
- `set_embedding_overrides(...)`
- `clear_embedding_overrides()`
- `set_mrope_positions(...)`
- `clear_mrope_positions()`

都已经作为 virtual 接口放到 `IModelBackend` 上。

意义：

- `Engine` 不再依赖 `dynamic_cast<Qwen35HybridTextBackend*>`
- dense text backend 和 quantized backend 都能走同一套 multimodal injection 边界
- 这是 quantized vision 能跑通的关键接口前提

### 4. `Qwen35VisionEncoder` has been moved onto the GPU path

主文件：

- `src/vision/Qwen35VisionEncoder.hpp`
- `src/vision/Qwen35VisionEncoder.cpp`
- `src/ops/Ops.hpp`
- `src/ops/NormOps.cpp`
- `src/ops/ElementwiseOps.cpp`
- `src/ops/VisionOps.cpp`
- `src/ops/AttentionOps.cpp`

当前结构：

- 图片 decode / resize 仍在 CPU
- 从 RGB 上传之后开始，patchify / patch projection / position add / reorder / ViT blocks / merger 全部走 GPU
- vision blocks 已经是标准的：
  - LayerNorm
  - qkv Linear
  - bias add
  - split qkv
  - Q/K vision mRoPE
  - bidirectional attention
  - proj
  - residual
  - LayerNorm
  - fc1
  - GELU-tanh
  - fc2
  - residual
- 当前 host 边界只保留最后一次 merger 输出的 D2H，用于塞回 `VisionEncodeResult`

补齐的新能力包括：

- `layer_norm(...)`
- `gelu_tanh_inplace(...)`
- `bias_add_inplace(...)`
- `vision_patchify_rgb_u8(...)`
- `vision_add_position_embedding(...)`
- `vision_reorder_merge_blocks(...)`
- `vision_mrope_inplace(...)`
- `attention_bidi(...)`

### 5. First targeted vision optimization: `attention_bidi(...)`

主文件：

- `src/ops/AttentionOps.cpp`

改动前：

- `attention_bidi(...)` 是典型三段式：
  1. 写完整 `[num_heads, seq_len, seq_len]` score matrix
  2. 再逐行 softmax
  3. 再读回 score matrix 与 V 相乘
- 这会导致全量 score buffer materialization 和多次全矩阵访存

改动后：

- 保持函数签名不变，但内部改成 **tiled streaming softmax / online merge**
- 不再依赖 `scores_buf` 作为主热路径的中间存储
- 采用 local `q_cache + scores + acc_local + merge_state`
- 对每个 `(head, query_row)` 逐 tile 扫描 key/value，并在线维护精确 softmax merge state

当前实现仍然是 generic tiled 版本，不是 `head_dim == 256` 的进一步展开特化版本。

## Validated Results

### A. Quantized multimodal artifact validated in Aila

已验证的 artifact：

- `models/qwen3.5-0.8B-bnb-nf4-offline-visiondense/`

通过的 smoke：

```powershell
./smoke.ps1 -Preset phase_gate_q35_vision -ModelDir './models/qwen3.5-0.8B-bnb-nf4-offline-visiondense' -ModelLabel 'q35_08_bnb_nf4_visiondense' -CaseNames @('q35_08_ocr_test_smoke')
```

```powershell
./smoke.ps1 -Preset phase_gate_q35_vision -ModelDir './models/qwen3.5-0.8B-bnb-nf4-offline-visiondense' -ModelLabel 'q35_08_bnb_nf4_visiondense' -CaseNames @('q35_08_rooster_big_smoke') -EnvOverrides @{ AILA_Q35_VISION_PROFILE='1'; AILA_Q35_VISION_MAX_TOKENS='1024' }
```

### B. Vision profile before the new attention rewrite

此前同一个大图 case（1024 vision tokens）保留到控制台的 profile：

- `decode=27.82ms`
- `resize=0.70ms`
- `prep=941.42ms`
- `patch=41.67ms`
- `pos=1.04ms`
- `blocks=3266.05ms`
- `merger=81.89ms`
- `total=4380.89ms`

结论非常明确：

- 真正的热点是 `blocks_ms`
- token cap 不是根治方案

### C. Vision profile after rewriting `attention_bidi(...)`

第一次复测：

- `decode=28.53ms`
- `resize=0.96ms`
- `prep=1048.41ms`
- `patch=48.93ms`
- `pos=0.94ms`
- `blocks=2454.93ms`
- `merger=101.67ms`
- `total=3706.40ms`

第二次复测：

- `decode=24.61ms`
- `resize=0.70ms`
- `prep=991.95ms`
- `patch=45.19ms`
- `pos=0.90ms`
- `blocks=2412.22ms`
- `merger=87.88ms`
- `total=3583.58ms`

结论：

- `blocks_ms` 从 `3266.05ms` 降到 `~2.41s - 2.45s`
- 降幅约 `25%`
- `total_ms` 从 `4380.89ms` 降到 `~3.58s - 3.71s`
- 降幅约 `15% - 18%`
- 这说明第一刀方向是对的，attention 确实是大头之一

### D. Regression check after the rewrite

再次跑短 OCR smoke：

```powershell
./smoke.ps1 -Preset phase_gate_q35_vision -ModelDir './models/qwen3.5-0.8B-bnb-nf4-offline-visiondense' -ModelLabel 'q35_08_bnb_nf4_visiondense' -CaseNames @('q35_08_ocr_test_smoke')
```

结果：

- PASS
- 输出仍为 `1 + 1 = 2`

## Known Issues / Not Yet Fixed

### 1. HF-side multimodal NF4 reload still fails during `generate()`

现象：

- 导出的 multimodal NF4 artifact 可以 reload
- 但在 Hugging Face / bitsandbytes 路径调用 `generate()` 时，仍可能在 quantized `lm_head` 或相关 FP4/NF4 state 上触发断言/未初始化问题

状态：

- **Aila runtime 不受这个问题阻塞**
- 这是外部 HF-side 问题，当前不是主线任务

### 2. `.venv-bnb-xpu` 里的 `AutoProcessor` / PIL / torchvision 仍不稳定

现象：

- 环境里 `AutoProcessor.from_pretrained(...)` 的 vision stack 探测并不稳定
- 导出流程已经通过直接保留 processor config 规避了这个问题

状态：

- 仍未在环境层面修复
- 但当前不会阻塞 Aila 验证

### 3. `scores_buf` 仍然在 vision runtime buffers 里分配

现状：

- `Qwen35VisionEncoder` 仍然分配 `scores` buffer
- 新版 `attention_bidi(...)` 不再依赖它作为主热路径

建议：

- 下一轮可以确认没有隐藏调用后，把这块 buffer 从 vision runtime 中删掉
- 但这一轮为了最小化接口扰动，先保留了签名和调用点

## Recommended Next Steps

### 1. Next target: `vision_mrope_inplace(...)`

主文件：

- `src/ops/VisionOps.cpp`

为什么：

- 它在每个 block 里对 Q 和 K 各跑一次
- 当前实现有大量 `pow / cos / sin` 的逐元素开销
- attention 热点被压下去之后，它会更容易浮出来

建议方向：

- 预计算 rotary scale / angle factor，避免每元素 `pow`
- 尽量把 trig 计算改成更便于复用的形式
- 评估能否把 Q/K 的 mRoPE 应用进一步共享结构

### 2. After that: `layer_norm(...)`

主文件：

- `src/ops/NormOps.cpp`

为什么：

- 每个 block 至少两次 layer norm
- 当前 multi-row 路径对输入有多次遍历，访存不够省

建议方向：

- 减少对输入的重复读取
- 检查是否能更好地利用 local/subgroup reduction

### 3. Only after vision path stabilizes, return to text-path optimization

用户要求已经明确：

- 当前优先级是 vision path
- text path 优化后移

所以建议不要在下一轮重新分散到 decode text kernel / oneDNN dispatch 方向，除非 vision profile 已经明显收敛。

## Files To Read First In The Next Conversation

如果新对话直接接手，建议先读：

- `handoff_phase11_q35_vision_optimization.md`
- `src/ops/AttentionOps.cpp`
- `src/ops/VisionOps.cpp`
- `src/ops/NormOps.cpp`
- `src/vision/Qwen35VisionEncoder.cpp`
- `src/vision/Qwen35VisionEncoder.hpp`
- `include/engine/Engine.hpp`
- `test_bnb_nf4_xpu_export.py`
- `test_bnb_nf4_xpu_offline.py`

## Suggested First Action For The Next Conversation

直接从 `vision_mrope_inplace(...)` 开始，不要再重新调查 quantized vision 是否能跑通。

最直接的起手式：

1. 保持当前 `attention_bidi(...)` 不回退
2. 对 `q35_08_rooster_big_smoke` 继续开 `AILA_Q35_VISION_PROFILE=1`
3. 优化 `vision_mrope_inplace(...)`
4. 复测 `blocks_ms`
5. 如果收益不足，再进入 `layer_norm(...)`
