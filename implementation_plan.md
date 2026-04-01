# Qwen3-0.6B 推理引擎实现计划

基于 oneDNN + SYCL 的 Qwen3-0.6B 推理引擎设计与实现方案。

## 背景

现有代码已实现：
- `SafeTensors.cpp/hpp`: 通过 mmap + simdjson 解析 safetensors 文件头，将权重加载到 GPU VRAM
- `MemoryMappedFile.cpp/hpp`: Windows 平台的 mmap 封装
- `main.cpp`: 创建 SYCL queue 和 oneDNN engine/stream，调用权重加载

需要实现完整的推理管线：**Tokenizer → Embedding → 28层 Transformer → LM Head → 采样 → 解码输出**

## Qwen3-0.6B 架构参数

| 参数 | 值 |
|------|-----|
| hidden_size | 1024 |
| num_attention_heads (Q) | 16 |
| num_key_value_heads (KV) | 8 (GQA, 2:1) |
| head_dim | 128 |
| num_hidden_layers | 28 |
| intermediate_size (FFN) | 3072 |
| vocab_size | 151936 |
| rope_theta | 1000000 |
| rms_norm_eps | 1e-6 |
| hidden_act | SiLU (SwiGLU) |
| torch_dtype | bfloat16 |
| tie_word_embeddings | true |

### 权重名称

```
model.embed_tokens.weight                        [151936, 1024]

每层 i (0~27):
  model.layers.{i}.input_layernorm.weight         [1024]
  model.layers.{i}.self_attn.q_proj.weight        [2048, 1024]
  model.layers.{i}.self_attn.k_proj.weight        [1024, 1024]
  model.layers.{i}.self_attn.v_proj.weight        [1024, 1024]
  model.layers.{i}.self_attn.o_proj.weight        [1024, 2048]
  model.layers.{i}.post_attention_layernorm.weight [1024]
  model.layers.{i}.mlp.gate_proj.weight           [3072, 1024]
  model.layers.{i}.mlp.up_proj.weight             [3072, 1024]
  model.layers.{i}.mlp.down_proj.weight           [1024, 3072]

model.norm.weight                                 [1024]
(lm_head 共享 embed_tokens.weight)
```

## User Review Required

> [!IMPORTANT]
> **计算精度**: 使用 BF16 进行所有计算。Intel Arc GPU 对 BF16 有原生硬件加速。

> [!IMPORTANT]
> **Tokenizer**: 使用 simdjson 从 `tokenizer.json` 加载 BPE 词表和合并规则，纯 C++ 实现。这是复杂度最高的非 GPU 组件。如有偏好使用外部库（如 sentencepiece/tiktoken），请告知。

> [!WARNING]
> **首版范围**: 仅支持单 batch 推理 + greedy/top-k 采样。`src/server.cpp` 暂不实现。不实现 beam search 和 continuous batching。

---

## 计算策略

**oneDNN MatMul** 用于所有大型矩阵乘法（计算密集型）：
- Q/K/V/O 线性投影、FFN gate/up/down 投影、LM Head

**自定义 SYCL kernel** 用于所有逐元素操作（内存密集型）：
- Embedding lookup、RMSNorm、RoPE、Attention GEMV（decode）
- Causal Mask + Softmax、SwiGLU、残差连接、采样

---

## 项目文件结构

```
include/engine/
  Types.hpp          ← 模型配置 + 通用类型
  Engine.hpp         ← 推理引擎主类

src/core/
  Context.hpp/.cpp   ← SYCL queue + oneDNN engine/stream
  Tensor.hpp/.cpp    ← GPU Tensor 封装

src/ops/
  Ops.hpp/.cpp       ← [NEW] SYCL kernel + oneDNN MatMul 封装

src/memory/
  KVCache.hpp/.cpp   ← KV-Cache

src/models/
  Qwen3.hpp/.cpp     ← 模型定义 + forward pass

src/utils/
  Tokenizer.hpp/.cpp ← [NEW] BPE Tokenizer
  SafeTensors.*      ← [MODIFY] 重构为返回 name→Tensor 映射
  MemoryMappedFile.* ← 不变

src/main.cpp         ← [MODIFY] 交互式推理
CMakeLists.txt       ← [MODIFY] 添加新源文件
```

删除 `src/core/Allocator.hpp`（合并到 Context）。

---

## Proposed Changes

### 组件 1: 基础设施

---

#### [MODIFY] [Types.hpp](file:///e:/RiderProjects/Aila/include/engine/Types.hpp)

模型配置结构体 `Qwen3Config`、生成参数 `GenerationConfig`。

---

#### [MODIFY] [Context.hpp](file:///e:/RiderProjects/Aila/src/core/Context.hpp) / [Context.cpp](file:///e:/RiderProjects/Aila/src/core/Context.cpp)

封装 SYCL/oneDNN 运行时：
- `sycl::queue` (in-order)、`dnnl::engine`、`dnnl::stream`
- USM 内存分配/释放/拷贝的便捷方法
- `synchronize()` 同步 GPU 操作

---

#### [MODIFY] [Tensor.hpp](file:///e:/RiderProjects/Aila/src/core/Tensor.hpp) / [Tensor.cpp](file:///e:/RiderProjects/Aila/src/core/Tensor.cpp)

轻量级 GPU Tensor：
- 持有 USM device 指针 + shape + dtype
- 按需创建 `dnnl::memory` 对象（用于传给 oneDNN primitive）
- `allocate()` / `view()` / `from_dnnl()` 静态工厂方法
- `reshape()` 零拷贝视图变换

---

### 组件 2: 权重管理

#### [MODIFY] [SafeTensors.hpp](file:///e:/RiderProjects/Aila/src/utils/SafeTensors.hpp) / [SafeTensors.cpp](file:///e:/RiderProjects/Aila/src/utils/SafeTensors.cpp)

重构要点：
1. 消除全局变量，改为 `ModelWeights` 结构体持有 `name → Tensor` 映射
2. `LoadSafetensors()` 接受 `Context&` 参数，返回 `ModelWeights`
3. `ModelWeights::get(name)` 按名称获取 Tensor，找不到时抛异常
4. mmap 文件的生命周期由 `ModelWeights` 管理

---

### 组件 3: 算子层

#### [NEW] [Ops.hpp](file:///e:/RiderProjects/Aila/src/ops/Ops.hpp) / [Ops.cpp](file:///e:/RiderProjects/Aila/src/ops/Ops.cpp)

##### 3.1 Linear 层 (oneDNN MatMul 封装)

```cpp
class Linear {
    // 权重在 safetensors 中为 [out, in] 行主序
    // 通过 memory::desc strides=[1, out] 实现零拷贝转置
    // matmul: input[seq, in] × weight_T[in, out] → output[seq, out]
    void forward(Context& ctx, Tensor& input, Tensor& output, int seq_len);
};
```

**关键设计**：为 decode 模式 (seq_len=1) 预创建 oneDNN primitive（热路径）；prefill 模式按需创建。

##### 3.2 SYCL Kernel 列表

| Kernel | 功能 | 输入→输出 |
|--------|------|-----------|
| `embedding_lookup` | 嵌入查找 | token_ids[seq] → hidden[seq, 1024] |
| `rms_norm` | RMSNorm | x[seq, 1024] × γ[1024] → out[seq, 1024] |
| `apply_rope` | 旋转位置编码 | Q, K in-place 修改 |
| `attention_decode` | Decode 注意力 GEMV | Q[heads, 1, dim] × KV_cache → out[heads, 1, dim] |
| `causal_mask_softmax` | 因果掩码 + Softmax | scores → probs (in-place) |
| `swiglu` | SiLU(gate) × up | gate, up → out |
| `residual_add` | 残差连接 | a += b (in-place) |
| `argmax` | 贪婪采样 | logits[vocab] → token_id |
| `topk_sample` | Top-k 采样 | logits[vocab] + temperature → token_id |

##### 3.3 RMSNorm SYCL Kernel 设计

```
对于每行 x[1..hidden_size]:
  rms = sqrt(mean(x[i]^2) + eps)
  output[i] = x[i] / rms * weight[i]
```
使用 work-group reduction 计算平方和，每个 work-group 处理一行。

##### 3.4 RoPE SYCL Kernel 设计

```
对于每个 head 的每对 (x[2i], x[2i+1]):
  freq = 1.0 / (theta ^ (2i / head_dim))
  angle = pos * freq
  x[2i]'   = x[2i] * cos(angle) - x[2i+1] * sin(angle)
  x[2i+1]' = x[2i] * sin(angle) + x[2i+1] * cos(angle)
```

##### 3.5 Attention Decode SYCL Kernel 设计

Decode 时 seq_len=1，注意力退化为 GEMV：

```
// Step 1: score 计算
对于每个 head h (0..15):
  kv_head = h / 2  // GQA 映射
  对于每个 cached position t (0..cached_len-1):
    score[h][t] = dot(Q[h], K_cache[kv_head][t]) / sqrt(128)

// Step 2: Softmax (每 head 独立)
scores[h] = softmax(scores[h])

// Step 3: 加权求和
对于每个 head h:
  kv_head = h / 2
  output[h] = sum_t(scores[h][t] * V_cache[kv_head][t])
```

---

### 组件 4: KV Cache

#### [MODIFY] [KVCache.hpp](file:///e:/RiderProjects/Aila/src/memory/KVCache.hpp) / [KVCache.cpp](file:///e:/RiderProjects/Aila/src/memory/KVCache.cpp)

- 预分配 28 层 × 2(K,V) × [num_kv_heads, max_seq, head_dim] 的 BF16 缓冲区
- `append()`: 将新 K/V 写入 cache 的 `current_len` 位置
- `reset()`: 重置 `current_len = 0`（新对话）

**内存估算** (BF16, max_seq=4096)：
- 每层: 2 × 8 × 4096 × 128 × 2B = 16 MB
- 28 层: **448 MB**
- 模型权重: **~1.2 GB**
- 总计: < 2 GB（A770 16GB 完全够用）

---

### 组件 5: Tokenizer

#### [NEW] [Tokenizer.hpp](file:///e:/RiderProjects/Aila/src/utils/Tokenizer.hpp) / [Tokenizer.cpp](file:///e:/RiderProjects/Aila/src/utils/Tokenizer.cpp)

从 `tokenizer.json` 加载完整 BPE tokenizer：

```cpp
class Tokenizer {
public:
    bool load(const std::string& tokenizer_json_path);
    
    // 编码: 文本 → token ID 序列
    std::vector<int> encode(const std::string& text);
    
    // 解码: token ID → 文本
    std::string decode(int token_id);
    std::string decode(const std::vector<int>& token_ids);
    
    // 应用 ChatML 模板
    std::vector<int> apply_chat_template(
        const std::string& system_prompt,
        const std::string& user_message);
    
    bool is_eos(int token_id) const;
};
```

**BPE 编码流程**：
1. UTF-8 文本按 regex 预分词 → 得到 word 列表
2. 每个 word 拆为字符/byte-level token
3. 循环查找优先级最高的相邻 token pair → 合并
4. 直到无可合并 → 查 vocab 得到 ID

**ChatML 模板**（Qwen3 格式）：
```
<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
```

---

### 组件 6: 模型 Forward Pass

#### [MODIFY] [Qwen3.hpp](file:///e:/RiderProjects/Aila/src/models/Qwen3.hpp) / [Qwen3.cpp](file:///e:/RiderProjects/Aila/src/models/Qwen3.cpp)

```cpp
class Qwen3Model {
    Qwen3Config config_;
    
    // 每层的 Linear 和 norm weight
    struct TransformerLayer {
        Linear q_proj, k_proj, v_proj, o_proj;
        Linear gate_proj, up_proj, down_proj;
        Tensor input_ln_weight;    // RMSNorm γ
        Tensor post_attn_ln_weight;
    };
    
    Tensor embed_weight_;              // [vocab, hidden]
    std::vector<TransformerLayer> layers_;
    Tensor final_norm_weight_;         // [hidden]
    // lm_head 与 embed_weight_ 共享 (tie_word_embeddings)
    
    KVCache kv_cache_;
    
    // 预分配的激活缓冲区 (避免推理时反复分配)
    struct Buffers {
        Tensor hidden;       // [max_seq, 1024]
        Tensor residual;     // [max_seq, 1024]
        Tensor q, k, v;      // Q/K/V 投影输出
        Tensor attn_out;     // 注意力输出
        Tensor gate, up;     // FFN 中间结果
        Tensor ffn_out;      // FFN 输出
        Tensor logits;       // [1, vocab_size]
        Tensor attn_scores;  // [num_heads, 1, max_cached]
    } buf_;
    
public:
    void load(Context& ctx, ModelWeights& weights, const Qwen3Config& cfg);
    
    // 单步 forward: 输入 token_ids → 输出 logits
    // start_pos: KV cache 中已有的长度（用于 RoPE）
    Tensor& forward(Context& ctx, const int* token_ids, int seq_len);
    
    void reset();  // 重置 KV cache
};
```

**Forward Pass 流程**（单步）：

```
1. embedding_lookup(token_ids) → hidden

对于每层 i = 0..27:
  2. residual = hidden
  3. rms_norm(hidden, input_ln_weight) → hidden
  4. q_proj(hidden) → Q;  k_proj(hidden) → K;  v_proj(hidden) → V
  5. reshape Q → [seq, num_heads, head_dim]
     reshape K → [seq, num_kv_heads, head_dim]
     reshape V → [seq, num_kv_heads, head_dim]
  6. apply_rope(Q, K, start_pos)
  7. kv_cache.append(layer_i, K, V)
  8. attention_decode(Q, K_cache, V_cache) → attn_out
  9. o_proj(attn_out) → hidden
  10. residual_add(hidden, residual)
  
  11. residual = hidden
  12. rms_norm(hidden, post_attn_ln_weight) → hidden
  13. gate_proj(hidden) → gate;  up_proj(hidden) → up
  14. swiglu(gate, up) → gate
  15. down_proj(gate) → hidden
  16. residual_add(hidden, residual)

17. rms_norm(hidden, final_norm_weight) → hidden
18. lm_head: matmul(hidden, embed_weight^T) → logits
19. return logits
```

---

### 组件 7: 推理引擎

#### [MODIFY] [Engine.hpp](file:///e:/RiderProjects/Aila/include/engine/Engine.hpp)

编排完整推理流程：

```cpp
class InferenceEngine {
    Context ctx_;
    Qwen3Model model_;
    Tokenizer tokenizer_;
    GenerationConfig gen_config_;
    
public:
    bool init(const std::string& model_dir);
    
    // 生成回复（流式输出，每个 token 调用 callback）
    std::string generate(const std::string& prompt,
                         std::function<void(const std::string&)> token_callback = nullptr);
};
```

---

### 组件 8: 入口

#### [MODIFY] [main.cpp](file:///e:/RiderProjects/Aila/src/main.cpp)

交互式推理循环：
```
初始化引擎 → 加载模型 + tokenizer
while (true):
    读取用户输入
    engine.generate(input, [](token) { print(token); })
    打印性能统计 (tokens/s)
```

#### [MODIFY] [CMakeLists.txt](file:///e:/RiderProjects/Aila/CMakeLists.txt)

添加新源文件到 `SOURCES`：
```cmake
set(SOURCES
    src/main.cpp
    src/core/Context.cpp
    src/core/Tensor.cpp
    src/ops/Ops.cpp
    src/memory/KVCache.cpp
    src/models/Qwen3.cpp
    src/utils/Tokenizer.cpp
    src/utils/MemoryMappedFile.cpp
    src/utils/SafeTensors.cpp
)
```
添加 `src/core`、`src/ops`、`src/memory`、`src/models` 到 include 路径。

---

## Open Questions

> [!IMPORTANT]
> 1. **Prefill 优化**: 首版是否需要支持 prefill（一次处理多个 token 的 prompt）？还是简单地逐 token 处理 prompt？Prefill 需要额外的注意力 kernel（全矩阵 matmul 而非 GEMV），但能显著加速长 prompt。**建议首版实现 prefill**。

> [!IMPORTANT]
> 2. **Tokenizer 来源**: `tokenizer.json` (11MB) 包含完整的 HuggingFace tokenizer 配置，也有单独的 `vocab.json` (2.7MB) + `merges.txt` (1.8MB)。两种加载方式都能实现，`tokenizer.json` 更自包含但解析更复杂。建议从 `vocab.json` + `merges.txt` 加载（更简单直接）。

> [!IMPORTANT]  
> 3. **最大序列长度**: KV Cache 预分配大小。默认 4096 tokens，是否需要调整？

---

## Verification Plan

### 自动化测试

1. **编译测试**: `cmake --build build --config RelWithDebInfo` 确保编译通过
2. **权重加载验证**: 打印所有权重的 name/shape/dtype，与已知结构对比
3. **Tokenizer 验证**: 编码 "Hello, world!" 然后解码，验证往返一致性
4. **单层 Forward 验证**: 对第 0 层运行 forward，检查输出形状正确
5. **端到端生成**: 输入 "你好" 并生成回复，确认文本连贯

### 手动验证

- 运行交互式推理，测试多轮对话
- 监控 GPU 内存使用（应 < 2GB）
- 测量生成速度（tokens/s），A770 预期 decode 30-60 tok/s

---

## 实施顺序

| 阶段 | 文件 | 预估复杂度 |
|------|------|-----------|
| 1 | Types.hpp, Context.hpp/.cpp, Tensor.hpp/.cpp | 低 |
| 2 | SafeTensors 重构 | 中 |
| 3 | Ops.hpp/.cpp (SYCL kernels + Linear) | **高** |
| 4 | KVCache.hpp/.cpp | 中 |
| 5 | Tokenizer.hpp/.cpp | **高** |
| 6 | Qwen3.hpp/.cpp (model forward) | **高** |
| 7 | Engine.hpp, main.cpp, CMakeLists.txt | 中 |
| 8 | 编译调试 + 端到端验证 | 高 |
