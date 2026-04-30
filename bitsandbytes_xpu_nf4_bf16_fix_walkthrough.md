# bitsandbytes XPU NF4 bf16 修复 walkthrough

本文整理这次 `bitsandbytes` 在 Intel Arc A770 上的 XPU / NF4 / `bfloat16` 问题定位与修复过程，目标是回答下面几个问题：

1. 问题最初是如何复现的。
2. 为什么能把问题定位到 XPU 的 `gemv_4bit` fast path。
3. 第一版内核补丁具体改了什么，为什么它**改善了**数值但**没有彻底修好**模型输出。
4. 第二层 Python fallback 为什么能把 correctness 拉回来。
5. 这两层修改分别对性能造成了什么影响。

当前实际落地的代码在 worktree：

- [.claude/worktrees/bitsandbytes-xpu-bf16-gemv/](.claude/worktrees/bitsandbytes-xpu-bf16-gemv/)

---

## 1. 问题现象：`float16` 正常，`bfloat16` 在同一路径上误差巨大

最小复现脚本是 [repro_bnb_xpu_nf4_bf16_gemv.py](repro_bnb_xpu_nf4_bf16_gemv.py)。核心逻辑如下：

```python
weight = torch.randn(OUT_FEATURES, IN_FEATURES, device=DEVICE, dtype=torch.float32)
qweight, qstate = F.quantize_4bit(weight, quant_type=QUANT_TYPE, compress_statistics=True)

x = torch.randn(1, IN_FEATURES, device=DEVICE, dtype=torch.float32)
dequantized_weight = F.dequantize_4bit(qweight, quant_state=qstate, quant_type=QUANT_TYPE).to(torch.float32)
reference = x @ dequantized_weight.t()

for dtype in (torch.float16, torch.bfloat16):
    out = F.gemv_4bit(x.to(dtype), qweight, state=qstate).to(torch.float32)
    mae = (out - reference).abs().mean().item()
    maxe = (out - reference).abs().max().item()
    print(f"{dtype}: gemv_4bit mae={mae:.6f}, max_error={maxe:.6f}")
```

原始问题机器上的典型输出是：

```text
torch.float16: gemv_4bit mae=0.008388, max_error=0.041985
torch.bfloat16: gemv_4bit mae=12.890043, max_error=53.826134
```

这说明问题并不是「4bit 量化本身不准」，因为同一份量化权重上：

- `float16` 路径误差很小
- `bfloat16` 路径误差大了三个数量级以上

也就是说，问题更像是 **XPU 上 bf16 的 `gemv_4bit` fast path 实现有 bug**。

---

## 2. 为什么能定位到 `gemv_4bit` fast path

### 2.1 单向量 decode 会优先走 `F.gemv_4bit(...)`

在 [bitsandbytes/autograd/_functions.py](.claude/worktrees/bitsandbytes-xpu-bf16-gemv/bitsandbytes/autograd/_functions.py) 里，`matmul_4bit(...)` 会在单向量推理场景下直接走 `gemv_4bit`：

```python
if A.numel() == A.shape[-1] and A.requires_grad == False and A.device.type != "hpu":
    if A.shape[-1] % quant_state.blocksize != 0:
        return MatMul4Bit.apply(A, B, out, bias, quant_state)
    else:
        out = F.gemv_4bit(A, B.t(), out, state=quant_state)
        if bias is not None:
            out += bias
        return out
else:
    return MatMul4Bit.apply(A, B, out, bias, quant_state)
```

这意味着 decode 场景最关键的一条路径，恰好就是这次出问题的 `F.gemv_4bit(...)`。

### 2.2 XPU 后端会按 dtype 分发到不同 native kernel

在 [bitsandbytes/backends/xpu/ops.py](.claude/worktrees/bitsandbytes-xpu-bf16-gemv/bitsandbytes/backends/xpu/ops.py) 里，可以看到 `bf16` 会命中独立的 native 实现：

```python
if A.dtype == torch.float16:
    lib.cgemv_4bit_inference_fp16(...)
elif A.dtype == torch.bfloat16:
    lib.cgemv_4bit_inference_bf16(...)
elif A.dtype == torch.float32:
    lib.cgemv_4bit_inference_fp32(...)
```

所以这不是 Python 张量通用逻辑的问题，而是 **bf16 的 XPU 原生 kernel 路径** 本身值得重点怀疑。

---

## 3. 原始内核里最可疑的地方

原始版本的 [bitsandbytes/csrc/xpu_kernels.cpp](bitsandbytes/csrc/xpu_kernels.cpp) 中，`kgemv_4bit_inference` 的关键片段是：

```cpp
unsigned char local_B_4bit[num_values_8bit];
T local_B[num_values_4bit / 4];
T local_A[num_values_4bit / 4];
T local_absmax = T(0.0f);

if (idx < 16) {
    quant_map[idx] = T(datatype[idx]);
}

...

local_B[k * 2] = quant_map[...] * local_absmax;
local_B[k * 2 + 1] = quant_map[...] * local_absmax;

...

local_C += (float)(local_A[k] * local_B[k]);
```

这里有两个明显风险点：

1. `quant_map`、`local_B`、`local_absmax` 都是模板类型 `T`
2. 对于 bf16 路径，`local_A[k] * local_B[k]` 很可能先在 **bf16 精度** 下完成，再把乘积转成 `float` 累加

也就是说：

- 累加器虽然是 `float`
- 但乘法之前的信息已经被 bf16 截断了

这和复现现象是吻合的：

- fp16 仍然可用
- bf16 在 fast path 中提前损失了过多精度

---

## 4. 第一层修复：先把 native kernel 的 bf16 乘加精度拉起来

### 4.1 头文件里为 bf16 单独声明更高精度的 dequant 中间类型

在 worktree 的 [csrc/xpu_kernels.h:24-55](.claude/worktrees/bitsandbytes-xpu-bf16-gemv/csrc/xpu_kernels.h#L24-L55) 中，加入了 `dequant_type`：

```cpp
template <typename T, size_t GROUP_SIZE, size_t NUM_PER_THREAD, size_t SUBG_SIZE, int BITS>
class kgemv_4bit_inference {
  private:
    using dequant_type = std::conditional_t<
        std::is_same_v<T, sycl::ext::oneapi::bfloat16>,
        float,
        T>;

  public:
    ...

    void sycl_ker_local_memory_creation(sycl::handler& cgh) {
        quant_map = sycl::local_accessor<dequant_type>(16, cgh);
    }

  private:
    ...
    sycl::local_accessor<dequant_type> quant_map;
};
```

核心思想：

- 对 `fp16` / `fp32` 维持原样
- 对 `bf16`，让查表值 `quant_map` 直接用 `float`

### 4.2 在 kernel 本体里把反量化中间值也提升到 `float`

在 [csrc/xpu_kernels.cpp:175-266](.claude/worktrees/bitsandbytes-xpu-bf16-gemv/csrc/xpu_kernels.cpp#L175-L266) 中：

```cpp
float local_C = 0.0f;
using DequantT = std::conditional_t<
    std::is_same_v<T, sycl::ext::oneapi::bfloat16>,
    float,
    T>;

unsigned char local_B_4bit[num_values_8bit];
DequantT local_B[num_values_4bit / 4];
T local_A[num_values_4bit / 4];
DequantT local_absmax = DequantT(0.0f);

if (idx < 16) {
    quant_map[idx] = static_cast<DequantT>(datatype[idx]);
}
```

这一步的目的，是避免：

- `datatype[idx]` 先被压成 bf16
- `absmax` 先被压成 bf16
- 反量化结果 `local_B` 再被存成 bf16

### 4.3 乘加时对 bf16 分支显式先转 `float`

同一处的乘加循环变成：

```cpp
#pragma unroll
for (int k = 0; k < num_values_4bit / 4; k++) {
    if constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        local_C += static_cast<float>(local_A[k]) * static_cast<float>(local_B[k]);
    } else {
        local_C += static_cast<float>(local_A[k] * local_B[k]);
    }
}
```

这一步只改 bf16：

- bf16：先提升到 `float` 再乘
- 其他 dtype：保留原逻辑

### 4.4 这一层修复的效果

用最小 repro 重新测，bf16 误差从原来的：

```text
mae = 12.890043
max_error = 53.826134
```

下降到了：

```text
mae = 0.040384
max_error = 0.237503
bf16_over_fp16_mae_ratio = 4.81
```

这说明：

- **内核补丁确实抓到了一个真实问题点**
- 而且这个修复对性能几乎没有影响（后面有数据）

但是这还不是最终答案，因为：

- 某些更大 shape 下，native bf16 GEMV 仍然和 `matmul_4bit` / dense reference 不一致
- 真实模型使用 `bnb_4bit_compute_dtype=torch.bfloat16` 时，输出仍然不正确

换句话说，**内核补丁改善了数值，但没有完全修掉 correctness**。

---

## 5. 第二层修复：对 XPU + bf16 + NF4 增加窄范围 correctness fallback

因为 native bf16 fast path 仍然残留 correctness 问题，最终在 Python 层做了一层非常窄的 fallback，只覆盖：

- device = `xpu`
- dtype = `torch.bfloat16`
- quant type = `nf4`

实现位于 [bitsandbytes/functional.py:1269-1306](.claude/worktrees/bitsandbytes-xpu-bf16-gemv/bitsandbytes/functional.py#L1269-L1306)：

```python
if A.device.type == "xpu" and A.dtype == torch.bfloat16 and state.quant_type == "nf4":
    dequantized_weight = dequantize_4bit(B, quant_state=state).to(torch.float32)
    if B.shape[0] == 1:
        result = A.to(torch.float32) @ dequantized_weight
    else:
        result = A.to(torch.float32) @ dequantized_weight.t()
    result = result.to(torch.bfloat16)
    if out is not None:
        out.copy_(result)
        return out
    return result
```

### 5.1 为什么这里要判断 `B.shape[0] == 1`

这个分支是这次 fallback 里一个容易写错的点。

`gemv_4bit(...)` 会在两种不同朝向下被调用：

1. 直接传 `qweight`
2. 从 `matmul_4bit(...)` 过来时传 `qB.t()`

而 `dequantize_4bit(B, quant_state=state)` 返回的矩阵朝向会跟 `B` 的 packed layout 有关，所以不能简单写死成：

```python
A @ dequantized_weight.t()
```

实测后得到的规则是：

- 如果 `B.shape[0] == 1`，就直接 `A @ dequantized_weight`
- 否则用 `A @ dequantized_weight.t()`

这一步保证了 fallback 能同时兼容：

- 直接的最小复现脚本
- `matmul_4bit(...)` 触发的真实 decode fast path

### 5.2 为什么 fallback 能修 correctness

这段 fallback 的本质不是“继续挽救 native fast path”，而是直接绕开它：

1. 先用 `dequantize_4bit(...)` 还原权重
2. 把输入和权重都提升到 `float32`
3. 用普通 matmul 得到结果
4. 最后再 cast 回 `bfloat16`

它牺牲的是 fast path 性能，换来的是 **稳定 correctness**。

---

## 6. 回归测试是怎么补的

为了把这次 bug 固化下来，增加了一个专门的 XPU / NF4 / bf16 回归测试：

文件位置：

- [tests/test_functional.py:901-927](.claude/worktrees/bitsandbytes-xpu-bf16-gemv/tests/test_functional.py#L901-L927)

代码如下：

```python
def test_gemv_4bit_xpu_nf4_bf16_regression(self):
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        pytest.skip("XPU is required")

    torch.manual_seed(1234)
    out_features = 2048
    in_features = 512
    weight = torch.randn(out_features, in_features, device="xpu", dtype=torch.float32)
    qweight, qstate = F.quantize_4bit(weight, quant_type="nf4", compress_statistics=True)

    x = torch.randn(1, in_features, device="xpu", dtype=torch.float32)
    dequantized_weight = F.dequantize_4bit(qweight, quant_state=qstate, quant_type="nf4").to(torch.float32)
    reference = x @ dequantized_weight.t()

    fp16_out = F.gemv_4bit(x.to(torch.float16), qweight, state=qstate).to(torch.float32)
    bf16_out = F.gemv_4bit(x.to(torch.bfloat16), qweight, state=qstate).to(torch.float32)

    fp16_mae = (fp16_out - reference).abs().mean().item()
    fp16_maxe = (fp16_out - reference).abs().max().item()
    bf16_mae = (bf16_out - reference).abs().mean().item()
    bf16_maxe = (bf16_out - reference).abs().max().item()

    assert fp16_mae < 0.1
    assert fp16_maxe < 0.5
    assert bf16_mae < 0.5
    assert bf16_maxe < 2.0
    assert bf16_mae / max(fp16_mae, 1e-6) < 20.0
```

这个测试的重点不是要求 bf16 和 fp16 完全一样，而是防止它重新退化回“比 fp16 差几个数量级”的状态。

---

## 7. 验证结果

### 7.1 最小 repro

当前最终版本下，最小 repro 的结果是：

```text
torch.float16: gemv_4bit mae=0.008388, max_error=0.041985
torch.bfloat16: gemv_4bit mae=0.040384, max_error=0.237503
bf16_over_fp16_mae_ratio=4.81
```

对比原始问题：

```text
bf16 mae=12.890043
```

已经从不可用状态恢复到了可接受范围。

### 7.2 bitsandbytes 相关测试子集

命令：

```bash
BNB_TEST_DEVICE=xpu pytest tests/test_functional.py -k "test_gemv_4bit and bf16 and nf4" -v --tb=short
```

结果：

```text
33 passed, 388 deselected
```

### 7.3 真实模型验证

在本地 worktree 版本上，用：

- `bnb_4bit_quant_type="nf4"`
- `bnb_4bit_compute_dtype=torch.bfloat16`
- `device_map={"": "xpu:0"}`

加载 `Qwen3-0.6B` 后，输出恢复正常：

```text
The capital of France is Paris.
```

而原始 baseline 版本在同样配置下会输出错误文本。

---

## 8. 性能影响：内核补丁几乎无损，fallback 明显更慢

这里要把两件事分开看：

1. **kernel-only native patch**
2. **最终 functional fallback**

### 8.1 GEMV 微基准

测试口径：

- XPU
- NF4 + bf16
- `compress_statistics=True`
- 4 个代表性 shape
- warmup 60，timed 240，repeat 5

| case | baseline fast path | kernel-only native | final fallback | final / baseline |
|---|---:|---:|---:|---:|
| fc1_512 | 0.279 ms | 0.277 ms | 0.527 ms | 1.89x |
| fc2_512 | 0.277 ms | 0.272 ms | 0.523 ms | 1.89x |
| attn_1024 | 0.268 ms | 0.270 ms | 0.520 ms | 1.94x |
| qkv_1024 | 0.270 ms | 0.268 ms | 0.521 ms | 1.93x |

结论：

- **内核补丁本身几乎不伤 fast path 性能**
- **当前 slowdown 主要来自 Python fallback**
- fallback 大约让孤立 GEMV 变慢 **1.9x**

### 8.2 模型级 16-token decode 对比

测试口径：

- `Qwen3-0.6B`
- `bnb_4bit_compute_dtype=torch.bfloat16`
- warmup 4 token，timed 16 token，repeat 3

| variant | generate mean | tok/s | output |
|---|---:|---:|---|
| baseline-functional | 3.295 s | 4.86 | 错误输出 |
| worktree-functional | 2.139 s | 7.48 | `The capital of France is Paris.` |

这里的模型级结果有一个重要说明：

- baseline 虽然是 fast path，但因为 correctness 本身已经出问题，端到端结果并不一定占优
- 因此**更可信的性能结论**还是孤立 GEMV 微基准：fallback 的确更慢

---

## 9. 这次修复的整体结论

这次修复实际上分成了两层：

### 第 1 层：native kernel 精度修复

优点：

- 改动小
- 性能几乎不受影响
- 能显著降低 bf16 数值误差

不足：

- 还没有彻底恢复真实模型 correctness

### 第 2 层：XPU + bf16 + NF4 的窄范围 correctness fallback

优点：

- 能稳定修复最终输出
- 能让测试和真实模型都恢复正常

不足：

- 作为 fast path 的替代，微基准上大约慢 1.9x

所以当前版本更准确地说是：

> **先用一层 native 精度改动改善问题，再用一个很窄的 Python fallback 兜底 correctness。**

如果后面要继续 upstream 或继续优化，最合理的下一步是：

1. 保留这次内核中的 bf16 精度提升思路
2. 继续深挖 native XPU GEMV 为什么在更复杂 shape / 模型 decode 下仍然错误
3. 最终争取把 correctness 拉回 native fast path，然后移除这个 Python fallback

---

## 10. 本次实际改动文件

worktree 中改动的文件是：

- [.claude/worktrees/bitsandbytes-xpu-bf16-gemv/csrc/xpu_kernels.h](.claude/worktrees/bitsandbytes-xpu-bf16-gemv/csrc/xpu_kernels.h)
- [.claude/worktrees/bitsandbytes-xpu-bf16-gemv/csrc/xpu_kernels.cpp](.claude/worktrees/bitsandbytes-xpu-bf16-gemv/csrc/xpu_kernels.cpp)
- [.claude/worktrees/bitsandbytes-xpu-bf16-gemv/bitsandbytes/functional.py](.claude/worktrees/bitsandbytes-xpu-bf16-gemv/bitsandbytes/functional.py)
- [.claude/worktrees/bitsandbytes-xpu-bf16-gemv/tests/test_functional.py](.claude/worktrees/bitsandbytes-xpu-bf16-gemv/tests/test_functional.py)

如果要继续写 PR / issue 说明，这份 walkthrough 可以直接作为基础材料拆分使用。
