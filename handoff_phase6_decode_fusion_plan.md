# Phase 6 Decode Fusion Plan

## Summary

以 `phase5_linear_cleanup_suite` 为新的稳定基线，而不是继续沿用更早的 `89.63 tok/s` 基线：

- `greedy_main`: `prefill 663.13 tok/s`, `decode 100.88 tok/s`
- `sample_main`: `prefill 663.20 tok/s`, `decode 95.25 tok/s`
- `decode_profile_main` 平均总耗时：`35.08 ms/token`
- 当前前五热点：`ffn_proj 4.881`、`down 4.506`、`linear_delta 4.190`、`linear_proj 3.780`、`linear_o 3.167`

下一轮不再做通用 `Linear.cpp` 试探，也不回到调参矩阵；主方向改成 Qwen3.5 decode 专用融合和旧路径收口。

## Key Changes

### 阶段 1：把当前“已验证正确”的主线真正收口

目标：删掉还留在运行时里的旧路径和旧开关，避免后续优化再次被 fallback 干扰。

实施内容：

- 在 `Qwen35HybridTextBackend` 里移除 runtime host fallback 主线依赖：
  - 删除 `use_device_linear_decode_` 的生产语义
  - 删除 `AILA_Q35_LINEAR_DECODE_GPU`
  - 删除 `run_linear_delta_host(...)` 在正常推理路径中的入口
- 保留 reference 行为时，不再通过运行时 env 切换，而是转为：
  - 调试专用 helper
  - 或单独的离线对拍入口
- 清理 `AILA_Q35_LINEAR_DECODE_WG` 这类只服务旧实验路径的开关
- 在 `perf/presets.json` 中把 attention autotune matrix 迁出默认主流程；默认 benchmark 不再暗示要继续扫 `AILA_ATTN_DECODE_WG` / `AILA_ATTN_JM_TILE`

验收标准：

- `greedy_main` 保持 `>= 660 / 100`
- `sample_main` 保持 `>= 660 / 95`
- `perf_suite` 与 smoke 仍全通过
- 主线不再存在“性能路径 / fallback 路径”双轨并存

### 阶段 2：做 decode 后处理融合波次

目标：先吃掉 `post_attn + post_mlp + ffn_act` 这组固定成本，因为它们合计约 `7.69 ms/token`，已经接近 `linear_delta + linear_o` 总和。

实施内容：

- 重写 `fused_add_rms_norm(seq_len==1)`：
  - 固定针对 `hidden_size=1024`
  - 用 subgroup reduction 或固定 workgroup 规约替代当前通用写法
  - 单次读取 `input + residual`，同时写回 residual 结果和 norm 输出，避免重复访存
  - 不再为 `seq_len==1` 保留与多 token 共用的局部缓存模式
- 重写 `fused_gate_up_swiglu(seq_len==1)`：
  - 固定针对 `ff_dim=3584`
  - 以 `gate_up` 连续布局为前提，按向量块读取 gate/up
  - 只保留 decode 热路径；多 token 继续走现有实现
- 若 `sigmoid_mul` 仍在 decode 层内显著出现，只允许做“固定形状、向量化读写”的小修，不新增新的开关

验收标准：

- `post_attn + post_mlp + ffn_act` 合计下降 `>= 20%`
- `greedy_main` decode 达到 `>= 103 tok/s`
- `sample_main` decode 不低于 `95 tok/s`

### 阶段 3：做 `linear_delta` 专项 kernel 收敛

目标：在不改状态布局的前提下，把 `linear_delta` 从约 `4.19 ms` 压到 `3.6 ms` 左右。

实施内容：

- 以 `run_linear_delta_decode_gpu(..., out_dst)` 为唯一生产入口，内部再拆成：
  - Qwen3.5 固定形状专用 fast path
  - 极小的保底 generic path
- 专用 fast path 固定假设：
  - `linear_q_heads == linear_kv_heads`
  - `linear_head_dim == linear_value_head_dim == 128`
  - 输入来自 `linear_all_proj` 的连续 fused row
- 改动重点：
  - 把 env 控制的 `wg` 选择改成编译期或固定常量分支
  - 降低 barrier 次数，避免“算一小段就全局同步”
  - 对 q/k/v/z 的读取做固定宽度向量化
  - 保持 ring-buffer conv state 和 recurrent state 的当前内存布局不变
  - 继续支持 `seq_len > 1` 的 iterative GPU prefill，但不再额外 split 出中间张量
- 不做的事：
  - 不引入新的状态格式迁移
  - 不回到 host 对拍路径
  - 不新增新的 env 调参入口

验收标准：

- `linear_delta` 平均耗时下降到 `<= 3.6 ms`
- `greedy_main` decode 达到 `>= 105 tok/s`
- `greedy_main` prefill 保持 `>= 655 tok/s`

### 阶段 4：只在最后一轮再评估 attention decode

目标：只有当前三阶段完成后，才判断是否还有必要动 attention decode；默认不把它作为下一轮主攻。

实施内容：

- attention decode 只做“收口式”改动：
  - 固定当前 A770 已验证的 tile 选择
  - 把 `AILA_ATTN_DECODE_WG` / `AILA_ATTN_JM_TILE` 降级为 debug-only
  - 默认主线不再依赖 preset autotune
- 若 `decode_profile_main` 中 `attn` 仍低于 `1.5 ms`，则本阶段不继续展开
- 若前三阶段完成后 decode 仍卡在 `< 105 tok/s`，再开独立后续计划处理 attention，不混入本轮

验收标准：

- 主线 benchmark 无调参依赖
- 默认环境下结果稳定复现 `greedy_main >= 105 tok/s`
- `perf_suite` 结果与默认主线一致，不需要额外 env

## Test Plan

固定按“少测但串行、只在正向后扩测”的顺序执行：

1. 每次结构性修改后只跑：
   `.\build.ps1 -BuildDir build -Config Release`
   `.\bench.ps1 -BuildDir build -Preset phase_gate_q35_text -Phase <phase_name> -CaseNames @('greedy_main')`

2. 只有 `greedy_main` 正向时才补：
   `.\profile_decode.ps1 -BuildDir build -Preset phase_gate_q35_text -Phase <phase_name>`

3. 只有阶段收口时才补：
   `.\bench.ps1 -BuildDir build -Preset phase_gate_q35_text -Phase <phase_name> -CaseNames @('sample_main')`
   `.\perf_suite.ps1 -Preset phase_gate_q35_text -Phase <phase_name>`

阶段验收数字统一使用当前稳定基线对比：

- 基线 benchmark：`phase5_linear_cleanup_suite`
- 基线 profile：`total 35.08 ms/token`
- 目标 benchmark：`greedy_main >= 105 tok/s`, `sample_main >= 95 tok/s`
- 目标 profile：`linear_delta <= 3.6 ms`，`post_attn + post_mlp + ffn_act` 合计下降 `>= 20%`

## Assumptions

- 下一轮优化对象仍以 `Qwen3.5-0.8B` 文本路径为主，`Qwen3` dense 只要求 smoke 非回归
- 可以接受 Qwen3.5 专用 decode kernel，只要默认主线更简单且 smoke 不回退
- 不再继续尝试 `Linear.cpp` 的通用 decode JM 主线化
- 已达到 Phase 5 的性能目标，所以接下来优先做“删旧路径 + 专用融合”，不是重新追 Phase 5 基线
- 默认允许把 reference/compare 路径移出 runtime 主线，而不是长期保留在生产代码中
