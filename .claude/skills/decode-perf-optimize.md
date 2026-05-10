---
name: decode-perf-optimize
description: Systematic SYCL kernel performance optimization for Aila decode. Use when asked to improve decode throughput, close gap to llama.cpp Vulkan, or optimize SYCL/NF4 GEMV kernels.
---

# Decode Performance Optimization Skill

This skill captures the methodology, patterns, and lessons learned during the Qwen3.5-4B BNB NF4 decode optimization on Intel Arc A770.

## Workflow

### 1. Establish Baseline

```
# llama.cpp Vulkan (reference)
./llama.cpp/build-vulkan/bin/Release/llama-bench.exe \
  -m ./models/Qwen3.5-4B-UD-Q4_K_XL.gguf -p 2048 -n 1024 -ngl 99

# Aila (target to optimize)
pwsh bench.ps1 -ModelDir "./models/<model>" \
  -PromptTokens 2048 -GenTokens 1024 -BenchIters 5 -WarmupIters 1
```

### 2. Profile Hotspots

```
AILA_PROFILE_Q35_DECODE=1 AILA_PROFILE_Q35_DECODE_EVERY=4 \
./build/Aila.exe -m "<model>" --bench --bench-pp 2048 --bench-tg 8 \
  --bench-iters 1 --bench-warmup 0
```

The profile breaks decode time into stages:
- `linear_proj` / `linear_delta` / `linear_o` — DeltaNet attention (per DeltaNet layer)
- `full_qkv` / `attn` / `full_o` — GQA attention (per GQA layer)
- `ffn_proj` / `ffn_act` / `down` — FFN (all layers)
- `post_attn` / `post_mlp` — residual + RMS norm (all layers)
- `lm_head` — output projection

### 3. Identify Targets

Rank stages by `ms` descending. Focus on the largest contributors. The profile shows AGGREGATE time across all layers — divide by layer count to get per-layer cost.

### 4. Apply Optimizations

See "Proven Optimizations" below for patterns that worked. Always measure before and after; revert immediately if regression.

### 5. Verify Correctness

```bash
# Smoke test model output quality
echo -e "What is 2+2?\n/quit\n" | ./build/Aila.exe -m "<model>" \
  --max-tokens 32 --greedy --no-stream 2>/dev/null

# Run comprehensive benchmark
pwsh bench.ps1 -ModelDir "<model>" \
  -PromptTokens 2048 -GenTokens 1024 -BenchIters 3 -WarmupIters 1
```

## Proven Optimizations

### Pattern 1: vec8 + FMA for Memory-Bound GEMV Kernels

**File**: `src/ops/Bnb4BitLinear.cpp` — `packed_nf4_gemv_bf16` SG16 path

**Before**: 8 individual `input_ptr[...]` reads + separate multiply-then-add
```cpp
partial += static_cast<float>(input_ptr[input_base + 0]) * q0;
partial += static_cast<float>(input_ptr[input_base + 1]) * q1;
// ... 6 more
```

**After**: One coalesced `vec<bf16,8>` load + `sycl::fma` chain
```cpp
const vec8 in_v = *reinterpret_cast<const vec8*>(input_ptr + ib);
partial = sycl::fma(static_cast<float>(in_v[0]), d0, partial);
partial = sycl::fma(static_cast<float>(in_v[1]), d1, partial);
// ... 6 more
```

**Why it works**: 
- 8 individual 2-byte reads → 1 coalesced 16-byte vector load
- Pre-compute dequantized weights (`d0..d7 = qmap[nybble] * am`) before FMA chain
- `sycl::fma` maps to hardware fused multiply-add (1 instruction vs 2)
- Separating dequant from FMA lets compiler schedule memory and compute independently

**Impact**: +6.8% decode (GEMV ops 8-11% faster)

### Pattern 2: vec8 for Elementwise Reduction Kernels

**File**: `src/ops/NormOps.cpp` — `fused_add_rms_norm` seq_len==1 path

**Before**: Per-element `in_ptr[h] + res_ptr[h]` in a stride loop
**After**: `vec<bf16,8>` loads for input, residual, and weight tensors

**Key detail**: For the 1024-hidden specialised path, reduce work-group size from 256→128 so each item processes exactly one vec8.

**Impact**: +1.4% decode (norm ops ~9% faster)

### Pattern 3: Kernel Fusion Without SLM Barriers

**File**: `src/ops/Bnb4BitLinear.cpp` — `packed_nf4_gemv_gate_up_swiglu`

**Original problem**: The fused gate+up GEMV + SiLU kernel used SLM input caching with a `barrier()`, causing significant slowdown.

**Fix**: Remove SLM entirely. Use the same vec8+FMA direct global memory access pattern as the standalone GEMV. The SiLU (`gate / (1 + exp(-gate))`) is computed once per output row AFTER the subgroup reduction — outside the memory-bound inner loop.

**Key principle**: Never add compute (especially exp/div) to the inner loop of a memory-bound kernel. Compute heavy ops go AFTER the reduction, where they execute once per output row, not once per weight element.

**Wiring**: In `src/models/Qwen35HybridBnb4Backend.cpp` decode FFN path:
```cpp
bool fused_ok = Bnb4BitLinear::try_forward_decode_gate_up_swiglu(
    ctx, layer.gate_up_proj, buf_.normed, buf_.gate, ff_dim_);
```

**Impact**: +1.6% decode (eliminates 32 SiLU kernel launches per token)

### Pattern 4: GEMV FMA Without Unrolling

The 2x unrolled GEMV variant (-0.7ms regression) increased register pressure, causing spills. Keep the single-iteration-per-chunk structure with vec8+FMA — it achieves the best balance of ILP and register usage.

## Regressed Optimizations (Do Not Attempt)

| Optimization | Regression | Root Cause |
|---|---|---|
| SLM input caching in GEMV | +15ms | `item.barrier()` sync across 448 work-groups |
| Blocked weight layout GEMV | +15ms | Transposed layout scatters cache lines; only helps GEMM (batch>1), not GEMV |
| oneDNN cached dequant | +5ms | bf16 weights use 4× VRAM; oneDNN matmul optimised for batch>1 |
| JM bf16 FFN (bypass NF4) | +4ms | bf16 reads 4× more weight data than NF4 packed; bandwidth loss dominates |
| uint64 weight loads | +3ms | Intel SIMD optimised for uint32; uint64 byte extraction adds shift+mask cost |
| SiLU inner-loop fusion | +2ms | exp/div in memory-bound inner loop adds compute latency |
| Linear delta loop fusion | +0.5ms | Recomputing decay in second loop costs more than saved S matrix write |
| 2× GEMV unroll | +0.7ms | Register pressure causes spills to stack |

## Key Architectural Insights

### NF4 Bandwidth Advantage Is Fundamental
NF4 packs 4 bits per weight → 4× less memory traffic than bf16. Any optimization that increases weight read size (JM bf16, oneDNN cache) eliminates this advantage. The GEMV kernels are memory-bandwidth-bound at ~3.3 GB/s effective, so bandwidth savings always beat compute savings.

### Memory-Bound vs Compute-Bound
The GEMV inner loop reads one uint32_t (4 bytes) of packed weight per iteration. At ~3.3 GB/s effective bandwidth, each byte takes ~300ps. The compute (dequant + FMA) takes ~50ps. The inner loop is 85% memory-wait, 15% compute. Adding ANY compute to the inner loop directly increases total time.

### Kernel Launch Overhead
Each `queue::submit()` costs ~100μs on Intel SYCL. With ~200+ submissions per token, overhead is ~20ms. Fusion eliminates submissions — the fused FFN path saves 32 submissions (3.2ms) per token.

### Barriers Are Expensive
`item.barrier(access::fence_space::local_space)` synchronizes all work-items in a group. For large group counts (e.g., 448 groups in gate_up GEMV), total barrier time dominates. Only use barriers when strictly necessary (quant_map load, subgroup reduction).

### Intel SIMD Preferences
- uint32 loads: optimal (native SIMD width)
- uint64 loads: slower (requires decomposition into two 32-bit ops + shift/merge)
- vec<bf16,8>: good (16-byte aligned, single instruction)
- SG16: best for large hidden dims (>768 packed bytes per row)
- SG8: best for small hidden dims

## Commit History

For reference, the commits on the decode optimization path:

```
f94af7d perf: vec8 input loads + sycl::fma in packed_nf4_gemv_bf16 decode path
ea1e1f2 perf: vec8 loads in fused_add_rms_norm decode path
3e2531e perf: rewrite packed_nf4_gemv_gate_up_swiglu with vec8+FMA, no SLM
```

## Quick Verification

After making changes:
```bash
pwsh build.ps1
echo -e "What is 2+2?\n/quit\n" | ./build/Aila.exe -m "<model>" \
  --max-tokens 32 --greedy --no-stream 2>/dev/null
pwsh bench.ps1 -ModelDir "<model>" -PromptTokens 2048 -GenTokens 1024 \
  -BenchIters 3 -WarmupIters 1
```
