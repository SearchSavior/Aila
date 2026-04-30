# Title

XPU: `gemv_4bit` with NF4 has huge `bfloat16` error compared with `float16` on Intel Arc A770

# System Info

- OS: Windows 11 Pro `10.0.26200`
- GPU: `Intel(R) Arc(TM) A770 Graphics`
- GPU driver: `32.0.101.8509`
- Python: `3.13.11`
- PyTorch: `2.11.0+xpu`
- bitsandbytes: `0.49.2`
- triton-xpu: `3.7.0`
- intel-sycl-rt: `2025.3.2`
- intel-opencl-rt: `2025.3.2`
- dpcpp-cpp-rt: `2025.3.2`
- numpy: `2.4.3`

# Reproduction

This reproducer does not require any model files. It uses random tensors only.

It compares `bitsandbytes.functional.gemv_4bit(...)` against a reference computed from the same quantized weights after `dequantize_4bit(...)`.

```python
import bitsandbytes as bnb
import bitsandbytes.functional as F
import torch

SEED = 1234
OUT_FEATURES = 2048
IN_FEATURES = 512
QUANT_TYPE = "nf4"
DEVICE = "xpu:0"

torch.manual_seed(SEED)
assert hasattr(torch, "xpu") and torch.xpu.is_available()

print(f"torch={torch.__version__}")
print(f"bitsandbytes={bnb.__version__}")
print(f"device={DEVICE}")
print(f"device_name={torch.xpu.get_device_name(0)}")

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

Observed output on my machine:

```text
torch=2.11.0+xpu
bitsandbytes=0.49.2
device=xpu:0
device_name=Intel(R) Arc(TM) A770 Graphics
torch.float16: gemv_4bit mae=0.008388, max_error=0.041985
torch.bfloat16: gemv_4bit mae=12.890043, max_error=53.826134
```

This same issue also shows up at the model level when using `bnb_4bit_compute_dtype=torch.bfloat16` for NF4 inference on XPU. `torch.float16` gives correct outputs in the same environment, while `torch.bfloat16` degrades badly.

For convenience, I also attached the same reproducer as a standalone file:

- `E:/RiderProjects/Aila/repro_bnb_xpu_nf4_bf16_gemv.py`

# Expected behavior

`torch.bfloat16` should have comparable numerical error to `torch.float16` for the same NF4 `gemv_4bit` operation on XPU, or at least remain in the same order of magnitude.

Instead, on this system, `bfloat16` error is more than three orders of magnitude larger than `float16` against the same dequantized reference.

At the model level, I would expect `bnb_4bit_compute_dtype=torch.bfloat16` to remain usable for XPU inference rather than producing severely degraded generation quality.

# Possible root cause

This looks specific to the XPU fast GEMV path rather than generic NF4 quantization/dequantization.

Some evidence that may help triage:

- `bitsandbytes.autograd._functions.matmul_4bit(...)` routes single-vector inference through `F.gemv_4bit(...)`, which is the path hit during decode.
- In `bitsandbytes/csrc/xpu_kernels.cpp`, `kgemv_4bit_inference` first materializes the dequantized lookup values into template type `T` and also loads activations into `T`.
- The inner loop then does `local_C += (float)(local_A[k] * local_B[k]);`, so the multiply appears to happen in `T` before promotion to `float`.
- That means `bfloat16` loses precision before accumulation, which matches the repro: `float16` stays accurate, while `bfloat16` becomes unusable on the same quantized weights.

I also tested a slower reference path (`dequantize_4bit(...)` followed by normal matmul), and that path did not show the same dramatic `bfloat16` failure. That suggests the issue is likely in the XPU `gemv_4bit` kernel itself, not in NF4 serialization or basic dequantization.
