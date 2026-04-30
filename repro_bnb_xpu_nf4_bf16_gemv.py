from __future__ import annotations

import sys

import bitsandbytes as bnb
import bitsandbytes.functional as F
import torch

SEED = 1234
OUT_FEATURES = 2048
IN_FEATURES = 512
QUANT_TYPE = "nf4"
DEVICE = "xpu:0"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise RuntimeError("torch.xpu is not available")

    torch.manual_seed(SEED)

    print(f"torch={torch.__version__}")
    print(f"bitsandbytes={bnb.__version__}")
    print(f"device={DEVICE}")
    print(f"device_name={torch.xpu.get_device_name(0)}")
    print(f"seed={SEED}")
    print(f"shape=({OUT_FEATURES}, {IN_FEATURES})")
    print(f"quant_type={QUANT_TYPE}")

    weight = torch.randn(OUT_FEATURES, IN_FEATURES, device=DEVICE, dtype=torch.float32)
    qweight, qstate = F.quantize_4bit(weight, quant_type=QUANT_TYPE, compress_statistics=True)

    x = torch.randn(1, IN_FEATURES, device=DEVICE, dtype=torch.float32)
    dequantized_weight = F.dequantize_4bit(qweight, quant_state=qstate, quant_type=QUANT_TYPE).to(torch.float32)
    reference = x @ dequantized_weight.t()

    results: dict[str, float] = {}
    for dtype in (torch.float16, torch.bfloat16):
        out = F.gemv_4bit(x.to(dtype), qweight, state=qstate).to(torch.float32)
        mae = (out - reference).abs().mean().item()
        maxe = (out - reference).abs().max().item()
        results[str(dtype)] = mae
        print(f"{dtype}: gemv_4bit mae={mae:.6f}, max_error={maxe:.6f}")

    fp16_mae = results[str(torch.float16)]
    bf16_mae = results[str(torch.bfloat16)]
    print(f"bf16_over_fp16_mae_ratio={bf16_mae / fp16_mae:.2f}")

    if not (fp16_mae < 0.1 and bf16_mae > 1.0):
        raise AssertionError(
            f"Did not reproduce the issue clearly enough. fp16_mae={fp16_mae}, bf16_mae={bf16_mae}"
        )

    print("Issue reproduced: bf16 gemv_4bit error is much larger than fp16 on XPU.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
