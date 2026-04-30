# bitsandbytes NF4 inference on Intel Arc A770

## Result

Verified on `Intel(R) Arc(TM) A770 Graphics` that the local `Qwen3-0.6B` model can be loaded with `bitsandbytes` NF4 quantization and can generate correct responses on `xpu:0`.

Successful outputs captured on this machine:

Online NF4 load from the original BF16 checkpoint:

- Prompt: `Answer briefly: What is the capital of France?`
- Output: `The capital of France is Paris.`

- Prompt: `请简短回答：法国的首都是哪里？`
- Output: `法国的首都是巴黎。`

Offline NF4 quantize-save-reload path:

- Quantized checkpoint directory: `E:/RiderProjects/Aila/tmp/qwen3-0.6B-bnb-nf4-offline`
- Reloaded output: `The capital of France is Paris.`
- Reloaded Chinese output: `法国的首都是巴黎。`
- Saved quantized weights size: about `514M`
- Original BF16 weights size: about `1.5G`

## Hardware and system

- OS: Windows 11 Pro `10.0.26200`
- GPU: `Intel(R) Arc(TM) A770 Graphics`
- GPU driver: `32.0.101.8509`

## Python environment

- Virtual environment path: `E:/RiderProjects/Aila/.venv-bnb-xpu`
- Python: `3.13.11`

## Exact dependency versions

Core packages:

```text
torch==2.11.0+xpu
bitsandbytes==0.49.2
transformers==5.6.2
accelerate==1.12.0
safetensors==0.7.0
tokenizers==0.22.2
huggingface-hub==1.12.0
numpy==2.4.3
```

XPU/runtime packages pulled by the PyTorch XPU wheel:

```text
triton-xpu==3.7.0
intel-sycl-rt==2025.3.2
intel-opencl-rt==2025.3.2
dpcpp-cpp-rt==2025.3.2
intel-cmplr-lib-rt==2025.3.2
intel-cmplr-lib-ur==2025.3.2
intel-openmp==2025.3.2
intel-pti==0.16.0
mkl==2025.3.1
onemkl-sycl-blas==2025.3.1
onemkl-sycl-dft==2025.3.1
onemkl-sycl-lapack==2025.3.1
onemkl-sycl-rng==2025.3.1
onemkl-sycl-sparse==2025.3.1
tbb==2022.3.1
tcmlib==1.4.1
umf==1.0.3
```

## Repro steps

Create the environment:

```bash
uv venv E:/RiderProjects/Aila/.venv-bnb-xpu --python 3.13
```

Install PyTorch XPU from the PyTorch XPU wheel index:

```bash
uv pip install --python E:/RiderProjects/Aila/.venv-bnb-xpu/Scripts/python.exe \
  --extra-index-url https://download.pytorch.org/whl/xpu \
  "torch==2.11.0+xpu"
```

Install bitsandbytes and Hugging Face dependencies:

```bash
uv pip install --python E:/RiderProjects/Aila/.venv-bnb-xpu/Scripts/python.exe \
  --extra-index-url https://download.pytorch.org/whl/xpu \
  "bitsandbytes==0.49.2" \
  "transformers==5.6.2" \
  "accelerate==1.12.0" \
  "safetensors==0.7.0"
```

## Test script

The working scripts are:

- `E:/RiderProjects/Aila/test_bnb_nf4_xpu.py`
- `E:/RiderProjects/Aila/test_bnb_nf4_xpu_offline.py`
- `E:/RiderProjects/Aila/test_bnb_nf4_xpu_export.py`

Run the verified default online NF4 test:

```bash
E:/RiderProjects/Aila/.venv-bnb-xpu/Scripts/python.exe E:/RiderProjects/Aila/test_bnb_nf4_xpu.py
```

Run the verified Chinese prompt:

```bash
E:/RiderProjects/Aila/.venv-bnb-xpu/Scripts/python.exe E:/RiderProjects/Aila/test_bnb_nf4_xpu.py \
  --prompt "请简短回答：法国的首都是哪里？" \
  --max-new-tokens 20
```

Run the verified offline NF4 quantize-save-reload test:

```bash
E:/RiderProjects/Aila/.venv-bnb-xpu/Scripts/python.exe E:/RiderProjects/Aila/test_bnb_nf4_xpu_offline.py
```

Run the pure offline NF4 export script without reloading:

```bash
E:/RiderProjects/Aila/.venv-bnb-xpu/Scripts/python.exe E:/RiderProjects/Aila/test_bnb_nf4_xpu_export.py
```

Reuse the saved offline checkpoint without requantizing:

```bash
E:/RiderProjects/Aila/.venv-bnb-xpu/Scripts/python.exe E:/RiderProjects/Aila/test_bnb_nf4_xpu_offline.py \
  --reuse-existing \
  --prompt "请简短回答：法国的首都是哪里？" \
  --max-new-tokens 20
```

## Working quantization settings

These settings worked on the Arc A770:

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
```

Model loading arguments used successfully:

```python
AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=quant_config,
    device_map={"": "xpu:0"},
    dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)
```

## Important notes

- On this machine, `bnb_4bit_compute_dtype=torch.float16` was the stable choice for NF4 inference on XPU.
- `bnb_4bit_compute_dtype=torch.bfloat16` and `torch.float32` loaded successfully but produced degraded or incorrect answers during testing.
- The wheel-based setup worked without separately installing the full Intel oneAPI Base Toolkit.
- `save_pretrained()` on the NF4-loaded model produced a reloadable checkpoint with `quantization_config` embedded in `config.json`, so the saved directory can be loaded directly with `AutoModelForCausalLM.from_pretrained(saved_dir, device_map={"": "xpu:0"})`.
- The local `bitsandbytes` source tree was useful for diagnosis only. It was not needed to run the verified inference path.
- If you want to build `bitsandbytes` from local source on Windows, the repository's `.github/scripts/build-xpu-windows.bat` expects Intel oneAPI `setvars.bat` to exist.
- `python -m bitsandbytes` may still print `PyTorch says XPU is not available` even when `torch.xpu.is_available()` is `True` and real model inference works. Prefer validating with a direct `torch.xpu` check and an actual model generation run.
- The default script uses Qwen3 non-thinking mode because it gave the cleanest short verification output. You can add `--enable-thinking` if needed.
