from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import bitsandbytes as bnb
import bitsandbytes.cextension as bnb_cext
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    BitsAndBytesConfig,
)

DEFAULT_SOURCE_MODEL_PATH = Path("E:/RiderProjects/Aila/models/Qwen3-0.6B")
DEFAULT_QUANTIZED_MODEL_PATH = Path("E:/RiderProjects/Aila/models/qwen3-0.6B-bnb-nf4-offline")
DEFAULT_PROMPT = "Answer briefly: What is the capital of France?"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-model-path", type=Path, default=DEFAULT_SOURCE_MODEL_PATH)
    parser.add_argument("--quantized-model-path", type=Path, default=DEFAULT_QUANTIZED_MODEL_PATH)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument(
        "--keep-dense-module",
        action="append",
        dest="keep_dense_modules",
        default=None,
        help=(
            "Module prefix or regex to keep in original precision during export. Repeatable. "
            "Multimodal Qwen3.5 models keep model.visual dense by default."
        ),
    )
    return parser.parse_args()


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")


def is_multimodal_config(config: AutoConfig) -> bool:
    return getattr(config, "model_type", None) == "qwen3_5" and getattr(config, "vision_config", None) is not None


def unique_preserve_order(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def resolve_keep_dense_modules(config: AutoConfig, requested_modules: list[str] | None) -> list[str]:
    modules = list(requested_modules or [])
    if is_multimodal_config(config):
        modules.append("model.visual")
    return unique_preserve_order(modules)


def build_quant_config(keep_dense_modules: list[str]) -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        llm_int8_skip_modules=keep_dense_modules or None,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def split_thinking_and_response(tokenizer: AutoTokenizer, prompt_length: int, generated_ids: torch.Tensor) -> tuple[str, str]:
    output_ids = generated_ids[0][prompt_length:].tolist()
    try:
        split_index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        split_index = 0
    thinking = tokenizer.decode(output_ids[:split_index], skip_special_tokens=True).strip()
    response = tokenizer.decode(output_ids[split_index:], skip_special_tokens=True).strip()
    return thinking, response


def print_environment(device: str) -> None:
    print(f"torch={torch.__version__}")
    print(f"bitsandbytes={bnb.__version__}")
    print(f"bnb_backend={getattr(bnb_cext, 'BNB_BACKEND', 'unknown')}")
    print(f"device={device}")
    print(f"device_name={torch.xpu.get_device_name(0)}")
    print("quantization=nf4, compute_dtype=float16, double_quant=true")


def prepare_output_directory(output_dir: Path, overwrite: bool, reuse_existing: bool) -> None:
    if output_dir.exists() and reuse_existing:
        return
    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Quantized model directory already exists: {output_dir}. Use --overwrite to rebuild it or --reuse-existing to skip requantization."
        )
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)


def copy_processor_assets(source_dir: Path, output_dir: Path) -> list[str]:
    copied: list[str] = []
    for path in sorted(source_dir.glob("*processor_config.json")):
        shutil.copy2(path, output_dir / path.name)
        copied.append(path.name)
    return copied


def build_chat_messages(prompt: str, multimodal: bool) -> list[dict[str, object]]:
    if multimodal:
        return [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    return [{"role": "user", "content": prompt}]


def quantize_and_save(
    source_model_path: Path,
    output_dir: Path,
    device: str,
    source_config: AutoConfig,
    keep_dense_modules: list[str],
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(source_model_path, trust_remote_code=False)
    quant_config = build_quant_config(keep_dense_modules)
    model_loader = AutoModelForImageTextToText if is_multimodal_config(source_config) else AutoModelForCausalLM

    print(f"source_model_type={getattr(source_config, 'model_type', 'unknown')}")
    print(f"source_has_vision_config={is_multimodal_config(source_config)}")
    print(f"model_loader={model_loader.__name__}")
    print(f"keep_dense_modules={keep_dense_modules}")

    load_start = time.perf_counter()
    model = model_loader.from_pretrained(
        source_model_path,
        quantization_config=quant_config,
        device_map={"": device},
        dtype=torch.bfloat16,
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    )
    model.eval()
    load_seconds = time.perf_counter() - load_start
    print(f"quantize_load_seconds={load_seconds:.2f}")

    save_start = time.perf_counter()
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    copied_processor_assets = copy_processor_assets(source_model_path, output_dir)
    save_seconds = time.perf_counter() - save_start
    print(f"save_seconds={save_seconds:.2f}")
    print(f"copied_processor_assets={copied_processor_assets}")

    config_path = output_dir / "config.json"
    saved_config = json.loads(config_path.read_text(encoding="utf-8"))
    print(f"saved_files={[path.name for path in sorted(output_dir.iterdir())]}")
    print(f"saved_model_type={saved_config.get('model_type')}")
    print(f"saved_has_vision_config={'vision_config' in saved_config}")
    print(f"saved_architectures={saved_config.get('architectures')}")
    print(f"has_quantization_config={'quantization_config' in saved_config}")
    if "quantization_config" in saved_config:
        print(f"saved_quantization_config={json.dumps(saved_config['quantization_config'], ensure_ascii=False)}")


def reload_and_generate(
    quantized_model_path: Path,
    prompt: str,
    max_new_tokens: int,
    enable_thinking: bool,
    device: str,
) -> tuple[str, str]:
    config = AutoConfig.from_pretrained(quantized_model_path, trust_remote_code=False)
    multimodal = is_multimodal_config(config)
    model_loader = AutoModelForImageTextToText if multimodal else AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(quantized_model_path, trust_remote_code=False)

    print(f"reload_model_type={getattr(config, 'model_type', 'unknown')}")
    print(f"reload_has_vision_config={multimodal}")
    print(f"reload_model_loader={model_loader.__name__}")
    print(f"reload_has_processor_assets={any(quantized_model_path.glob('*processor_config.json'))}")

    reload_start = time.perf_counter()
    model = model_loader.from_pretrained(
        quantized_model_path,
        device_map={"": device},
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    )
    model.eval()
    reload_seconds = time.perf_counter() - reload_start
    print(f"reload_seconds={reload_seconds:.2f}")

    messages = build_chat_messages(prompt, multimodal)
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    model_inputs = tokenizer([prompt_text], return_tensors="pt").to(device)

    generate_start = time.perf_counter()
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.6 if enable_thinking else 0.7,
        top_p=0.95 if enable_thinking else 0.8,
        top_k=20,
    )
    generate_seconds = time.perf_counter() - generate_start
    print(f"generate_seconds={generate_seconds:.2f}")

    return split_thinking_and_response(tokenizer, model_inputs.input_ids.shape[1], generated_ids)


def main() -> int:
    configure_stdout()
    args = parse_args()

    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise RuntimeError("torch.xpu is not available in this environment")
    if not args.source_model_path.exists():
        raise FileNotFoundError(f"Source model path does not exist: {args.source_model_path}")

    torch.manual_seed(args.seed)
    device = "xpu:0"
    print_environment(device)
    print(f"source_model_path={args.source_model_path}")
    print(f"quantized_model_path={args.quantized_model_path}")

    source_config = AutoConfig.from_pretrained(args.source_model_path, trust_remote_code=False)
    keep_dense_modules = resolve_keep_dense_modules(source_config, args.keep_dense_modules)

    prepare_output_directory(args.quantized_model_path, args.overwrite, args.reuse_existing)
    if args.reuse_existing and args.quantized_model_path.exists():
        print("reusing_existing_quantized_model=true")
    else:
        print("reusing_existing_quantized_model=false")
        quantize_and_save(
            args.source_model_path,
            args.quantized_model_path,
            device,
            source_config,
            keep_dense_modules,
        )

    thinking, response = reload_and_generate(
        args.quantized_model_path,
        args.prompt,
        args.max_new_tokens,
        args.enable_thinking,
        device,
    )

    print(f"prompt={args.prompt}")
    if thinking:
        print("thinking=")
        print(thinking)
    print("response=")
    print(response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
