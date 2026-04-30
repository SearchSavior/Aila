from __future__ import annotations

import argparse
import sys
import time

import bitsandbytes as bnb
import bitsandbytes.cextension as bnb_cext
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

DEFAULT_MODEL_PATH = "E:/RiderProjects/Aila/models/Qwen3-0.6B"
DEFAULT_PROMPT = "Answer briefly: What is the capital of France?"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")


def build_quant_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
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


def main() -> int:
    configure_stdout()
    args = parse_args()

    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise RuntimeError("torch.xpu is not available in this environment")

    torch.manual_seed(args.seed)
    device = "xpu:0"

    print(f"torch={torch.__version__}")
    print(f"bitsandbytes={bnb.__version__}")
    print(f"bnb_backend={getattr(bnb_cext, 'BNB_BACKEND', 'unknown')}")
    print(f"device={device}")
    print(f"device_name={torch.xpu.get_device_name(0)}")
    print("quantization=nf4, compute_dtype=float16, double_quant=true")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=False)
    quant_config = build_quant_config()

    load_start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=quant_config,
        device_map={"": device},
        dtype=torch.bfloat16,
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    )
    model.eval()
    load_seconds = time.perf_counter() - load_start
    print(f"load_seconds={load_seconds:.2f}")

    messages = [{"role": "user", "content": args.prompt}]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=args.enable_thinking,
    )
    model_inputs = tokenizer([prompt_text], return_tensors="pt").to(device)

    generate_start = time.perf_counter()
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=0.6 if args.enable_thinking else 0.7,
        top_p=0.95 if args.enable_thinking else 0.8,
        top_k=20,
    )
    generate_seconds = time.perf_counter() - generate_start
    thinking, response = split_thinking_and_response(tokenizer, model_inputs.input_ids.shape[1], generated_ids)

    print(f"generate_seconds={generate_seconds:.2f}")
    print(f"prompt={args.prompt}")
    if thinking:
        print("thinking=")
        print(thinking)
    print("response=")
    print(response)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
