#!/usr/bin/env python3
"""Test /think and /no_think behavior across single-turn, multi-turn, and model sizes."""
import ctypes, json, os, sys, time
from ctypes import c_int, c_float, c_char_p, c_void_p, CFUNCTYPE, Structure, byref, cdll, string_at

os.add_dll_directory(os.path.abspath("./build"))
dll = cdll.LoadLibrary("./build/AilaShared.dll")

class AilaGenConfig(Structure):
    _fields_ = [
        ("max_new_tokens", c_int), ("temperature", c_float), ("top_k", c_int),
        ("top_p", c_float), ("repetition_penalty", c_float),
        ("presence_penalty", c_float), ("frequency_penalty", c_float),
        ("do_sample", c_int), ("decode_chunk_size", c_int), ("stream_chunk_size", c_int),
    ]

TokenCB = CFUNCTYPE(c_int, c_char_p, c_void_p)

def bind(fn, restype, argtypes):
    fn.restype, fn.argtypes = restype, argtypes

bind(dll.aila_version, c_char_p, [])
bind(dll.aila_engine_create, c_void_p, [])
bind(dll.aila_engine_destroy, None, [c_void_p])
bind(dll.aila_engine_init, c_int, [c_void_p, c_char_p, c_int])
bind(dll.aila_default_gen_config, AilaGenConfig, [])
bind(dll.aila_generate_messages, c_void_p, [c_void_p, c_char_p, c_void_p])
bind(dll.aila_generate, c_void_p, [c_void_p, c_char_p, c_void_p])
bind(dll.aila_free_string, None, [c_void_p])
bind(dll.aila_engine_reset_context, None, [c_void_p])
bind(dll.aila_last_error_code, c_int, [c_void_p])
bind(dll.aila_last_error_message, c_char_p, [c_void_p])

def make_cfg(max_tokens=64, do_sample=0):
    cfg = AilaGenConfig()
    dll.aila_default_gen_config(byref(cfg))
    cfg.max_new_tokens = max_tokens
    cfg.do_sample = do_sample
    return cfg

def generate(engine, prompt, cfg):
    out = dll.aila_generate(engine, prompt.encode(), byref(cfg))
    if out:
        text = string_at(out).decode("utf-8", errors="replace")
        dll.aila_free_string(out)
        return text
    code = dll.aila_last_error_code(engine)
    msg = string_at(dll.aila_last_error_message(engine)).decode() if dll.aila_last_error_message(engine) else "?"
    return f"<ERROR code={code} msg={msg}>"

def generate_msgs(engine, msgs_json, cfg):
    payload = json.dumps(msgs_json)
    out = dll.aila_generate_messages(engine, payload.encode(), byref(cfg))
    if out:
        text = string_at(out).decode("utf-8", errors="replace")
        dll.aila_free_string(out)
        return text
    code = dll.aila_last_error_code(engine)
    msg = string_at(dll.aila_last_error_message(engine)).decode() if dll.aila_last_error_message(engine) else "?"
    return f"<ERROR code={code} msg={msg}>"

def reset(engine):
    dll.aila_engine_reset_context(engine)

def has_tag(text, tag):
    return tag in text

def report(case_name, text, checks):
    status = "PASS" if all(checks.values()) else "FAIL"
    print(f"\n{'='*60}")
    print(f"[{status}] {case_name}")
    for name, ok in checks.items():
        mark = "OK" if ok else "XX"
        print(f"  {mark} {name}")
    print(f"  Output ({len(text)} chars):")
    # Show first 300 chars, flag <think> tags
    preview = text[:300].replace("\n", "\\n")
    if "<think>" in preview:
        preview = preview.replace("<think>", ">>>THINK>>>").replace("</think>", "<<<THINK<<<")
    print(f"  {preview}{'...(truncated)' if len(text) > 300 else ''}")
    return status == "PASS"


def main():
    MODEL = sys.argv[1] if len(sys.argv) > 1 else "./models/qwen3.5-0.8B-bnb-nf4-offline"
    print(f"Model: {MODEL}")

    engine = dll.aila_engine_create()
    rc = dll.aila_engine_init(engine, MODEL.encode(), 4096)
    if rc != 0:
        print("INIT FAILED")
        return 1

    cfg = make_cfg(64, 0)  # greedy, 64 tokens
    results = []

    # === SINGLE-TURN: interactive path (aila_generate) ===
    print("\n" + "="*60)
    print("SECTION 1: Single-turn via aila_generate (interactive path)")
    print("="*60)

    # 1a: no suffix
    reset(engine)
    r = generate(engine, "What is 1+1? Reply briefly.", cfg)
    results.append(report("1a. no suffix, single-turn", r, {
        "non-empty": len(r) > 0,
        "no raw /no_think": "/no_think" not in r,
        "no raw /think": "/think" not in r,
    }))

    # 1b: /no_think
    reset(engine)
    r = generate(engine, "What is 1+1? Reply briefly. /no_think", cfg)
    results.append(report("1b. /no_think, single-turn", r, {
        "non-empty": len(r) > 0,
        "no raw /no_think": "/no_think" not in r,
        "no <think> block": "<think>" not in r,
    }))

    # 1c: /think
    reset(engine)
    r = generate(engine, "What is 1+1? Reply briefly. /think", cfg)
    results.append(report("1c. /think, single-turn", r, {
        "non-empty": len(r) > 0,
        "no raw /think": "/think" not in r,
        "has <think> in output": "<think>" in r,
        "has </think> in output": "</think>" in r,
    }))

    # === SINGLE-TURN: messages path (aila_generate_messages) ===
    print("\n" + "="*60)
    print("SECTION 2: Single-turn via aila_generate_messages (JSON path)")
    print("="*60)

    # 2a: no suffix
    r = generate_msgs(engine, [{"role":"user","content":"What is 1+1? Reply briefly."}], cfg)
    results.append(report("2a. no suffix, JSON", r, {
        "non-empty": len(r) > 0,
        "no raw /no_think": "/no_think" not in r,
    }))

    # 2b: /no_think
    r = generate_msgs(engine, [{"role":"user","content":"What is 1+1? Reply briefly. /no_think"}], cfg)
    results.append(report("2b. /no_think, JSON", r, {
        "non-empty": len(r) > 0,
        "no raw /no_think": "/no_think" not in r,
        "no <think> block": "<think>" not in r,
    }))

    # 2c: /think
    r = generate_msgs(engine, [{"role":"user","content":"What is 1+1? Reply briefly. /think"}], cfg)
    results.append(report("2c. /think, JSON", r, {
        "non-empty": len(r) > 0,
        "no raw /think": "/think" not in r,
        "has <think> in output": "<think>" in r,
        "has </think> in output": "</think>" in r,
    }))

    # === MULTI-TURN: interactive path ===
    print("\n" + "="*60)
    print("SECTION 3: Multi-turn via aila_generate (interactive path)")
    print("="*60)

    # 3a: Turn 1 no suffix, Turn 2 no suffix
    reset(engine)
    r1 = generate(engine, "My name is Alice.", make_cfg(16, 0))
    r2 = generate(engine, "What is my name? Reply briefly.", cfg)
    results.append(report("3a. multi-turn, neither suffix", r2, {
        "non-empty": len(r2) > 0,
        "mentions Alice": "Alice" in r2 or "alice" in r2.lower(),
    }))

    # 3b: Turn 1 /no_think, Turn 2 /no_think
    reset(engine)
    r1 = generate(engine, "My name is Bob. /no_think", cfg)
    r2 = generate(engine, "What is my name? Reply briefly. /no_think", cfg)
    results.append(report("3b. multi-turn, both /no_think", r2, {
        "non-empty": len(r2) > 0,
        "mentions Bob": "Bob" in r2 or "bob" in r2.lower(),
        "no <think> block in turn 2": "<think>" not in r2,
    }))

    # 3c: Turn 1 /think, Turn 2 no suffix
    reset(engine)
    r1 = generate(engine, "My name is Carol. /think", cfg)
    r2 = generate(engine, "What is my name? Reply briefly.", cfg)
    results.append(report("3c. /think then no suffix", r2, {
        "non-empty": len(r2) > 0,
        "mentions Carol": "Carol" in r2 or "carol" in r2.lower(),
    }))

    # 3d: Turn 1 /think, Turn 2 /no_think (switch modes)
    reset(engine)
    r1 = generate(engine, "My name is Dave. /think", make_cfg(32, 0))
    r2 = generate(engine, "What is my name? Reply briefly. /no_think", cfg)
    results.append(report("3d. /think then /no_think", r2, {
        "non-empty": len(r2) > 0,
        "mentions Dave": "Dave" in r2 or "dave" in r2.lower(),
        "no <think> block in turn 2": "<think>" not in r2,
    }))

    # 3e: Turn 1 no suffix, Turn 2 /think
    reset(engine)
    r1 = generate(engine, "My name is Eve.", make_cfg(16, 0))
    r2 = generate(engine, "Greet me by name. /think", make_cfg(64, 0))
    results.append(report("3e. no suffix then /think", r2, {
        "non-empty": len(r2) > 0,
        "has <think> in turn 2": "<think>" in r2,
        "has </think> in turn 2": "</think>" in r2,
    }))

    # 3f: Four turns mixed
    reset(engine)
    r1 = generate(engine, "I like red. /no_think", make_cfg(16, 0))
    r2 = generate(engine, "I also like blue. /think", make_cfg(32, 0))
    r3 = generate(engine, "What colors did I mention first?", cfg)
    r4 = generate(engine, "And second?", make_cfg(32, 0))
    results.append(report("3f. mixed 4-turn conversation", r4, {
        "non-empty": len(r4) > 0,
        "mentions blue": "blue" in r4.lower(),
    }))

    # === SUMMARY ===
    passed = sum(results)
    total = len(results)
    print(f"\n{'='*60}")
    print(f"SUMMARY: {passed}/{total} passed")
    if passed < total:
        print("FAILING TESTS:")
        names = ["1a","1b","1c","2a","2b","2c","3a","3b","3c","3d","3e","3f"]
        for name, ok in zip(names, results):
            if not ok:
                print(f"  - {name}")
    print("="*60)

    dll.aila_engine_destroy(engine)
    return 0 if passed == total else 1

if __name__ == "__main__":
    raise SystemExit(main())
