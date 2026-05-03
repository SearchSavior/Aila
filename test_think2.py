"""Quick-focused tests for /think behavior."""
import ctypes, json, os, sys
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

def bind(fn, restype, argtypes):
    fn.restype, fn.argtypes = restype, argtypes

bind(dll.aila_engine_create, c_void_p, [])
bind(dll.aila_engine_destroy, None, [c_void_p])
bind(dll.aila_engine_init, c_int, [c_void_p, c_char_p, c_int])
bind(dll.aila_default_gen_config, AilaGenConfig, [])
bind(dll.aila_generate, c_void_p, [c_void_p, c_char_p, c_void_p])
bind(dll.aila_generate_messages, c_void_p, [c_void_p, c_char_p, c_void_p])
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

def gen(engine, prompt, max_tok=64):
    cfg = make_cfg(max_tok, 0)
    out = dll.aila_generate(engine, prompt.encode(), byref(cfg))
    if out:
        return string_at(out).decode("utf-8", errors="replace")
    return f"ERROR: {dll.aila_last_error_code(engine)}"

def gen_msgs(engine, msgs, max_tok=64):
    cfg = make_cfg(max_tok, 0)
    out = dll.aila_generate_messages(engine, json.dumps(msgs).encode(), byref(cfg))
    if out:
        return string_at(out).decode("utf-8", errors="replace")
    return f"ERROR: {dll.aila_last_error_code(engine)}"

def reset(engine):
    dll.aila_engine_reset_context(engine)

MODEL = sys.argv[1] if len(sys.argv) > 1 else "./models/qwen3.5-0.8B-bnb-nf4-offline"
engine = dll.aila_engine_create()
dll.aila_engine_init(engine, MODEL.encode(), 4096)
print(f"Model: {MODEL}\n")

# === Test 1: /think with 256 tokens — does </think> appear? ===
print("="*60)
print("Test 1: /think with 256 tokens (interactive)")
reset(engine)
r = gen(engine, "1+1? /think", 256)
has_open = "<think>" in r
has_close = "</think>" in r
print(f"  has <think>: {has_open}")
print(f"  has </think>: {has_close}")
print(f"  length: {len(r)} chars")
# Show start and end
print(f"  start: {r[:80].replace(chr(10),'\\n')}")
if has_close:
    idx = r.index("</think>")
    print(f"  near </think>: ...{r[max(0,idx-30):idx+38].replace(chr(10),'\\n')}")
else:
    print(f"  end: ...{r[-80:].replace(chr(10),'\\n')}")

# === Test 2: Multi-turn WITHOUT any suffix ===
print("\n" + "="*60)
print("Test 2: Multi-turn, no suffix (baseline)")
reset(engine)
r1 = gen(engine, "My name is Alice.", 32)
print(f"  T1: {r1[:80]}")
r2 = gen(engine, "What is my name?", 32)
print(f"  T2: {r2[:120]}")

# === Test 3: /think on 4B model if available ===
# (skipped for now — requires 4B model)

# === Test 4: messages JSON path with /think, more tokens ===
print("\n" + "="*60)
print("Test 4: messages JSON /think with 256 tokens")
r = gen_msgs(engine, [{"role":"user","content":"1+1? /think"}], 256)
has_open = "<think>" in r
has_close = "</think>" in r
print(f"  has <think>: {has_open}")
print(f"  has </think>: {has_close}")
print(f"  length: {len(r)} chars")
print(f"  start: {r[:80].replace(chr(10),'\\n')}")
print(f"  end: ...{r[-80:].replace(chr(10),'\\n')}")

dll.aila_engine_destroy(engine)
