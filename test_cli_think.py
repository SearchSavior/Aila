"""Quick test: does CLI output contain <think> tags?"""
import json, os, subprocess, sys, tempfile

MODEL = sys.argv[1] if len(sys.argv) > 1 else "./models/qwen3.5-4B-bnb-nf4-offline-visiondense"
BUILD = os.path.abspath("./build")
EXE = os.path.join(BUILD, "Aila.exe")

with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
    json.dump([{"role":"user","content":"Say just hello. /think"}], f)
    tmp = f.name

env = os.environ.copy()
env["PATH"] = BUILD + ";" + env.get("PATH", "")

print(f"Running: {EXE} -m {MODEL} --messages-json {tmp} --max-tokens 512 --greedy --no-stream")
result = subprocess.run(
    [EXE, "-m", MODEL, "--messages-json", tmp, "--max-tokens", "512", "--greedy", "--no-stream", "--log-level", "error"],
    capture_output=True, text=True, timeout=300, env=env
)

stdout = result.stdout
stderr = result.stderr

print(f"\n=== STDOUT ({len(stdout)} chars) ===")
# Show everything after "Engine ready!"
idx = stdout.find("Engine ready!")
if idx >= 0:
    text = stdout[idx:]
else:
    text = stdout
print(text[:2000])
if len(text) > 2000:
    print(f"... (truncated, {len(text)} total)")

print(f"\n=== STDERR ({len(stderr)} chars, first 500) ===")
print(stderr[:500])

has_think = "<think>" in stdout
has_closethink = "</think>" in stdout
print(f"\n=== Analysis ===")
print(f"<think> in stdout: {has_think}")
print(f"</think> in stdout: {has_closethink}")

os.unlink(tmp)
