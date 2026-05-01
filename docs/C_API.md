# Aila C API

The C API (`aila_api.h`) is the stable public interface for integrating Aila into any language with C FFI support. The shared library is `AilaShared.dll` (Windows).

## Quick Example

```c
#include "aila_api.h"

int main() {
    // Create engine
    AilaEngine* engine = aila_engine_create();
    if (!engine) return 1;

    // Load model
    if (aila_engine_init(engine, "./models/qwen3.5-0.8B-bnb-nf4-offline", 4096) != 0) {
        printf("Init failed: %s\n", aila_last_error_message(engine));
        aila_engine_destroy(engine);
        return 1;
    }

    // Generate (blocking)
    AilaGenConfig cfg = aila_default_gen_config();
    cfg.max_new_tokens = 64;
    cfg.do_sample = 0;  // greedy

    char* response = aila_generate(engine, "Hello!", &cfg);
    if (response) {
        printf("%s\n", response);
        aila_free_string(response);
    }

    // Cleanup
    aila_engine_destroy(engine);
    return 0;
}
```

## API Reference

### Lifecycle

#### `aila_engine_create`

```c
AilaEngine* aila_engine_create(void);
```

Creates a new engine instance. Returns `NULL` on allocation failure. The engine is not yet initialized — call `aila_engine_init` before generation.

#### `aila_engine_init`

```c
int aila_engine_init(AilaEngine* engine, const char* model_dir, int max_seq_len);
```

Loads model weights and tokenizer from `model_dir`. `max_seq_len` sets the maximum context window length (e.g. 4096).

Returns `0` on success, non-zero on failure. On failure, call `aila_last_error_code` / `aila_last_error_message` for diagnostics.

The model directory must contain:
- `config.json` — model architecture configuration
- `tokenizer.json` (or `tokenizer_config.json` + vocab files) — tokenizer data
- `model.safetensors` or sharded `model-*.safetensors` — weights

#### `aila_engine_destroy`

```c
void aila_engine_destroy(AilaEngine* engine);
```

Destroys the engine and frees all resources. Passing `NULL` is safe.

### Generation (Blocking)

#### `aila_generate`

```c
char* aila_generate(AilaEngine* engine, const char* prompt, const AilaGenConfig* config);
```

Generates a response for a single user message. Returns a newly allocated UTF-8 string. The caller must free it with `aila_free_string`. Returns `NULL` on error — check `aila_last_error_code` / `aila_last_error_message`.

`config` may be `NULL` to use defaults.

#### `aila_generate_messages`

```c
char* aila_generate_messages(AilaEngine* engine, const char* messages_json, const AilaGenConfig* config);
```

Generates a response from an OpenAI-style messages JSON array. Same return/free semantics as `aila_generate`.

`messages_json` must be a valid JSON array of message objects:

```json
[
  {"role": "system", "content": "You are helpful."},
  {"role": "user",   "content": "Hello!"}
]
```

Each message has a `role` (`"system"`, `"user"`, or `"assistant"`) and `content` (string or array of content parts). Content parts may include `text`, `image`, and `video` types (see README for details).

### Generation (Streaming)

#### `aila_generate_stream`

```c
int aila_generate_stream(AilaEngine* engine, const char* prompt,
                         const AilaGenConfig* config,
                         AilaTokenCallback callback, void* user_data);
```

Generates with token-level streaming. The callback receives each token as a null-terminated UTF-8 string.

Returns:
- `0` — success
- `1` — aborted by callback (callback returned non-zero)
- `-1` — error

#### `aila_generate_messages_stream`

```c
int aila_generate_messages_stream(AilaEngine* engine, const char* messages_json,
                                  const AilaGenConfig* config,
                                  AilaTokenCallback callback, void* user_data);
```

Streaming version of `aila_generate_messages`. Same return values as `aila_generate_stream`.

#### `AilaTokenCallback`

```c
typedef int (*AilaTokenCallback)(const char* token_text, void* user_data);
```

- `token_text` — UTF-8 token string, valid only during the callback
- Return `0` to continue, non-zero to abort generation

### Memory Management

#### `aila_free_string`

```c
void aila_free_string(char* str);
```

Frees a string returned by `aila_generate` or `aila_generate_messages`. Passing `NULL` is safe.

### Configuration

#### `AilaGenConfig`

```c
typedef struct {
    int   max_new_tokens;       // default: 512
    float temperature;          // default: 0.6
    int   top_k;                // default: 20
    float top_p;                // default: 0.95
    float repetition_penalty;   // default: 1.0
    float presence_penalty;     // default: 0.0
    float frequency_penalty;    // default: 0.0
    int   do_sample;            // 0 = greedy, 1 = sampling
    int   decode_chunk_size;    // default: 12
    int   stream_chunk_size;    // default: 4
} AilaGenConfig;
```

#### `aila_default_gen_config`

```c
AilaGenConfig aila_default_gen_config(void);
```

Returns a config struct with sensible defaults. All fields can be overridden before passing to generation functions.

### Context Management

#### `aila_engine_reset_context`

```c
void aila_engine_reset_context(AilaEngine* engine);
```

Clears conversation history and KV cache. Use to start a fresh conversation without destroying/recreating the engine.

#### `aila_engine_context_length`

```c
int aila_engine_context_length(AilaEngine* engine);
```

Returns the current context length in tokens (prompt + generated). Returns `0` if engine is `NULL`.

### Error Handling

#### `aila_last_error_code`

```c
int aila_last_error_code(AilaEngine* engine);
```

Returns the error code from the last API call on this engine. Returns `AILA_ERR_INVALID_ARGUMENT` if engine is `NULL`.

#### `aila_last_error_message`

```c
const char* aila_last_error_message(AilaEngine* engine);
```

Returns a human-readable error message. The pointer is valid until the next API call on the same engine. Returns an empty string if engine is `NULL`.

### Error Codes

| Code | Constant | Description |
|------|----------|-------------|
| 0 | `AILA_OK` | Success |
| 1 | `AILA_ERR_INVALID_ARGUMENT` | Invalid argument (NULL pointer, etc.) |
| 2 | `AILA_ERR_TEMPLATE` | Chat template rendering failed |
| 3 | `AILA_ERR_JSON_PARSE` | JSON parse error |
| 4 | `AILA_ERR_VISION_NOT_ENABLED` | Vision content in prompt but vision not enabled |
| 5 | `AILA_ERR_CONTEXT_OVERFLOW` | Prompt exceeds context window |
| 6 | `AILA_ERR_RUNTIME` | Generic runtime error |

### Logging

#### `aila_set_log_callback`

```c
void aila_set_log_callback(AilaLogCallback callback, void* user_data);
```

Sets a global log callback. Pass `NULL` for `callback` to restore default (stdout/stderr) logging.

#### `aila_set_log_level`

```c
void aila_set_log_level(int level);
```

Sets the minimum log level. Messages below this level are suppressed.

| Level | Description |
|-------|-------------|
| 0 | Debug |
| 1 | Info |
| 2 | Warning |
| 3 | Error |

#### `AilaLogCallback`

```c
typedef void (*AilaLogCallback)(int level, const char* message, void* user_data);
```

### Version

#### `aila_version`

```c
const char* aila_version(void);
```

Returns the library version string (e.g. `"0.1.0"`). The pointer is static and must not be freed.

## Language Bindings

### Python (ctypes)

```python
import ctypes, json, os

os.add_dll_directory("./build")
lib = ctypes.cdll.LoadLibrary("./build/AilaShared.dll")

# Define AilaGenConfig
class AilaGenConfig(ctypes.Structure):
    _fields_ = [
        ("max_new_tokens", ctypes.c_int),
        ("temperature", ctypes.c_float),
        ("top_k", ctypes.c_int),
        ("top_p", ctypes.c_float),
        ("repetition_penalty", ctypes.c_float),
        ("presence_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("do_sample", ctypes.c_int),
        ("decode_chunk_size", ctypes.c_int),
        ("stream_chunk_size", ctypes.c_int),
    ]

# Bind functions
lib.aila_engine_create.restype = ctypes.c_void_p
lib.aila_engine_init.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
lib.aila_engine_init.restype = ctypes.c_int
lib.aila_generate.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
lib.aila_generate.restype = ctypes.c_void_p
lib.aila_free_string.argtypes = [ctypes.c_void_p]
lib.aila_engine_destroy.argtypes = [ctypes.c_void_p]

# Use
engine = lib.aila_engine_create()
lib.aila_engine_init(engine, b"./models/qwen3.5-0.8B-bnb-nf4-offline", 4096)

cfg = AilaGenConfig()
lib.aila_default_gen_config(ctypes.byref(cfg))  # or set fields manually
cfg.max_new_tokens = 32
cfg.do_sample = 0

result = lib.aila_generate(engine, b"Hello!", ctypes.byref(cfg))
if result:
    print(ctypes.string_at(result).decode())
    lib.aila_free_string(result)

lib.aila_engine_destroy(engine)
```

See `test_api.py` in the repository root for a complete Python test script.

### C# (P/Invoke)

```csharp
using System;
using System.Runtime.InteropServices;

[StructLayout(LayoutKind.Sequential)]
struct AilaGenConfig {
    public int max_new_tokens;
    public float temperature;
    public int top_k;
    public float top_p;
    public float repetition_penalty;
    public float presence_penalty;
    public float frequency_penalty;
    public int do_sample;
    public int decode_chunk_size;
    public int stream_chunk_size;
}

class Aila {
    [DllImport("AilaShared.dll")] static extern IntPtr aila_engine_create();
    [DllImport("AilaShared.dll")] static extern int aila_engine_init(IntPtr e, string dir, int maxSeq);
    [DllImport("AilaShared.dll")] static extern IntPtr aila_generate(IntPtr e, string prompt, ref AilaGenConfig cfg);
    [DllImport("AilaShared.dll")] static extern void aila_free_string(IntPtr s);
    [DllImport("AilaShared.dll")] static extern void aila_engine_destroy(IntPtr e);
    // ... etc
}
```

### Rust (FFI)

```rust
use std::ffi::{c_char, c_int, c_void, CStr, CString};

#[repr(C)]
struct AilaGenConfig { /* ... same fields ... */ }

extern "C" {
    fn aila_engine_create() -> *mut c_void;
    fn aila_engine_init(engine: *mut c_void, model_dir: *const c_char, max_seq: c_int) -> c_int;
    fn aila_generate(engine: *mut c_void, prompt: *const c_char, config: *const AilaGenConfig) -> *mut c_char;
    fn aila_free_string(s: *mut c_char);
    fn aila_engine_destroy(engine: *mut c_void);
}
```

## Thread Safety

AilaEngine instances are **not thread-safe**. Each thread should create its own engine, or external synchronization must be provided. Multiple engines can operate independently in parallel.
