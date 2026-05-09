#include "Qwen3Bnb4Backend.hpp"
#include "profile/Profiling.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

using bf16 = sycl::ext::oneapi::bfloat16;

namespace {
int round_up_seq(int v, int granularity) {
    return ((v + granularity - 1) / granularity) * granularity;
}
}

Qwen3Bnb4Backend::~Qwen3Bnb4Backend() = default;

void Qwen3Bnb4Backend::ensure_runtime_buffers(Context& ctx, int seq_len) {
    if (seq_len <= runtime_seq_capacity_) return;

    int H = hidden_size_;
    int new_cap = std::min(max_seq_len_, round_up_seq(seq_len, 64));

    buf_.hidden   = Tensor::allocate(ctx, {(int64_t)new_cap, H});
    buf_.normed   = Tensor::allocate(ctx, {(int64_t)new_cap, H});
    buf_.qkv      = Tensor::allocate(ctx, {(int64_t)new_cap, (int64_t)fused_qkv_dim_});
    buf_.q        = Tensor::allocate(ctx, {(int64_t)new_cap, (int64_t)q_dim_});
    buf_.k        = Tensor::allocate(ctx, {(int64_t)new_cap, (int64_t)kv_dim_});
    buf_.v        = Tensor::allocate(ctx, {(int64_t)new_cap, (int64_t)kv_dim_});
    buf_.attn_out = Tensor::allocate(ctx, {(int64_t)new_cap, (int64_t)q_dim_});
    buf_.gate_up  = Tensor::allocate(ctx, {(int64_t)new_cap, (int64_t)(2 * ff_dim_)});
    buf_.gate     = Tensor::allocate(ctx, {(int64_t)new_cap, (int64_t)ff_dim_});
    buf_.up       = Tensor::allocate(ctx, {(int64_t)new_cap, (int64_t)ff_dim_});

    runtime_seq_capacity_ = new_cap;
    AILA_LOG_INFO("[Qwen3Bnb4] Runtime buffers resized: seq_cap=%d", runtime_seq_capacity_);
}

void Qwen3Bnb4Backend::ensure_prefill_scores(Context& ctx, int seq_len) {
    if (seq_len <= prefill_scores_capacity_) return;

    int new_cap = std::min(max_seq_len_, round_up_seq(seq_len, 64));
    buf_.scores = Tensor::allocate(ctx,
        {(int64_t)cfg_.num_attention_heads, (int64_t)new_cap, (int64_t)new_cap},
        dnnl::memory::data_type::f32);

    prefill_scores_capacity_ = new_cap;
    AILA_LOG_INFO("[Qwen3Bnb4] Prefill score buffer resized: seq_cap=%d", prefill_scores_capacity_);
}

void Qwen3Bnb4Backend::ensure_incr_prefill_scores(Context& ctx, int seq_len, int total_len) {
    if (seq_len <= incr_prefill_seq_cap_ && total_len <= incr_prefill_total_cap_) return;

    int new_seq_cap = std::max(seq_len, incr_prefill_seq_cap_);
    int new_total_cap = std::max(total_len, incr_prefill_total_cap_);
    new_seq_cap = round_up_seq(new_seq_cap, 16);
    new_total_cap = round_up_seq(new_total_cap, 64);

    buf_.incr_scores = Tensor::allocate(ctx,
        {(int64_t)cfg_.num_attention_heads, (int64_t)new_seq_cap, (int64_t)new_total_cap},
        dnnl::memory::data_type::f32);

    incr_prefill_seq_cap_ = new_seq_cap;
    incr_prefill_total_cap_ = new_total_cap;
    AILA_LOG_INFO("[Qwen3Bnb4] Incremental prefill score buffer resized: seq_cap=%d, total_cap=%d",
                  incr_prefill_seq_cap_, incr_prefill_total_cap_);
}

bool Qwen3Bnb4Backend::load(Context& ctx, ModelWeights& weights, const ModelSpec& spec,
                             int max_seq_len, std::string* error_message) {
    if (spec.family != ModelFamily::Qwen3Dense) {
        if (error_message) *error_message = "Qwen3Bnb4Backend: invalid model family";
        return false;
    }
    if (!spec.is_bitsandbytes_4bit()) {
        if (error_message) *error_message = "Qwen3Bnb4Backend: model is not a bitsandbytes 4-bit checkpoint";
        return false;
    }

    cfg_ = spec.qwen3;
    max_seq_len_ = max_seq_len;

    hidden_size_ = cfg_.hidden_size;
    q_dim_ = cfg_.num_attention_heads * cfg_.head_dim;
    kv_dim_ = cfg_.num_key_value_heads * cfg_.head_dim;
    fused_qkv_dim_ = q_dim_ + 2 * kv_dim_;
    ff_dim_ = cfg_.intermediate_size;

    auto transpose_weight = [&](const std::string& name) -> Tensor* {
        Tensor& src = weights.get(name);
        int64_t out_f = src.shape(0);
        int64_t in_f = src.shape(1);
        Tensor dst = Tensor::allocate(ctx, {in_f, out_f}, src.dtype());
        ops::transpose(ctx, src, dst);
        ctx.synchronize();
        weights.replace(name, std::move(dst));
        return &weights.get(name);
    };

    auto erase_weight = [&](const std::string& name) {
        if (weights.has(name)) {
            weights.erase(name);
        }
    };
    auto erase_bnb_weight = [&](const std::string& name) {
        erase_weight(name);
        erase_weight(name + ".quant_state.bitsandbytes__nf4");
        erase_weight(name + ".absmax.bitsandbytes__nf4");
        erase_weight(name + ".nested_absmax.bitsandbytes__nf4");
        erase_weight(name + ".nested_quant_map.bitsandbytes__nf4");
    };
    auto init_quant_linear = [&](const std::string& name, Bnb4BitLinear& linear) -> bool {
        Bnb4BitWeightRef ref;
        std::string local_error;
        if (!LoadBnb4BitWeightRef(ctx, weights, name, ref, &local_error)) {
            if (error_message) *error_message = local_error;
            return false;
        }
        if (!linear.init(ctx, ref, &local_error)) {
            if (error_message) *error_message = local_error;
            return false;
        }
        return true;
    };

    auto fuse_two_cols = [&](Tensor& a, Tensor& b) {
        int64_t rows = a.shape(0);
        int64_t a_cols = a.shape(1);
        int64_t b_cols = b.shape(1);
        int64_t total_cols = a_cols + b_cols;

        Tensor out = Tensor::allocate(ctx, {rows, total_cols}, a.dtype());
        bf16* a_ptr = static_cast<bf16*>(a.data());
        bf16* b_ptr = static_cast<bf16*>(b.data());
        bf16* o_ptr = static_cast<bf16*>(out.data());

        ctx.queue().parallel_for(sycl::range<2>(rows, total_cols),
            [=](sycl::id<2> idx) {
                int r = idx[0];
                int cidx = idx[1];
                int64_t out_idx = static_cast<int64_t>(r) * total_cols + cidx;
                if (cidx < a_cols) {
                    o_ptr[out_idx] = a_ptr[static_cast<int64_t>(r) * a_cols + cidx];
                } else {
                    int bc = cidx - static_cast<int>(a_cols);
                    o_ptr[out_idx] = b_ptr[static_cast<int64_t>(r) * b_cols + bc];
                }
            });
        return out;
    };

    auto fuse_three_cols = [&](Tensor& a, Tensor& b, Tensor& c) {
        int64_t rows = a.shape(0);
        int64_t a_cols = a.shape(1);
        int64_t b_cols = b.shape(1);
        int64_t c_cols = c.shape(1);
        int64_t total_cols = a_cols + b_cols + c_cols;

        Tensor out = Tensor::allocate(ctx, {rows, total_cols}, a.dtype());
        bf16* a_ptr = static_cast<bf16*>(a.data());
        bf16* b_ptr = static_cast<bf16*>(b.data());
        bf16* c_ptr = static_cast<bf16*>(c.data());
        bf16* o_ptr = static_cast<bf16*>(out.data());

        ctx.queue().parallel_for(sycl::range<2>(rows, total_cols),
            [=](sycl::id<2> idx) {
                int r = idx[0];
                int cidx = idx[1];
                int64_t out_idx = static_cast<int64_t>(r) * total_cols + cidx;
                if (cidx < a_cols) {
                    o_ptr[out_idx] = a_ptr[static_cast<int64_t>(r) * a_cols + cidx];
                } else if (cidx < a_cols + b_cols) {
                    int bc = cidx - static_cast<int>(a_cols);
                    o_ptr[out_idx] = b_ptr[static_cast<int64_t>(r) * b_cols + bc];
                } else {
                    int cc = cidx - static_cast<int>(a_cols + b_cols);
                    o_ptr[out_idx] = c_ptr[static_cast<int64_t>(r) * c_cols + cc];
                }
            });
        return out;
    };

    // --- Embedding ---
    embed_weight_ = &weights.get("model.embed_tokens.weight");
    AILA_LOG_INFO("[Qwen3Bnb4] embed_tokens loaded");

    // --- Layers ---
    layers_.clear();
    fused_weights_.clear();
    layers_.resize(cfg_.num_hidden_layers);
    fused_weights_.reserve(static_cast<size_t>(cfg_.num_hidden_layers) * 2 + 1);

    for (int i = 0; i < cfg_.num_hidden_layers; ++i) {
        auto& layer = layers_[i];
        std::string prefix = "model.layers." + std::to_string(i) + ".";
        std::vector<std::string> weights_to_erase;

        // Qwen3 uses direct RMSNorm weights (not plus-one like Qwen3.5).
        layer.input_ln_weight = &weights.get(prefix + "input_layernorm.weight");
        layer.post_attn_ln_weight = &weights.get(prefix + "post_attention_layernorm.weight");

        // Fuse Q/K/V into a single Bnb4BitLinear
        if (!init_quant_linear(prefix + "self_attn.o_proj.weight", layer.o_proj)) return false;

        Bnb4BitLinear q_proj, k_proj, v_proj;
        if (!init_quant_linear(prefix + "self_attn.q_proj.weight", q_proj) ||
            !init_quant_linear(prefix + "self_attn.k_proj.weight", k_proj) ||
            !init_quant_linear(prefix + "self_attn.v_proj.weight", v_proj)) {
            return false;
        }
        std::string fused_qkv_error;
        if (!layer.qkv_proj.init_fused_rows(ctx, q_proj, k_proj, v_proj, &fused_qkv_error)) {
            if (error_message) *error_message = fused_qkv_error;
            return false;
        }
        weights_to_erase.push_back(prefix + "self_attn.q_proj.weight");
        weights_to_erase.push_back(prefix + "self_attn.k_proj.weight");
        weights_to_erase.push_back(prefix + "self_attn.v_proj.weight");

        layer.q_norm_weight = &weights.get(prefix + "self_attn.q_norm.weight");
        layer.k_norm_weight = &weights.get(prefix + "self_attn.k_norm.weight");

        // Fuse gate/up and init down
        Bnb4BitLinear gate_proj, up_proj;
        if (!init_quant_linear(prefix + "mlp.gate_proj.weight", gate_proj) ||
            !init_quant_linear(prefix + "mlp.up_proj.weight", up_proj) ||
            !init_quant_linear(prefix + "mlp.down_proj.weight", layer.down_proj)) {
            return false;
        }
        std::string fused_gate_up_error;
        if (!layer.gate_up_proj.init_fused_rows(ctx, gate_proj, up_proj, &fused_gate_up_error)) {
            if (error_message) *error_message = fused_gate_up_error;
            return false;
        }
        weights_to_erase.push_back(prefix + "mlp.gate_proj.weight");
        weights_to_erase.push_back(prefix + "mlp.up_proj.weight");

        // Erase individual BNB4 tensors after fusion (keep quant_map — fused linears still reference it).
        if (!weights_to_erase.empty()) {
            ctx.synchronize();
            for (const auto& name : weights_to_erase) {
                erase_bnb_weight(name);
            }
        }
    }
    AILA_LOG_INFO("[Qwen3Bnb4] %d transformer layers loaded", cfg_.num_hidden_layers);

    // --- Final norm ---
    final_norm_weight_ = &weights.get("model.norm.weight");

    // --- KV Cache ---
    kv_cache_.init(ctx, cfg_, max_seq_len);

    // --- LM Head (always dense, even in quantized mode) ---
    if (weights.has("lm_head.weight") &&
        !weights.has("lm_head.weight.quant_state.bitsandbytes__nf4") &&
        weights.get("lm_head.weight").dtype() != dnnl::memory::data_type::u8) {
        Tensor* lm_w = transpose_weight("lm_head.weight");
        lm_head_.init(ctx, *lm_w, hidden_size_, cfg_.vocab_size, true);
        AILA_LOG_INFO("[Qwen3Bnb4] lm_head loaded (standalone dense)");
    } else {
        // Tied: transpose copy of embed_weight_
        Tensor& src = *embed_weight_;
        Tensor dst = Tensor::allocate(ctx, {src.shape(1), src.shape(0)}, src.dtype());
        ops::transpose(ctx, src, dst);
        ctx.synchronize();
        weights.put("lm_head.weight_preprocessed", std::move(dst));
        lm_head_.init(ctx, weights.get("lm_head.weight_preprocessed"), hidden_size_, cfg_.vocab_size, true);
        AILA_LOG_INFO("[Qwen3Bnb4] lm_head (tied, preprocessed copy)");
    }

    // --- Runtime buffers (lazy grow) ---
    runtime_seq_capacity_ = 0;
    prefill_scores_capacity_ = 0;
    ensure_runtime_buffers(ctx, 1);

    // --- Persistent buffers ---
    buf_.logits = Tensor::allocate(ctx, {1, (int64_t)cfg_.vocab_size});
    buf_.decode_scores = Tensor::allocate(ctx,
        {(int64_t)cfg_.num_attention_heads, (int64_t)max_seq_len},
        dnnl::memory::data_type::f32);
    buf_.rope_freq = Tensor::allocate(ctx,
        {(int64_t)(cfg_.head_dim / 2)},
        dnnl::memory::data_type::f32);
    {
        std::vector<float> rope_freq_host(static_cast<size_t>(cfg_.head_dim / 2));
        for (int d = 0; d < cfg_.head_dim / 2; ++d) {
            rope_freq_host[static_cast<size_t>(d)] =
                1.0f / std::pow(cfg_.rope_theta, (2.0f * d) / static_cast<float>(cfg_.head_dim));
        }
        ctx.memcpy_h2d(buf_.rope_freq.data(), rope_freq_host.data(),
                       rope_freq_host.size() * sizeof(float));
    }

    AILA_LOG_INFO("[Qwen3Bnb4] Model fully loaded and initialized");
    return true;
}

Tensor& Qwen3Bnb4Backend::forward(Context& ctx, const int* token_ids_device, int seq_len) {
    if (seq_len <= 0) {
        throw std::runtime_error("Qwen3Bnb4Backend::forward: seq_len must be positive");
    }

    int H = hidden_size_;
    int QD = q_dim_;
    int KVD = kv_dim_;
    int FF = ff_dim_;
    int start_pos = kv_cache_.current_length();
    int cached_len = start_pos + seq_len;
    if (cached_len > max_seq_len_) {
        throw std::runtime_error("Qwen3Bnb4Backend::forward: context window exceeded (cached_len=" +
                                 std::to_string(cached_len) + ", max_seq_len=" +
                                 std::to_string(max_seq_len_) + ")");
    }

    ensure_runtime_buffers(ctx, seq_len);
    if (seq_len > 1) {
        if (start_pos == 0) {
            ensure_prefill_scores(ctx, seq_len);
        } else {
            ensure_incr_prefill_scores(ctx, seq_len, start_pos + seq_len);
        }
    }

    // 1. Embedding lookup
    ops::embedding_lookup(ctx, *embed_weight_, token_ids_device, seq_len, buf_.hidden, H);

    // 2. Initial norm for the first layer
    ops::rms_norm(ctx, buf_.hidden, *layers_[0].input_ln_weight,
                  cfg_.rms_norm_eps, buf_.normed, seq_len, H);

    // 3. Iterate through transformer layers
    for (int i = 0; i < cfg_.num_hidden_layers; i++) {
        auto& layer = layers_[i];
        Tensor q_decode_view;
        Tensor* q_for_attn = &buf_.q;

        if (seq_len == 1) {
            // Decode path: fused QKV projection via Bnb4BitLinear
            layer.qkv_proj.forward(ctx, linear_scratch_, buf_.normed, buf_.qkv, 1);
            bf16* qkv_ptr = static_cast<bf16*>(buf_.qkv.data());
            q_decode_view = Tensor::view(ctx, qkv_ptr, {1, (int64_t)QD});
            Tensor k_view = Tensor::view(ctx, qkv_ptr + QD, {1, (int64_t)KVD});
            Tensor v_view = Tensor::view(ctx, qkv_ptr + QD + KVD, {1, (int64_t)KVD});
            q_for_attn = &q_decode_view;

            ops::decode_prepare_qkv(ctx, q_decode_view, k_view, v_view, buf_.rope_freq,
                                    *layer.q_norm_weight, *layer.k_norm_weight,
                                    kv_cache_.k_cache(i), kv_cache_.v_cache(i),
                                    start_pos,
                                    cfg_.num_attention_heads, cfg_.num_key_value_heads,
                                    cfg_.head_dim, cfg_.rms_norm_eps, cfg_.rope_theta);
        } else {
            // Prefill path
            layer.qkv_proj.forward(ctx, linear_scratch_, buf_.normed, buf_.qkv, seq_len);
            ops::split_qkv(ctx, buf_.qkv, buf_.q, buf_.k, buf_.v, seq_len, QD, KVD);

            ops::head_rms_norm(ctx, buf_.q, *layer.q_norm_weight,
                               cfg_.rms_norm_eps, seq_len,
                               cfg_.num_attention_heads, cfg_.head_dim);
            ops::head_rms_norm(ctx, buf_.k, *layer.k_norm_weight,
                               cfg_.rms_norm_eps, seq_len,
                               cfg_.num_key_value_heads, cfg_.head_dim);

            ops::apply_rope(ctx, buf_.q, buf_.k, seq_len, start_pos,
                            cfg_.num_attention_heads, cfg_.num_key_value_heads,
                            cfg_.head_dim, cfg_.rope_theta);

            ops::copy_to_cache(ctx, buf_.k, kv_cache_.k_cache(i),
                               seq_len, start_pos,
                               cfg_.num_key_value_heads, cfg_.head_dim,
                               kv_cache_.max_length());
            ops::copy_to_cache(ctx, buf_.v, kv_cache_.v_cache(i),
                               seq_len, start_pos,
                               cfg_.num_key_value_heads, cfg_.head_dim,
                               kv_cache_.max_length());
        }

        // Attention
        if (seq_len == 1) {
            ops::attention_decode(ctx, *q_for_attn, kv_cache_.k_cache(i), kv_cache_.v_cache(i),
                                  buf_.attn_out, buf_.decode_scores,
                                  cfg_.num_attention_heads, cfg_.num_key_value_heads,
                                  cfg_.head_dim, cached_len);
        } else if (start_pos == 0) {
            ops::attention_prefill(ctx, buf_.q, buf_.k, buf_.v,
                                   buf_.attn_out, buf_.scores,
                                   seq_len,
                                   cfg_.num_attention_heads, cfg_.num_key_value_heads,
                                   cfg_.head_dim);
        } else {
            ops::attention_prefill_cached(ctx, buf_.q,
                                          kv_cache_.k_cache(i), kv_cache_.v_cache(i),
                                          buf_.attn_out, buf_.incr_scores,
                                          seq_len, start_pos,
                                          cfg_.num_attention_heads, cfg_.num_key_value_heads,
                                          cfg_.head_dim, kv_cache_.max_length());
        }

        // O projection
        layer.o_proj.forward(ctx, linear_scratch_, buf_.attn_out, buf_.gate, seq_len);

        // Fused: residual + post-attn norm
        ops::fused_add_rms_norm(ctx, buf_.hidden, buf_.gate,
                                *layer.post_attn_ln_weight, cfg_.rms_norm_eps,
                                buf_.normed, seq_len, H);

        // FFN
        if (seq_len == 1) {
            layer.gate_up_proj.forward(ctx, linear_scratch_, buf_.normed, buf_.gate_up, 1);
            ops::fused_gate_up_swiglu(ctx, buf_.gate_up, buf_.gate, FF);
        } else {
            layer.gate_up_proj.forward(ctx, linear_scratch_, buf_.normed, buf_.gate_up, seq_len);
            ops::split_gate_up(ctx, buf_.gate_up, buf_.gate, buf_.up, seq_len, FF);
            ops::swiglu(ctx, buf_.gate, buf_.up, buf_.gate, seq_len * FF);
        }

        // Down projection
        layer.down_proj.forward(ctx, linear_scratch_, buf_.gate, buf_.up, seq_len);

        // Residual + norm for next layer (or final)
        if (i < cfg_.num_hidden_layers - 1) {
            ops::fused_add_rms_norm(ctx, buf_.hidden, buf_.up,
                                    *layers_[i + 1].input_ln_weight, cfg_.rms_norm_eps,
                                    buf_.normed, seq_len, H);
        } else {
            ops::fused_add_rms_norm(ctx, buf_.hidden, buf_.up,
                                    *final_norm_weight_, cfg_.rms_norm_eps,
                                    buf_.normed, seq_len, H);
        }
    }

    // LM Head: logits for the last token only
    if (seq_len > 1) {
        bf16* last_token_ptr = static_cast<bf16*>(buf_.normed.data()) + (seq_len - 1) * H;
        Tensor last_hidden = Tensor::view(ctx, last_token_ptr, {1, (int64_t)H});
        lm_head_.forward(ctx, last_hidden, buf_.logits, 1);
    } else {
        lm_head_.forward(ctx, buf_.normed, buf_.logits, 1);
    }

    kv_cache_.advance(seq_len);

    return buf_.logits;
}

void Qwen3Bnb4Backend::reset() {
    kv_cache_.reset();
}

bool Qwen3Bnb4Backend::truncate_kv_cache(int new_len) {
    kv_cache_.truncate(new_len);
    return true;
}
