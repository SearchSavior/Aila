#include "Qwen3.hpp"
#include <iostream>
#include <string>
#include <cassert>

using bf16 = sycl::ext::oneapi::bfloat16;

// ============================================================
// Load model weights from SafeTensors
// ============================================================

void Qwen3Model::load(Context& ctx, ModelWeights& weights, const Qwen3Config& config, int max_seq_len) {
    config_ = config;
    max_seq_len_ = max_seq_len;

    int H = config.hidden_size;
    int QD = config.num_attention_heads * config.head_dim;  // 2048
    int KVD = config.num_key_value_heads * config.head_dim;  // 1024
    int FF = config.intermediate_size;  // 3072

    // Helper: 物理转置权重 [out, in] -> [in, out]
    auto transpose_weight = [&](const std::string& name) {
        Tensor& src = weights.get(name);
        int64_t out_f = src.shape(0);
        int64_t in_f = src.shape(1);
        
        // 分配转置后的空间
        Tensor dst = Tensor::allocate(ctx, {in_f, out_f}, src.dtype());
        ops::transpose(ctx, src, dst);
        
        // 替换原始 tensor
        weights.replace(name, std::move(dst));
        return &weights.get(name);
    };

    // --- Embedding ---
    embed_weight_ = &weights.get("model.embed_tokens.weight");
    std::cout << "[Qwen3] embed_tokens loaded" << std::endl;

    // --- Layers ---
    layers_.resize(config.num_hidden_layers);
    for (int i = 0; i < config.num_hidden_layers; i++) {
        auto& layer = layers_[i];
        std::string prefix = "model.layers." + std::to_string(i) + ".";

        layer.input_ln_weight = &weights.get(prefix + "input_layernorm.weight");
        layer.post_attn_ln_weight = &weights.get(prefix + "post_attention_layernorm.weight");

        // 加载并转置权重
        Tensor* qw = transpose_weight(prefix + "self_attn.q_proj.weight");
        layer.q_proj.init(ctx, *qw, H, QD, true);

        Tensor* kw = transpose_weight(prefix + "self_attn.k_proj.weight");
        layer.k_proj.init(ctx, *kw, H, KVD, true);

        Tensor* vw = transpose_weight(prefix + "self_attn.v_proj.weight");
        layer.v_proj.init(ctx, *vw, H, KVD, true);

        Tensor* ow = transpose_weight(prefix + "self_attn.o_proj.weight");
        layer.o_proj.init(ctx, *ow, QD, H, true);

        // Qwen3 QK-norm: per-head RMSNorm on Q and K
        layer.q_norm_weight = &weights.get(prefix + "self_attn.q_norm.weight");
        layer.k_norm_weight = &weights.get(prefix + "self_attn.k_norm.weight");

        Tensor* gw = transpose_weight(prefix + "mlp.gate_proj.weight");
        layer.gate_proj.init(ctx, *gw, H, FF, true);

        Tensor* uw = transpose_weight(prefix + "mlp.up_proj.weight");
        layer.up_proj.init(ctx, *uw, H, FF, true);

        Tensor* dw = transpose_weight(prefix + "mlp.down_proj.weight");
        layer.down_proj.init(ctx, *dw, FF, H, true);
    }
    std::cout << "[Qwen3] " << config.num_hidden_layers << " transformer layers loaded" << std::endl;

    // --- Final norm ---
    final_norm_weight_ = &weights.get("model.norm.weight");

    // --- LM Head ---
    if (weights.has("lm_head.weight")) {
        Tensor* lw = transpose_weight("lm_head.weight");
        lm_head_.init(ctx, *lw, H, config.vocab_size, true);
        std::cout << "[Qwen3] lm_head loaded (standalone)" << std::endl;
    } else {
        // lm_head shares embed_weight_, but it is [vocab, hidden]
        // Actually, in Qwen, lm_head is typically tied with embed,
        // and embed is [vocab, hidden]. Linear expects [hidden, vocab] after transpose.
        // We'll treat tied case differently or just transpose a copy.
        Tensor& src = *embed_weight_;
        Tensor dst = Tensor::allocate(ctx, {src.shape(1), src.shape(0)}, src.dtype());
        ops::transpose(ctx, src, dst);
        weights.put("lm_head.weight_preprocessed", std::move(dst));
        lm_head_.init(ctx, weights.get("lm_head.weight_preprocessed"), H, config.vocab_size, true);
        std::cout << "[Qwen3] lm_head (tied, preprocessed copy)" << std::endl;
    }

    // --- KV Cache ---
    kv_cache_.init(ctx, config, max_seq_len);

    // --- Activation buffers ---
    buf_.hidden   = Tensor::allocate(ctx, {(int64_t)max_seq_len, H});
    buf_.residual = Tensor::allocate(ctx, {(int64_t)max_seq_len, H});
    buf_.normed   = Tensor::allocate(ctx, {(int64_t)max_seq_len, H});
    buf_.q        = Tensor::allocate(ctx, {(int64_t)max_seq_len, (int64_t)QD});
    buf_.k        = Tensor::allocate(ctx, {(int64_t)max_seq_len, (int64_t)KVD});
    buf_.v        = Tensor::allocate(ctx, {(int64_t)max_seq_len, (int64_t)KVD});
    buf_.attn_out = Tensor::allocate(ctx, {(int64_t)max_seq_len, (int64_t)QD});
    buf_.gate     = Tensor::allocate(ctx, {(int64_t)max_seq_len, (int64_t)FF});
    buf_.up       = Tensor::allocate(ctx, {(int64_t)max_seq_len, (int64_t)FF});
    buf_.ffn_out  = Tensor::allocate(ctx, {(int64_t)max_seq_len, H});
    buf_.logits   = Tensor::allocate(ctx, {1, (int64_t)config.vocab_size});
    // Prefill attention scores: [num_heads, max_seq, max_seq] in f32
    buf_.scores   = Tensor::allocate(ctx,
        {(int64_t)config.num_attention_heads, (int64_t)max_seq_len, (int64_t)max_seq_len},
        dnnl::memory::data_type::f32);

    std::cout << "[Qwen3] Model fully loaded and initialized" << std::endl;
}

// ============================================================
// Forward pass
// ============================================================

Tensor& Qwen3Model::forward(Context& ctx, const int* token_ids_device, int seq_len) {
    int H = config_.hidden_size;
    int QD = config_.num_attention_heads * config_.head_dim;
    int KVD = config_.num_key_value_heads * config_.head_dim;
    int FF = config_.intermediate_size;
    int start_pos = kv_cache_.current_length();
    int cached_len = start_pos + seq_len;

    // 1. Embedding lookup
    ops::embedding_lookup(ctx, *embed_weight_, token_ids_device, seq_len, buf_.hidden, H);

    // 2. Initial norm for the first layer
    ops::rms_norm(ctx, buf_.hidden, *layers_[0].input_ln_weight,
                  config_.rms_norm_eps, buf_.normed, seq_len, H);

    // 3. Iterate through transformer layers
    for (int i = 0; i < config_.num_hidden_layers; i++) {
        auto& layer = layers_[i];

        // 4. Q/K/V projections
        layer.q_proj.forward(ctx, buf_.normed, buf_.q, seq_len);
        layer.k_proj.forward(ctx, buf_.normed, buf_.k, seq_len);
        layer.v_proj.forward(ctx, buf_.normed, buf_.v, seq_len);

        // 4.5 Qwen3 QK-norm: per-head RMSNorm on Q and K (before RoPE)
        ops::head_rms_norm(ctx, buf_.q, *layer.q_norm_weight,
                           config_.rms_norm_eps, seq_len,
                           config_.num_attention_heads, config_.head_dim);
        ops::head_rms_norm(ctx, buf_.k, *layer.k_norm_weight,
                           config_.rms_norm_eps, seq_len,
                           config_.num_key_value_heads, config_.head_dim);

        // 5. Apply RoPE to Q and K
        ops::apply_rope(ctx, buf_.q, buf_.k, seq_len, start_pos,
                        config_.num_attention_heads, config_.num_key_value_heads,
                        config_.head_dim, config_.rope_theta);

        // 6. Write K, V to cache
        ops::copy_to_cache(ctx, buf_.k, kv_cache_.k_cache(i),
                           seq_len, start_pos,
                           config_.num_key_value_heads, config_.head_dim,
                           kv_cache_.max_length());
        ops::copy_to_cache(ctx, buf_.v, kv_cache_.v_cache(i),
                           seq_len, start_pos,
                           config_.num_key_value_heads, config_.head_dim,
                           kv_cache_.max_length());

        // 7. Attention
        if (seq_len == 1) {
            // Decode mode: GEMV attention
            ops::attention_decode(ctx, buf_.q, kv_cache_.k_cache(i), kv_cache_.v_cache(i),
                                  buf_.attn_out,
                                  config_.num_attention_heads, config_.num_key_value_heads,
                                  config_.head_dim, cached_len);
        } else {
            // Prefill mode: full attention matrix
            ops::attention_prefill(ctx, buf_.q, buf_.k, buf_.v,
                                   buf_.attn_out, buf_.scores,
                                   seq_len,
                                   config_.num_attention_heads, config_.num_key_value_heads,
                                   config_.head_dim);
        }

        // 8. O projection
        // Use buf_.gate as a temporary addition buffer
        layer.o_proj.forward(ctx, buf_.attn_out, buf_.gate, seq_len);

        // 9. Fused: hidden += addition, then normed = norm(hidden)
        ops::fused_add_rms_norm(ctx, buf_.hidden, buf_.gate,
                                *layer.post_attn_ln_weight, config_.rms_norm_eps,
                                buf_.normed, seq_len, H);

        // 10. FFN (already use buf_.normed from fused op above)
        layer.gate_proj.forward(ctx, buf_.normed, buf_.gate, seq_len);
        layer.up_proj.forward(ctx, buf_.normed, buf_.up, seq_len);
        ops::swiglu(ctx, buf_.gate, buf_.up, buf_.gate, seq_len * FF);
        
        // 11. Down projection -> buf_.up (temp addition)
        layer.down_proj.forward(ctx, buf_.gate, buf_.up, seq_len);

        // 12. Final fused for sub-layer
        if (i < config_.num_hidden_layers - 1) {
            // hidden += addition, then normed = input_norm of NEXT layer
            ops::fused_add_rms_norm(ctx, buf_.hidden, buf_.up,
                                    *layers_[i + 1].input_ln_weight, config_.rms_norm_eps,
                                    buf_.normed, seq_len, H);
        } else {
            // Last layer: just residual add
            ops::residual_add(ctx, buf_.hidden, buf_.up, seq_len * H);
            ops::rms_norm(ctx, buf_.hidden, *final_norm_weight_,
                          config_.rms_norm_eps, buf_.normed, seq_len, H);
        }
    }

    // 17. LM Head: logits = normed[-1] @ embed_weight^T
    // Only compute logits for the last token
    if (seq_len > 1) {
        // Create a view pointing to the last token's hidden state
        bf16* last_token_ptr = static_cast<bf16*>(buf_.normed.data()) + (seq_len - 1) * H;
        Tensor last_hidden = Tensor::view(ctx, last_token_ptr, {1, (int64_t)H});
        lm_head_.forward(ctx, last_hidden, buf_.logits, 1);
    } else {
        lm_head_.forward(ctx, buf_.normed, buf_.logits, 1);
    }

    // Update KV cache position
    kv_cache_.advance(seq_len);

    return buf_.logits;
}

void Qwen3Model::reset() {
    kv_cache_.reset();
}
