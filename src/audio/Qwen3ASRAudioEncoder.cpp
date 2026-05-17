#include "Qwen3ASRAudioEncoder.hpp"
#include "../core/Context.hpp"
#include "../utils/EnvUtils.hpp"
#include <cmath>
#include <algorithm>
#include <fstream>
#include <cstdint>

using bf16 = sycl::ext::oneapi::bfloat16;

namespace aila {
namespace audio {

// ---- helpers ----

static void nchw_to_ntch(Context& ctx, Tensor& src, Tensor& dst,
                          int C, int H, int W) {
    // src: [C*H*W] NCHW row-major (N=1)
    // dst: [W, C*H] row-major
    bf16* s = src.data_as<bf16>();
    bf16* d = dst.data_as<bf16>();
    int ch = C * H;
    ctx.queue().parallel_for(sycl::range<1>(W * ch),
        [=](sycl::id<1> idx) {
            int i = static_cast<int>(idx[0]);
            int t = i / ch;        // w dimension → time
            int r = i % ch;        // c*H combined
            // NCHW: [n][c][h][w], n=0
            int c = r / H;
            int h = r % H;
            int src_idx = ((c * H) + h) * W + t;
            d[i] = s[src_idx];
        });
}

Qwen3ASRAudioEncoder::~Qwen3ASRAudioEncoder() = default;

Tensor* Qwen3ASRAudioEncoder::get_tensor(ModelWeights& weights,
                                          const std::string& name,
                                          std::string* error_message,
                                          bool required) {
    if (!weights.has(name)) {
        if (required && error_message) {
            *error_message = "Missing weight: " + name;
        }
        return nullptr;
    }
    return &weights.get(name);
}

void Qwen3ASRAudioEncoder::ensure_buffers(int seq_len) {
    if (seq_len <= buf_capacity_) return;
    buf_capacity_ = seq_len + 64;  // padding

    auto* ctx = ctx_;
    int d = cfg_.d_model;
    int ffn = cfg_.encoder_ffn_dim;
    int heads = cfg_.encoder_attention_heads;
    int hd = cfg_.head_dim;

    enc_hidden_ = Tensor::allocate(*ctx, {buf_capacity_, d});
    enc_normed_ = Tensor::allocate(*ctx, {buf_capacity_, d});
    enc_q_ = Tensor::allocate(*ctx, {buf_capacity_, d});
    enc_k_ = Tensor::allocate(*ctx, {buf_capacity_, d});
    enc_v_ = Tensor::allocate(*ctx, {buf_capacity_, d});
    enc_attn_out_ = Tensor::allocate(*ctx, {buf_capacity_, d});
    enc_ffn_ = Tensor::allocate(*ctx, {buf_capacity_, ffn});
    enc_proj1_ = Tensor::allocate(*ctx, {buf_capacity_, d});
    enc_scores_ = Tensor::allocate(*ctx, {heads, buf_capacity_, buf_capacity_},
                                    dnnl::memory::data_type::f32);
}

// ---- load ----

bool Qwen3ASRAudioEncoder::load(Context& ctx,
                                 ModelWeights& weights,
                                 const AudioEncoderConfig& cfg,
                                 std::string* error_message) {
    ctx_ = &ctx;
    cfg_ = cfg;
    loaded_ = false;

    auto err = [&](const std::string& msg) {
        if (error_message) *error_message = msg;
    };

    auto G = [&](const std::string& name, bool required = true) -> Tensor* {
        return get_tensor(weights, "thinker.audio_tower." + name, error_message, required);
    };

    // Conv2D frontend
    conv2d1_weight_ = G("conv2d1.weight");
    conv2d1_bias_   = G("conv2d1.bias");
    conv2d2_weight_ = G("conv2d2.weight");
    conv2d2_bias_   = G("conv2d2.bias");
    conv2d3_weight_ = G("conv2d3.weight");
    conv2d3_bias_   = G("conv2d3.bias");
    if (!conv2d1_weight_ || !conv2d2_weight_ || !conv2d3_weight_) {
        err("Missing conv2d weights"); return false;
    }

    // conv_out Linear: in_features = downsample_hidden_size * feat_height
    // feat_height = ((mel_bins-1)//2+1-1)//2+1-1)//2+1 = 16 for 128 mel bins
    int feat_height = cfg.num_mel_bins;
    for (int i = 0; i < 3; ++i) {
        feat_height = (feat_height - 1) / 2 + 1;
    }
    int conv_in_dim = cfg.downsample_hidden_size * feat_height;  // 480 * 16 = 7680
    Tensor* conv_out_weight = G("conv_out.weight");
    if (conv_out_weight) {
        conv_out_.init(ctx, *conv_out_weight, conv_in_dim, cfg.d_model);
    } else {
        err("Missing conv_out.weight"); return false;
    }

    // Encoder layers
    layers_.resize(cfg.encoder_layers);
    for (int i = 0; i < cfg.encoder_layers; ++i) {
        auto& L = layers_[i];
        std::string p = "layers." + std::to_string(i) + ".";

        L.sa_ln_weight = G(p + "self_attn_layer_norm.weight");
        L.sa_ln_bias   = G(p + "self_attn_layer_norm.bias");
        L.final_ln_weight = G(p + "final_layer_norm.weight");
        L.final_ln_bias   = G(p + "final_layer_norm.bias");

        Tensor* qw = G(p + "self_attn.q_proj.weight");
        Tensor* kw = G(p + "self_attn.k_proj.weight");
        Tensor* vw = G(p + "self_attn.v_proj.weight");
        Tensor* ow = G(p + "self_attn.out_proj.weight");
        if (!qw || !kw || !vw || !ow) { err("Missing attn weights layer " + std::to_string(i)); return false; }

        L.q_proj.init(ctx, *qw, cfg.d_model, cfg.d_model);
        L.k_proj.init(ctx, *kw, cfg.d_model, cfg.d_model);
        L.v_proj.init(ctx, *vw, cfg.d_model, cfg.d_model);
        L.out_proj.init(ctx, *ow, cfg.d_model, cfg.d_model);

        L.q_bias = G(p + "self_attn.q_proj.bias", false);
        L.k_bias = G(p + "self_attn.k_proj.bias", false);
        L.v_bias = G(p + "self_attn.v_proj.bias", false);
        L.out_bias = G(p + "self_attn.out_proj.bias", false);

        Tensor* fc1w = G(p + "fc1.weight");
        Tensor* fc2w = G(p + "fc2.weight");
        if (!fc1w || !fc2w) { err("Missing fc weights layer " + std::to_string(i)); return false; }

        L.fc1.init(ctx, *fc1w, cfg.d_model, cfg.encoder_ffn_dim);
        L.fc2.init(ctx, *fc2w, cfg.encoder_ffn_dim, cfg.d_model);
        L.fc1_bias = G(p + "fc1.bias", false);
        L.fc2_bias = G(p + "fc2.bias", false);
    }

    // Output projection
    ln_post_weight = G("ln_post.weight");
    ln_post_bias   = G("ln_post.bias");
    if (!ln_post_weight) { err("Missing ln_post.weight"); return false; }

    Tensor* proj1_weight = G("proj1.weight");
    Tensor* proj2_weight = G("proj2.weight");
    if (!proj1_weight || !proj2_weight) { err("Missing output projection weights"); return false; }
    proj1_.init(ctx, *proj1_weight, cfg.d_model, cfg.d_model);
    proj2_.init(ctx, *proj2_weight, cfg.d_model, cfg.output_dim);

    if (cfg.output_dim <= 0) {
        err("Invalid audio output_dim");
        return false;
    }

    loaded_ = true;
    return true;
}

// ---- encode ----

bool Qwen3ASRAudioEncoder::encode(Context& ctx,
                                   Tensor& mel_input, int mel_time_len,
                                   Tensor& output_features, int& output_len,
                                   std::string* error_message) {
    if (!loaded_) {
        if (error_message) *error_message = "Audio encoder not loaded";
        return false;
    }

    int d = cfg_.d_model;
    int ffn = cfg_.encoder_ffn_dim;
    int heads = cfg_.encoder_attention_heads;
    int hd = cfg_.head_dim;
    int ds = cfg_.downsample_hidden_size;
    int mel_bins = cfg_.num_mel_bins;

    // Chunked processing: split mel into chunks of n_window*2 = 100 frames
    int chunk_size = cfg_.n_window * 2;  // 100
    int n_chunks = (mel_time_len + chunk_size - 1) / chunk_size;
    int feat_height = mel_bins;
    for (int i = 0; i < 3; ++i) feat_height = (feat_height - 1) / 2 + 1;  // 16
    int conv_flat_dim = ds * feat_height;  // 7680

    // Compute per-chunk conv output length (13 frames per 100-frame chunk)
    int chunk_out_w = 0;
    int total_out = 0;
    std::vector<int> chunk_out_lengths;
    for (int ci = 0; ci < n_chunks; ++ci) {
        int chunk_start = ci * chunk_size;
        int this_chunk_len = std::min(chunk_size, mel_time_len - chunk_start);
        int cw = this_chunk_len;
        for (int s = 0; s < 3; ++s) cw = (cw + 2 - 3) / 2 + 1;
        chunk_out_lengths.push_back(cw);
        total_out += cw;
        if (ci == 0) chunk_out_w = cw;  // first chunk determines padded width
    }

    // Allocate buffer for all chunks' conv output
    Tensor all_conv_out = Tensor::allocate(ctx, {total_out, d});
    int out_offset = 0;

    // Process each chunk through conv2d independently
    for (int ci = 0; ci < n_chunks; ++ci) {
        int chunk_start = ci * chunk_size;
        int this_chunk_len = std::min(chunk_size, mel_time_len - chunk_start);
        int cw = chunk_out_lengths[ci];

        // Copy chunk: mel_input({1,mel_bins,padded}) → chunk_mel({1,1,mel_bins,chunk_len})
        int mel_row_len = mel_input.shape(2);
        Tensor chunk_mel = Tensor::allocate(ctx, {1, 1, mel_bins, this_chunk_len});
        {
            bf16* src_base = mel_input.data_as<bf16>();
            bf16* dst = chunk_mel.data_as<bf16>();
            int ck = chunk_start, cl = this_chunk_len;
            int mb = mel_bins, mrl = mel_row_len;
            ctx.queue().parallel_for(sycl::range<1>(mb * cl), [=](sycl::id<1> idx) {
                int i = (int)idx[0];
                int h = i / cl;
                int w = i % cl;
                dst[h * cl + w] = src_base[h * mrl + ck + w];
            });
        }

        int c1_h = (mel_bins + 2 - 3) / 2 + 1;
        int c1_w = (this_chunk_len + 2 - 3) / 2 + 1;
        Tensor c1_out = Tensor::allocate(ctx, {1, ds, c1_h, c1_w});
        ops::conv2d_gelu(ctx, chunk_mel, *conv2d1_weight_, *conv2d1_bias_,
                         c1_out, 1, 1, ds, mel_bins, this_chunk_len, c1_h, c1_w);

        int c2_h = (c1_h + 2 - 3) / 2 + 1;
        int c2_w = (c1_w + 2 - 3) / 2 + 1;
        Tensor c2_out = Tensor::allocate(ctx, {1, ds, c2_h, c2_w});
        ops::conv2d_gelu(ctx, c1_out, *conv2d2_weight_, *conv2d2_bias_,
                         c2_out, 1, ds, ds, c1_h, c1_w, c2_h, c2_w);

        int c3_h = (c2_h + 2 - 3) / 2 + 1;
        int c3_w = (c2_w + 2 - 3) / 2 + 1;
        Tensor c3_out = Tensor::allocate(ctx, {1, ds, c3_h, c3_w});
        ops::conv2d_gelu(ctx, c2_out, *conv2d3_weight_, *conv2d3_bias_,
                         c3_out, 1, ds, ds, c2_h, c2_w, c3_h, c3_w);
        if (ci == 0) { // save first chunk c3
            std::vector<bf16> dbg(c3_out.numel());
            ctx.memcpy_d2h(dbg.data(), c3_out.data(), dbg.size()*sizeof(bf16)); ctx.synchronize();
            std::vector<float> f32(dbg.size()); for(size_t i=0;i<dbg.size();++i) f32[i]=(float)dbg[i];
            std::ofstream ofs("debug_cpp_c3.bin",std::ios::binary);
            int32_t hdr[4]={1,ds,c3_h,c3_w}; ofs.write((char*)hdr,16); ofs.write((char*)f32.data(),f32.size()*4);
        }

        // Reshape + conv_out
        Tensor conv_flat = Tensor::allocate(ctx, {cw, conv_flat_dim});
        nchw_to_ntch(ctx, c3_out, conv_flat, ds, c3_h, c3_w);

        // View of output buffer for this chunk
        bf16* out_ptr = all_conv_out.data_as<bf16>() + out_offset * d;
        Tensor chunk_dst = Tensor::view(ctx, out_ptr, {cw, d});
        conv_out_.forward(ctx, conv_flat, chunk_dst, cw);

        // Add sinusoidal PE (per-chunk positions 0..cw-1)
        ops::sinusoidal_position_embedding(ctx, chunk_dst, cw, d);

        out_offset += cw;
    }

    // Now run encoder layers on full sequence
    int audio_seq_len = total_out;
    ensure_buffers(audio_seq_len);

    // Copy all_conv_out to enc_hidden_
    ops::copy_tensor(ctx, all_conv_out, enc_hidden_, audio_seq_len * d);

    float ln_eps = 1e-5f;
    for (int i = 0; i < cfg_.encoder_layers; ++i) {
        auto& L = layers_[i];

        ops::layer_norm(ctx, enc_hidden_, *L.sa_ln_weight, *L.sa_ln_bias,
                        ln_eps, enc_normed_, audio_seq_len, d);

        L.q_proj.forward(ctx, enc_normed_, enc_q_, audio_seq_len);
        L.k_proj.forward(ctx, enc_normed_, enc_k_, audio_seq_len);
        L.v_proj.forward(ctx, enc_normed_, enc_v_, audio_seq_len);

        if (L.q_bias) ops::bias_add_inplace(ctx, enc_q_, *L.q_bias, audio_seq_len, d);
        if (L.k_bias) ops::bias_add_inplace(ctx, enc_k_, *L.k_bias, audio_seq_len, d);
        if (L.v_bias) ops::bias_add_inplace(ctx, enc_v_, *L.v_bias, audio_seq_len, d);

        ops::attention_bidi(ctx, enc_q_, enc_k_, enc_v_,
                           enc_attn_out_, enc_scores_,
                           audio_seq_len, heads, hd);

        L.out_proj.forward(ctx, enc_attn_out_, enc_normed_, audio_seq_len);
        if (L.out_bias) ops::bias_add_inplace(ctx, enc_normed_, *L.out_bias, audio_seq_len, d);
        ops::residual_add(ctx, enc_hidden_, enc_normed_, audio_seq_len * d);

        ops::layer_norm(ctx, enc_hidden_, *L.final_ln_weight, *L.final_ln_bias,
                        ln_eps, enc_normed_, audio_seq_len, d);

        L.fc1.forward(ctx, enc_normed_, enc_ffn_, audio_seq_len);
        if (L.fc1_bias) ops::bias_add_inplace(ctx, enc_ffn_, *L.fc1_bias, audio_seq_len, ffn);
        ops::gelu_tanh_inplace(ctx, enc_ffn_, audio_seq_len * ffn);

        L.fc2.forward(ctx, enc_ffn_, enc_normed_, audio_seq_len);
        if (L.fc2_bias) ops::bias_add_inplace(ctx, enc_normed_, *L.fc2_bias, audio_seq_len, d);
        ops::residual_add(ctx, enc_hidden_, enc_normed_, audio_seq_len * d);
    }

    // Output projection
    ops::layer_norm(ctx, enc_hidden_, *ln_post_weight, *ln_post_bias,
                    ln_eps, enc_normed_, audio_seq_len, d);
    proj1_.forward(ctx, enc_normed_, enc_proj1_, audio_seq_len);
    ops::gelu_tanh_inplace(ctx, enc_proj1_, audio_seq_len * d);
    proj2_.forward(ctx, enc_proj1_, output_features, audio_seq_len);

    output_len = audio_seq_len;
    return true;
}

} // namespace audio
} // namespace aila
