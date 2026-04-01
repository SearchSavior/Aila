#include "Ops.hpp"
#include <sycl/sycl.hpp>
#include <cmath>

using bf16 = sycl::ext::oneapi::bfloat16;

namespace ops {

// ============================================================
// SYCL Kernel: Embedding Lookup
// ============================================================

void embedding_lookup(Context& ctx, Tensor& table,
                       const int* token_ids_device, int seq_len,
                       Tensor& output, int hidden_size) {
    bf16* table_ptr = static_cast<bf16*>(table.data());
    bf16* out_ptr = static_cast<bf16*>(output.data());

    ctx.queue().parallel_for(sycl::range<2>(seq_len, hidden_size),
        [=](sycl::id<2> idx) {
            int s = idx[0];
            int h = idx[1];
            int token_id = token_ids_device[s];
            out_ptr[s * hidden_size + h] = table_ptr[token_id * hidden_size + h];
        });
}

// ============================================================
// SYCL Kernel: RoPE (Rotary Position Embedding)
// ============================================================

void apply_rope(Context& ctx, Tensor& q, Tensor& k,
                 int seq_len, int start_pos,
                 int num_heads_q, int num_kv_heads, int head_dim,
                 float theta) {
    bf16* q_ptr = static_cast<bf16*>(q.data());
    bf16* k_ptr = static_cast<bf16*>(k.data());
    int q_dim = num_heads_q * head_dim;
    int k_dim = num_kv_heads * head_dim;
    int half_dim = head_dim / 2;

    ctx.queue().parallel_for(sycl::range<2>(seq_len, num_heads_q * half_dim),
        [=](sycl::id<2> idx) {
            int s = idx[0];
            int flat = idx[1];
            int h = flat / half_dim;   
            int d = flat % half_dim;   
            int pos = start_pos + s;

            float freq = 1.0f / sycl::pow(theta, (2.0f * d) / static_cast<float>(head_dim));
            float angle = static_cast<float>(pos) * freq;
            float cos_val = sycl::cos(angle);
            float sin_val = sycl::sin(angle);

            int q_base = s * q_dim + h * head_dim;
            float q0 = static_cast<float>(q_ptr[q_base + d]);
            float q1 = static_cast<float>(q_ptr[q_base + d + half_dim]);
            q_ptr[q_base + d]            = bf16(q0 * cos_val - q1 * sin_val);
            q_ptr[q_base + d + half_dim] = bf16(q1 * cos_val + q0 * sin_val);

            if (h < num_kv_heads) {
                int k_base = s * k_dim + h * head_dim;
                float k0 = static_cast<float>(k_ptr[k_base + d]);
                float k1 = static_cast<float>(k_ptr[k_base + d + half_dim]);
                k_ptr[k_base + d]            = bf16(k0 * cos_val - k1 * sin_val);
                k_ptr[k_base + d + half_dim] = bf16(k1 * cos_val + k0 * sin_val);
            }
        });
}

// ============================================================
// SYCL Kernel: SwiGLU
// output = silu(gate) * up = (gate * sigmoid(gate)) * up
// ============================================================

void swiglu(Context& ctx, Tensor& gate, Tensor& up, Tensor& output, int n) {
    using vec8 = sycl::vec<bf16, 8>;
    vec8* g_ptr = reinterpret_cast<vec8*>(gate.data());
    vec8* u_ptr = reinterpret_cast<vec8*>(up.data());
    vec8* o_ptr = reinterpret_cast<vec8*>(output.data());

    int n8 = n / 8; // Assumes perfectly aligned, which is true for Qwen3 0.6B hidden sizes

    ctx.queue().parallel_for(sycl::range<1>(n8),
        [=](sycl::id<1> i) {
            vec8 g_vec = g_ptr[i];
            vec8 u_vec = u_ptr[i];
            vec8 o_vec;

            for (int k = 0; k < 8; ++k) {
                float g = static_cast<float>(g_vec[k]);
                float u = static_cast<float>(u_vec[k]);
                float silu_g = g / (1.0f + sycl::exp(-g));
                o_vec[k] = bf16(silu_g * u);
            }
            o_ptr[i] = o_vec;
        });
}

// ============================================================
// SYCL Kernel: Residual Add (a += b, in-place)
// ============================================================

void residual_add(Context& ctx, Tensor& a, Tensor& b, int n) {
    using vec8 = sycl::vec<bf16, 8>;
    vec8* a_ptr = reinterpret_cast<vec8*>(a.data());
    vec8* b_ptr = reinterpret_cast<vec8*>(b.data());
    
    int n8 = n / 8;

    ctx.queue().parallel_for(sycl::range<1>(n8),
        [=](sycl::id<1> i) {
            vec8 va = a_ptr[i];
            vec8 vb = b_ptr[i];
            vec8 vo;
            for (int k = 0; k < 8; ++k) {
                float f_a = static_cast<float>(va[k]);
                float f_b = static_cast<float>(vb[k]);
                vo[k] = bf16(f_a + f_b);
            }
            a_ptr[i] = vo;
        });
}

// ============================================================
// SYCL Kernel: Copy Tensor
// ============================================================

void copy_tensor(Context& ctx, Tensor& src, Tensor& dst, int n) {
    using vec8 = sycl::vec<bf16, 8>;
    vec8* s_ptr = reinterpret_cast<vec8*>(src.data());
    vec8* d_ptr = reinterpret_cast<vec8*>(dst.data());
    
    int n8 = n / 8;

    ctx.queue().parallel_for(sycl::range<1>(n8),
        [=](sycl::id<1> i) {
            d_ptr[i] = s_ptr[i];
        });
}

} // namespace ops
