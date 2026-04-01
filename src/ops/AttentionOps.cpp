#include "Ops.hpp"
#include <sycl/sycl.hpp>
#include <cmath>

using bf16 = sycl::ext::oneapi::bfloat16;
using namespace sycl::ext::oneapi::experimental::matrix;

namespace ops {

// ============================================================
// SYCL Kernel: Attention Decode (seq_len=1, GQA) - Original
// ============================================================

void attention_decode(Context& ctx,
                       Tensor& q, Tensor& k_cache, Tensor& v_cache,
                       Tensor& output,
                       int num_heads, int num_kv_heads, int head_dim,
                       int cached_len) {
    bf16* q_ptr = static_cast<bf16*>(q.data());
    bf16* k_ptr = static_cast<bf16*>(k_cache.data());
    bf16* v_ptr = static_cast<bf16*>(v_cache.data());
    bf16* o_ptr = static_cast<bf16*>(output.data());

    int heads_per_kv = num_heads / num_kv_heads;
    float scale = 1.0f / sycl::sqrt(static_cast<float>(head_dim));
    int max_seq_len = static_cast<int>(k_cache.shape(1));

    int wg_size = 128; 

    ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> shared(sycl::range<1>(cached_len + wg_size), cgh);
        sycl::local_accessor<float, 1> q_cache(sycl::range<1>(head_dim), cgh);

        cgh.parallel_for(sycl::nd_range<1>(num_heads * wg_size, wg_size),
            [=](sycl::nd_item<1> item) {
                int head = item.get_group(0);
                int lid = item.get_local_id(0);
                int kv_head = head / heads_per_kv;

                if (lid < head_dim) {
                    q_cache[lid] = static_cast<float>(q_ptr[head * head_dim + lid]);
                }
                item.barrier(sycl::access::fence_space::local_space);

                int red_offset = cached_len; 
                for (int t = lid; t < cached_len; t += wg_size) {
                    float sum = 0.0f;
                    for (int d = 0; d < head_dim; d++) {
                        sum += q_cache[d] * static_cast<float>(k_ptr[kv_head * max_seq_len * head_dim + t * head_dim + d]);
                    }
                    shared[t] = sum * scale;
                }
                item.barrier(sycl::access::fence_space::local_space);

                float max_val = -1e30f;
                for (int t = lid; t < cached_len; t += wg_size) {
                    max_val = sycl::fmax(max_val, shared[t]);
                }
                shared[red_offset + lid] = max_val;
                item.barrier(sycl::access::fence_space::local_space);
                for (int stride = wg_size / 2; stride > 0; stride >>= 1) {
                    if (lid < stride) shared[red_offset+lid] = sycl::fmax(shared[red_offset+lid], shared[red_offset+lid+stride]);
                    item.barrier(sycl::access::fence_space::local_space);
                }
                max_val = shared[red_offset];

                float sum_val = 0.0f;
                for (int t = lid; t < cached_len; t += wg_size) {
                    float e = sycl::exp(shared[t] - max_val);
                    shared[t] = e;
                    sum_val += e;
                }
                shared[red_offset + lid] = sum_val;
                item.barrier(sycl::access::fence_space::local_space);
                for (int stride = wg_size / 2; stride > 0; stride >>= 1) {
                    if (lid < stride) shared[red_offset+lid] += shared[red_offset+lid+stride];
                    item.barrier(sycl::access::fence_space::local_space);
                }
                sum_val = shared[red_offset];

                for (int t = lid; t < cached_len; t += wg_size) {
                    shared[t] /= sum_val;
                }
                item.barrier(sycl::access::fence_space::local_space);

                if (lid < head_dim) {
                    float acc = 0.0f;
                    for (int t = 0; t < cached_len; t++) {
                        acc += shared[t] * static_cast<float>(v_ptr[kv_head * max_seq_len * head_dim + t * head_dim + lid]);
                    }
                    o_ptr[head * head_dim + lid] = bf16(acc);
                }
            });
    });
}

// ============================================================
// SYCL Kernel: Attention Prefill (seq_len > 1)
// ============================================================

void attention_prefill(Context& ctx,
                        Tensor& q, Tensor& k, Tensor& v,
                        Tensor& output, Tensor& scores_buf,
                        int seq_len,
                        int num_heads, int num_kv_heads, int head_dim) {
    bf16* q_ptr = static_cast<bf16*>(q.data());
    bf16* k_ptr = static_cast<bf16*>(k.data());
    bf16* v_ptr = static_cast<bf16*>(v.data());
    bf16* o_ptr = static_cast<bf16*>(output.data());

    int heads_per_kv = num_heads / num_kv_heads;
    float scale = 1.0f / sycl::sqrt(static_cast<float>(head_dim));
    int q_total = num_heads * head_dim;
    int k_total = num_kv_heads * head_dim;

    float* scores_device = static_cast<float*>(scores_buf.data());

    ctx.queue().parallel_for(sycl::range<3>(num_heads, seq_len, seq_len),
        [=](sycl::id<3> idx) {
            int h = idx[0];
            int qi = idx[1];
            int ki = idx[2];
            int kv_h = h / heads_per_kv;

            if (ki > qi) {
                scores_device[h * seq_len * seq_len + qi * seq_len + ki] = -1e30f;
                return;
            }

            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                float qv = static_cast<float>(q_ptr[qi * q_total + h * head_dim + d]);
                float kv = static_cast<float>(k_ptr[ki * k_total + kv_h * head_dim + d]);
                dot += qv * kv;
            }
            scores_device[h * seq_len * seq_len + qi * seq_len + ki] = dot * scale;
        });

    ctx.queue().parallel_for(sycl::range<2>(num_heads, seq_len),
        [=](sycl::id<2> idx) {
            int h = idx[0];
            int qi = idx[1];
            float* row = scores_device + h * seq_len * seq_len + qi * seq_len;

            float max_val = -1e30f;
            for (int ki = 0; ki <= qi; ki++) {
                max_val = sycl::fmax(max_val, row[ki]);
            }
            float sum = 0.0f;
            for (int ki = 0; ki <= qi; ki++) {
                row[ki] = sycl::exp(row[ki] - max_val);
                sum += row[ki];
            }
            for (int ki = 0; ki <= qi; ki++) {
                row[ki] /= sum;
            }
            for (int ki = qi + 1; ki < seq_len; ki++) {
                row[ki] = 0.0f;
            }
        });

    ctx.queue().parallel_for(sycl::range<3>(num_heads, seq_len, head_dim),
        [=](sycl::id<3> idx) {
            int h = idx[0];
            int qi = idx[1];
            int d = idx[2];
            int kv_h = h / heads_per_kv;

            float acc = 0.0f;
            float* row = scores_device + h * seq_len * seq_len + qi * seq_len;
            for (int t = 0; t <= qi; t++) {
                float vv = static_cast<float>(v_ptr[t * k_total + kv_h * head_dim + d]);
                acc += row[t] * vv;
            }
            o_ptr[qi * q_total + h * head_dim + d] = bf16(acc);
        });
}

// ============================================================
// SYCL Kernel: Copy K/V to Cache
// ============================================================

void copy_to_cache(Context& ctx, Tensor& new_kv, Tensor& cache,
                    int seq_len, int start_pos,
                    int num_heads, int head_dim, int max_seq_len) {
    bf16* src = static_cast<bf16*>(new_kv.data());
    bf16* dst = static_cast<bf16*>(cache.data());
    int total_dim = num_heads * head_dim;

    ctx.queue().parallel_for(sycl::range<2>(seq_len, total_dim),
        [=](sycl::id<2> idx) {
            int s = idx[0];
            int flat = idx[1];
            int h = flat / head_dim;   
            int d = flat % head_dim;   

            int src_idx = s * total_dim + flat;
            int dst_idx = h * max_seq_len * head_dim + (start_pos + s) * head_dim + d;
            dst[dst_idx] = src[src_idx];
        });
}

} // namespace ops
