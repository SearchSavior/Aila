#include "Ops.hpp"
#include <sycl/sycl.hpp>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <cmath>

using bf16 = sycl::ext::oneapi::bfloat16;

namespace ops {

// ============================================================
// SYCL Kernel: Argmax (GPU Side)
// ============================================================

void argmax(Context& ctx, Tensor& logits, int vocab_size, int* d_result) {
    bf16* l_ptr = static_cast<bf16*>(logits.data());
    
    int wg_size = 256;

    ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> local_max(sycl::range<1>(wg_size), cgh);
        sycl::local_accessor<int, 1> local_idx(sycl::range<1>(wg_size), cgh);

        cgh.parallel_for(sycl::nd_range<1>(wg_size, wg_size), [=](sycl::nd_item<1> item) {
            int lid = item.get_local_id(0);
            float max_val = -1e30f;
            int max_idx = 0;

            for (int i = lid; i < vocab_size; i += wg_size) {
                float v = static_cast<float>(l_ptr[i]);
                if (v > max_val) {
                    max_val = v;
                    max_idx = i;
                }
            }
            local_max[lid] = max_val;
            local_idx[lid] = max_idx;
            item.barrier(sycl::access::fence_space::local_space);

            for (int stride = wg_size / 2; stride > 0; stride >>= 1) {
                if (lid < stride) {
                    if (local_max[lid + stride] > local_max[lid]) {
                        local_max[lid] = local_max[lid + stride];
                        local_idx[lid] = local_idx[lid + stride];
                    }
                }
                item.barrier(sycl::access::fence_space::local_space);
            }

            if (lid == 0) *d_result = local_idx[0];
        });
    }); 
}

int argmax(Context& ctx, Tensor& logits, int vocab_size) {
    bf16* l_ptr = static_cast<bf16*>(logits.data());
    
    int wg_size = 256;
    int* d_result = static_cast<int*>(ctx.alloc_device(sizeof(int)));
    int h_result = 0;

    ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> local_max(sycl::range<1>(wg_size), cgh);
        sycl::local_accessor<int, 1> local_idx(sycl::range<1>(wg_size), cgh);

        cgh.parallel_for(sycl::nd_range<1>(wg_size, wg_size), [=](sycl::nd_item<1> item) {
            int lid = item.get_local_id(0);
            float max_val = -1e30f;
            int max_idx = 0;

            for (int i = lid; i < vocab_size; i += wg_size) {
                float v = static_cast<float>(l_ptr[i]);
                if (v > max_val) {
                    max_val = v;
                    max_idx = i;
                }
            }
            local_max[lid] = max_val;
            local_idx[lid] = max_idx;
            item.barrier(sycl::access::fence_space::local_space);

            for (int stride = wg_size / 2; stride > 0; stride >>= 1) {
                if (lid < stride) {
                    if (local_max[lid + stride] > local_max[lid]) {
                        local_max[lid] = local_max[lid + stride];
                        local_idx[lid] = local_idx[lid + stride];
                    }
                }
                item.barrier(sycl::access::fence_space::local_space);
            }

            if (lid == 0) *d_result = local_idx[0];
        });
    }).wait();

    ctx.memcpy_d2h(&h_result, d_result, sizeof(int));
    ctx.free_device(d_result);
    return h_result;
}

// ============================================================
// Top-k Sampling (CPU 端)
// ============================================================

int topk_sample(Context& ctx, Tensor& logits, int vocab_size,
                 float temperature, int top_k) {
    std::vector<bf16> host_logits_bf16(vocab_size);
    ctx.memcpy_d2h(host_logits_bf16.data(), logits.data(), vocab_size * sizeof(bf16));

    std::vector<float> logits_f(vocab_size);
    for (int i = 0; i < vocab_size; i++) {
        logits_f[i] = static_cast<float>(host_logits_bf16[i]) / temperature;
    }

    std::vector<int> indices(vocab_size);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + top_k, indices.end(),
        [&](int a, int b) { return logits_f[a] > logits_f[b]; });

    float max_val = logits_f[indices[0]];
    float sum = 0.0f;
    std::vector<float> probs(top_k);
    for (int i = 0; i < top_k; i++) {
        probs[i] = std::exp(logits_f[indices[i]] - max_val);
        sum += probs[i];
    }
    for (int i = 0; i < top_k; i++) {
        probs[i] /= sum;
    }

    static std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);

    float cumsum = 0.0f;
    for (int i = 0; i < top_k; i++) {
        cumsum += probs[i];
        if (r <= cumsum) {
            return indices[i];
        }
    }
    return indices[top_k - 1];
}

} // namespace ops
