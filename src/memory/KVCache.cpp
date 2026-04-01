#include "KVCache.hpp"
#include <iostream>

void KVCache::init(Context& ctx, const Qwen3Config& config, int max_seq_len) {
    max_len_ = max_seq_len;
    num_kv_heads_ = config.num_key_value_heads;
    head_dim_ = config.head_dim;
    current_len_ = 0;

    layers_.resize(config.num_hidden_layers);

    size_t per_tensor_bytes = (size_t)num_kv_heads_ * max_seq_len * head_dim_ * sizeof(uint16_t); // BF16
    size_t total_bytes = 0;

    for (int i = 0; i < config.num_hidden_layers; i++) {
        std::vector<int64_t> shape = {num_kv_heads_, (int64_t)max_seq_len, head_dim_};
        layers_[i].k = Tensor::allocate(ctx, shape, dnnl::memory::data_type::bf16);
        layers_[i].v = Tensor::allocate(ctx, shape, dnnl::memory::data_type::bf16);
        total_bytes += 2 * per_tensor_bytes;
    }

    std::cout << "[KVCache] Allocated " << config.num_hidden_layers << " layers, "
              << "max_seq=" << max_seq_len << ", "
              << total_bytes / (1024.0 * 1024.0) << " MB" << std::endl;
}
