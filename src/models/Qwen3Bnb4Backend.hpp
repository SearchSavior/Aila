#pragma once

#include "IModelBackend.hpp"
#include <stdexcept>
#include <string>
#include <vector>

class Qwen3Bnb4Backend : public IModelBackend {
public:
    bool load(Context& ctx, ModelWeights& weights, const ModelSpec& spec,
              int max_seq_len, std::string* error_message) override {
        if (error_message) *error_message = "Qwen3 BNB4 backend not available (sources lost)";
        return false;
    }
    Tensor& forward(Context& ctx, const int*, int) override {
        throw std::runtime_error("Qwen3Bnb4Backend not available");
    }
    void reset() override {}
    void truncate_kv_cache(int) override {}
    int max_seq_len() const override { return 4096; }
    int vocab_size() const override { return 151936; }
    ModelFamily family() const override { return ModelFamily::Qwen3Dense; }
};
