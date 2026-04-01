#pragma once

#include "../src/core/Context.hpp"
#include "../src/models/Qwen3.hpp"
#include "../src/utils/Tokenizer.hpp"
#include "../src/utils/SafeTensors.hpp"
#include "Types.hpp"
#include <string>
#include <functional>
#include <chrono>
#include <iostream>

// ============================================================
// Inference Engine: orchestrates loading, tokenization, inference
// ============================================================
class InferenceEngine {
public:
    InferenceEngine() = default;

    // Initialize: load model + tokenizer from model directory
    bool init(const std::string& model_dir, int max_seq_len = 4096) {
        model_dir_ = model_dir;

        std::cout << "========================================" << std::endl;
        std::cout << "  Aila Inference Engine (Qwen3-0.6B)" << std::endl;
        std::cout << "========================================" << std::endl;

        // 1. Create context
        std::cout << "\n[1/3] Initializing GPU context..." << std::endl;
        ctx_ = std::make_unique<Context>();

        // 2. Load tokenizer
        std::cout << "\n[2/3] Loading tokenizer..." << std::endl;
        if (!tokenizer_.load(model_dir)) {
            std::cerr << "Failed to load tokenizer" << std::endl;
            return false;
        }

        // 3. Load model weights
        std::cout << "\n[3/3] Loading model weights..." << std::endl;
        std::string safetensors_path = model_dir + "/model.safetensors";
        weights_ = std::make_unique<ModelWeights>(LoadSafetensors(safetensors_path, *ctx_));

        // 4. Initialize model
        model_.load(*ctx_, *weights_, config_, max_seq_len);

        std::cout << "\n========================================" << std::endl;
        std::cout << "  Engine ready! Type your message." << std::endl;
        std::cout << "========================================\n" << std::endl;

        return true;
    }

    // Generate response from prompt
    // token_callback is called for each generated token (streaming)
    std::string generate(const std::string& user_message,
                         const GenerationConfig& gen_config = GenerationConfig(),
                         std::function<void(const std::string&)> token_callback = nullptr) {

        // Reset KV cache for new conversation
        model_.reset();

        // Tokenize with ChatML template (thinking mode bypassed via injected tokens)
        auto input_ids = tokenizer_.apply_chat_template(
            "You are a helpful assistant.", user_message);

        std::cout << "[Generate] Prompt tokens: " << input_ids.size() << std::endl;

        // Upload token IDs to GPU (async, will be waited by first kernel)
        int* token_ids_device = static_cast<int*>(ctx_->alloc_device(input_ids.size() * sizeof(int)));
        ctx_->memcpy_h2d_async(token_ids_device, input_ids.data(), input_ids.size() * sizeof(int));

        // Prefill: process all prompt tokens at once
        auto t_start = std::chrono::high_resolution_clock::now();
        Tensor& logits = model_.forward(*ctx_, token_ids_device, static_cast<int>(input_ids.size()));
        ctx_->synchronize();
        auto t_prefill = std::chrono::high_resolution_clock::now();

        double prefill_ms = std::chrono::duration<double, std::milli>(t_prefill - t_start).count();
        std::cout << "[Generate] Prefill: " << prefill_ms << " ms ("
                  << input_ids.size() / (prefill_ms / 1000.0) << " tok/s)" << std::endl;

        ctx_->free_device(token_ids_device);

        // Allocate buffers on GPU for decode loop
        int* single_token_device = static_cast<int*>(ctx_->alloc_device(sizeof(int)));
        int* argmax_result_device = static_cast<int*>(ctx_->alloc_device(sizeof(int)));

        // Sample first token
        int next_token = 0;
        if (gen_config.do_sample) {
            next_token = ops::topk_sample(*ctx_, logits, config_.vocab_size,
                                          gen_config.temperature, gen_config.top_k);
            ctx_->memcpy_h2d(argmax_result_device, &next_token, sizeof(int));
        } else {
            ops::argmax(*ctx_, logits, config_.vocab_size, argmax_result_device);
            // 此时不进行同步 block 的 d2h.
        }

        // Decode loop (Asynchronous Pipeline)
        std::string output_text;
        int generated_count = 0;
        auto t_decode_start = std::chrono::high_resolution_clock::now();

        while (generated_count < gen_config.max_new_tokens) {
            // 启动该轮的前向传播，且完全无需 CPU 等待！
            // 此时 argmax_result_device 已经准备在 GPU 上
            Tensor& logits_next = model_.forward(*ctx_, argmax_result_device, 1);

            // 在 GPU 执行 next token 推理的同时，CPU 同步取回当前刚刚发射的 token
            // 由于 SYCL In-order 队列，memcpy_d2h 会等待上一轮的 argmax 或上上轮的 forward 完成
            // 而当前刚刚塞入的 forwarded 已经被注册到了 SYCL 后台
            ctx_->memcpy_d2h(&next_token, argmax_result_device, sizeof(int));

            if (tokenizer_.is_eos(next_token)) {
                break;
            }

            // Decode token to text and print on Host while GPU is crunching logits_next
            std::string token_text = tokenizer_.decode(next_token);
            output_text += token_text;
            if (token_callback) {
                token_callback(token_text);
            }
            generated_count++;

            // 为下一轮准备 argmax (依然异步挂载)
            if (gen_config.do_sample) {
                // 等待上一轮的结果回传完毕，这里有同步损耗，但仅针对 top_k
                ctx_->synchronize();
                next_token = ops::topk_sample(*ctx_, logits_next, config_.vocab_size,
                                              gen_config.temperature, gen_config.top_k);
                ctx_->memcpy_h2d(argmax_result_device, &next_token, sizeof(int));
            } else {
                // argmax 异步挂载到队列
                ops::argmax(*ctx_, logits_next, config_.vocab_size, argmax_result_device);
            }
        }

        // flush stream
        ctx_->synchronize();

        ctx_->free_device(argmax_result_device);

        auto t_decode_end = std::chrono::high_resolution_clock::now();
        double decode_ms = std::chrono::duration<double, std::milli>(t_decode_end - t_decode_start).count();

        std::cout << "\n[Generate] Decoded " << generated_count << " tokens in "
                  << decode_ms << " ms ("
                  << (generated_count > 0 ? generated_count / (decode_ms / 1000.0) : 0)
                  << " tok/s)" << std::endl;

        return output_text;
    }

private:
    std::string model_dir_;
    Qwen3Config config_;
    std::unique_ptr<Context> ctx_;
    std::unique_ptr<ModelWeights> weights_;
    Qwen3Model model_;
    Tokenizer tokenizer_;
};
