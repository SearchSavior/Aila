#pragma once

#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_sycl.hpp"
#include <sycl/sycl.hpp>
#include <cstddef>
#include <iostream>
#include <stdexcept>

// ============================================================
// SYCL + oneDNN 运行时上下文
// ============================================================
class Context {
public:
    Context() {
        q_ = sycl::queue{sycl::default_selector_v, sycl::property::queue::in_order()};
        eng_ = dnnl::sycl_interop::make_engine(q_.get_device(), q_.get_context());
        stream_ = dnnl::sycl_interop::make_stream(eng_, q_);

        auto dev = q_.get_device();
        std::cout << "[Context] Device: " << dev.get_info<sycl::info::device::name>() << std::endl;
        std::cout << "[Context] Global memory: "
                  << dev.get_info<sycl::info::device::global_mem_size>() / (1024*1024)
                  << " MB" << std::endl;
    }

    sycl::queue& queue() { return q_; }
    dnnl::engine& engine() { return eng_; }
    dnnl::stream& stream() { return stream_; }

    // USM Device memory allocation
    void* alloc_device(size_t bytes) {
        void* ptr = sycl::malloc_device(bytes, q_);
        if (!ptr) {
            throw std::runtime_error("GPU memory allocation failed: " + std::to_string(bytes) + " bytes");
        }
        return ptr;
    }

    void free_device(void* ptr) {
        if (ptr) sycl::free(ptr, q_);
    }

    // Host -> Device copy (blocking, for synchronous operations)
    void memcpy_h2d(void* dst, const void* src, size_t bytes) {
        q_.memcpy(dst, src, bytes).wait();
    }

    // Device -> Host copy (blocking, for synchronous operations)
    void memcpy_d2h(void* dst, const void* src, size_t bytes) {
        q_.memcpy(dst, src, bytes).wait();
    }

    // Async Host -> Device copy (non-blocking, for pipeline)
    sycl::event memcpy_h2d_async(void* dst, const void* src, size_t bytes) {
        return q_.memcpy(dst, src, bytes);  // No wait!
    }

    // Async Device -> Host copy (non-blocking, for pipeline)
    sycl::event memcpy_d2h_async(void* dst, const void* src, size_t bytes) {
        return q_.memcpy(dst, src, bytes);  // No wait!
    }

    void synchronize() {
        q_.wait_and_throw();
    }

private:
    sycl::queue q_;
    dnnl::engine eng_;
    dnnl::stream stream_;
};
