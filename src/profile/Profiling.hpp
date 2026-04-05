#pragma once

#include <cstdarg>
#include <cstdio>
#include <chrono>
#include <string>
#include <mutex>

// ============================================================
// Aila Unified Logging & Profiling
// ============================================================

namespace aila {

enum class LogLevel { Debug = 0, Info = 1, Warning = 2, Error = 3 };

// C-compatible callback signature (also used by C API)
using LogCallback = void(*)(int level, const char* message, void* user_data);

// Set custom log callback. Pass nullptr to restore default (stdout/stderr).
void set_log_callback(LogCallback cb, void* user_data = nullptr);

// Set minimum log level (messages below this level are suppressed).
void set_log_level(LogLevel level);

// Core logging function (internal use, prefer macros below)
void log(LogLevel level, const char* fmt, ...);
void vlog(LogLevel level, const char* fmt, va_list args);

// Convenience macros
#define AILA_LOG_DEBUG(...) ::aila::log(::aila::LogLevel::Debug, __VA_ARGS__)
#define AILA_LOG_INFO(...)  ::aila::log(::aila::LogLevel::Info, __VA_ARGS__)
#define AILA_LOG_WARN(...)  ::aila::log(::aila::LogLevel::Warning, __VA_ARGS__)
#define AILA_LOG_ERROR(...) ::aila::log(::aila::LogLevel::Error, __VA_ARGS__)

// ============================================================
// Scoped Timer for performance profiling
// ============================================================
class ScopedTimer {
public:
    explicit ScopedTimer(const char* label)
        : label_(label), start_(std::chrono::high_resolution_clock::now()) {}

    ~ScopedTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start_).count();
        AILA_LOG_INFO("[Timer] %s: %.2f ms", label_, ms);
    }

    double elapsed_ms() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(now - start_).count();
    }

    // Non-copyable
    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

private:
    const char* label_;
    std::chrono::high_resolution_clock::time_point start_;
};

} // namespace aila
