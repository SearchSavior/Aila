#pragma once

#include <cstdlib>
#include <string>

// ============================================================
// Cross-platform environment variable utilities
// Consolidates duplicated read_env_* functions from multiple files.
// ============================================================

namespace aila {
namespace env {

inline bool read_flag(const char* name, bool default_value) {
#ifdef _WIN32
    char* value = nullptr;
    size_t len = 0;
    if (_dupenv_s(&value, &len, name) == 0 && value) {
        bool enabled = (std::atoi(value) != 0);
        free(value);
        return enabled;
    }
    if (value) free(value);
    return default_value;
#else
    const char* value = std::getenv(name);
    if (!value) return default_value;
    return (std::atoi(value) != 0);
#endif
}

inline int read_int(const char* name, int default_value) {
#ifdef _WIN32
    char* value = nullptr;
    size_t len = 0;
    if (_dupenv_s(&value, &len, name) == 0 && value) {
        int parsed = std::atoi(value);
        free(value);
        return parsed > 0 ? parsed : default_value;
    }
    if (value) free(value);
    return default_value;
#else
    const char* value = std::getenv(name);
    if (!value) return default_value;
    int parsed = std::atoi(value);
    return parsed > 0 ? parsed : default_value;
#endif
}

inline std::string read_string(const char* name, const std::string& default_value = "") {
#ifdef _WIN32
    char* value = nullptr;
    size_t len = 0;
    if (_dupenv_s(&value, &len, name) == 0 && value) {
        std::string result(value);
        free(value);
        return result;
    }
    if (value) free(value);
    return default_value;
#else
    const char* value = std::getenv(name);
    return value ? std::string(value) : default_value;
#endif
}

// Variant returning raw int (allows 0 or negative values)
inline int read_int_raw(const char* name, int default_value) {
#ifdef _WIN32
    char* value = nullptr;
    size_t len = 0;
    if (_dupenv_s(&value, &len, name) == 0 && value) {
        int parsed = std::atoi(value);
        free(value);
        return parsed;
    }
    if (value) free(value);
    return default_value;
#else
    const char* value = std::getenv(name);
    return value ? std::atoi(value) : default_value;
#endif
}

} // namespace env
} // namespace aila
