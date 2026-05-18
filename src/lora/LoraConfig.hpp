#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <cmath>

namespace aila {
namespace lora {

struct LoraConfig {
    int r = 16;
    int lora_alpha = 32;
    float scaling = 2.0f;
    std::vector<std::string> target_modules;

    static LoraConfig from_json_file(const std::string& lora_dir, std::string* error = nullptr) {
        LoraConfig cfg;
        std::string config_path = lora_dir;
        if (config_path.back() != '/' && config_path.back() != '\\') {
            config_path += "/";
        }
        config_path += "adapter_config.json";

        std::ifstream f(config_path);
        if (!f.is_open()) {
            if (error) *error = "Cannot open adapter_config.json: " + config_path;
            return cfg;
        }

        std::string json_str((std::istreambuf_iterator<char>(f)),
                              std::istreambuf_iterator<char>());
        f.close();

        // Simple JSON parsing for the fields we need
        auto extract_int = [&](const std::string& key, int default_val) -> int {
            std::string search = "\"" + key + "\"";
            size_t pos = json_str.find(search);
            if (pos == std::string::npos) return default_val;
            pos = json_str.find(':', pos + search.size());
            if (pos == std::string::npos) return default_val;
            // Skip whitespace and colon
            while (++pos < json_str.size() && (json_str[pos] == ' ' || json_str[pos] == '\t' || json_str[pos] == '\n' || json_str[pos] == '\r')) {}
            std::string num_str;
            while (pos < json_str.size() && (json_str[pos] == '-' || (json_str[pos] >= '0' && json_str[pos] <= '9') || json_str[pos] == '.')) {
                num_str += json_str[pos++];
            }
            return num_str.empty() ? default_val : static_cast<int>(std::stof(num_str));
        };

        auto extract_bool = [&](const std::string& key, bool default_val) -> bool {
            std::string search = "\"" + key + "\"";
            size_t pos = json_str.find(search);
            if (pos == std::string::npos) return default_val;
            pos = json_str.find(':', pos + search.size());
            if (pos == std::string::npos) return default_val;
            while (++pos < json_str.size() && (json_str[pos] == ' ' || json_str[pos] == '\t' || json_str[pos] == '\n' || json_str[pos] == '\r')) {}
            if (json_str.compare(pos, 4, "true") == 0) return true;
            if (json_str.compare(pos, 5, "false") == 0) return false;
            return default_val;
        };

        cfg.r = extract_int("r", 16);
        cfg.lora_alpha = extract_int("lora_alpha", 32);

        bool use_rslora = extract_bool("use_rslora", false);
        if (use_rslora) {
            cfg.scaling = static_cast<float>(cfg.lora_alpha) / std::sqrt(static_cast<float>(cfg.r));
        } else {
            cfg.scaling = static_cast<float>(cfg.lora_alpha) / static_cast<float>(cfg.r);
        }

        // Extract target_modules array
        size_t tm_pos = json_str.find("\"target_modules\"");
        if (tm_pos != std::string::npos) {
            size_t bracket = json_str.find('[', tm_pos);
            size_t end_bracket = json_str.find(']', bracket);
            if (bracket != std::string::npos && end_bracket != std::string::npos) {
                std::string inner = json_str.substr(bracket + 1, end_bracket - bracket - 1);
                size_t p = 0;
                while (p < inner.size()) {
                    size_t q_start = inner.find('"', p);
                    if (q_start == std::string::npos) break;
                    size_t q_end = inner.find('"', q_start + 1);
                    if (q_end == std::string::npos) break;
                    cfg.target_modules.push_back(inner.substr(q_start + 1, q_end - q_start - 1));
                    p = q_end + 1;
                }
            }
        }

        return cfg;
    }
};

} // namespace lora
} // namespace aila
