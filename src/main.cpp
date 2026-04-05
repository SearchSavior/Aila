#include "engine/Engine.hpp"
#include "cli/CLI.hpp"
#include "profile/Device.hpp"
#include "profile/Profiling.hpp"
#include <iostream>

int main(int argc, char** argv) {
    // Parse command line arguments
    CLIOptions opts;
    if (!parse_cli_args(argc, argv, opts)) {
        return 1;
    }

    if (opts.show_help) {
        print_help();
        return 0;
    }

    if (opts.show_version) {
        print_version();
        return 0;
    }

    // Validate model directory
    if (opts.model_dir.empty()) {
        std::cerr << "Error: Model directory is required." << std::endl;
        std::cerr << "Use -m <path> or set AILA_MODEL_DIR environment variable." << std::endl;
        std::cerr << "Use --help for usage information." << std::endl;
        return 1;
    }

    // Check GPU device
    CheckDevice();
    AILA_LOG_INFO("[Config] max_seq_len=%d", opts.max_seq_len);

    // Initialize engine
    InferenceEngine engine;
    if (!engine.init(opts.model_dir, opts.max_seq_len)) {
        AILA_LOG_ERROR("Failed to initialize inference engine");
        return 1;
    }

    // Build generation config from CLI options
    GenerationConfig gen_config;
    gen_config.max_new_tokens   = opts.max_new_tokens;
    gen_config.temperature      = opts.temperature;
    gen_config.top_k            = opts.top_k;
    gen_config.do_sample        = opts.do_sample;
    gen_config.decode_chunk_size = opts.decode_chunk_size;
    gen_config.stream_chunk_size = opts.stream_chunk_size;

    // Run interactive loop
    return run_interactive(engine, gen_config, opts.stream_output);
}
