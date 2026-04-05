#include "Device.hpp"
#include "Profiling.hpp"
#include "utils/EnvUtils.hpp"
#include <string>

namespace {

const char* matrix_type_name(sycl::ext::oneapi::experimental::matrix::matrix_type t) {
    using matrix_type = sycl::ext::oneapi::experimental::matrix::matrix_type;
    switch (t) {
        case matrix_type::bf16: return "bf16";
        case matrix_type::fp16: return "fp16";
        case matrix_type::tf32: return "tf32";
        case matrix_type::fp32: return "fp32";
        case matrix_type::fp64: return "fp64";
        case matrix_type::sint8: return "s8";
        case matrix_type::sint16: return "s16";
        case matrix_type::sint32: return "s32";
        case matrix_type::sint64: return "s64";
        case matrix_type::uint8: return "u8";
        case matrix_type::uint16: return "u16";
        case matrix_type::uint32: return "u32";
        case matrix_type::uint64: return "u64";
        default: return "unknown";
    }
}

} // namespace

int CheckDevice()
{
    sycl::queue q(sycl::default_selector_v);
    auto dev = q.get_device();
    AILA_LOG_INFO("[Context] Device: %s",
                  dev.get_info<sycl::info::device::name>().c_str());
    AILA_LOG_INFO("[Context] Global memory: %llu MB",
                  (unsigned long long)(dev.get_info<sycl::info::device::global_mem_size>() / (1024*1024)));

    if (aila::env::read_int_raw("AILA_PRINT_MATRIX_COMBOS", 0) != 0) {
        using sycl::ext::oneapi::experimental::info::device::matrix_combinations;
        auto combos = dev.get_info<matrix_combinations>();
        AILA_LOG_INFO("[Context] matrix_combinations=%zu", combos.size());

        int shown = 0;
        const int show_max = 64;
        for (const auto& c : combos) {
            AILA_LOG_INFO("  [JM] m=%zu n=%zu k=%zu A=%s B=%s C=%s D=%s",
                          c.msize, c.nsize, c.ksize,
                          matrix_type_name(c.atype),
                          matrix_type_name(c.btype),
                          matrix_type_name(c.ctype),
                          matrix_type_name(c.dtype));
            shown++;
            if (shown >= show_max) {
                AILA_LOG_INFO("  [JM] ... truncated at %d entries", show_max);
                break;
            }
        }
    }

    return 0;
}
