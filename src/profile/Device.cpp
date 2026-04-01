#include "Device.hpp"
int CheckDevice()
{
    sycl::queue q(sycl::default_selector_v);
    auto dev = q.get_device();
    std::cout << "[Context] Device: " << dev.get_info<sycl::info::device::name>() << std::endl;
        std::cout << "[Context] Global memory: "
                  << dev.get_info<sycl::info::device::global_mem_size>() / (1024*1024)
                  << " MB" << std::endl;
    return 0;
}