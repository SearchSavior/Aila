#include "Ops.hpp"
#include <sycl/sycl.hpp>

using bf16 = sycl::ext::oneapi::bfloat16;

namespace ops {

// ============================================================
// SYCL Kernel: Physical Transpose (for weights)
// src: [rows, cols] -> dst: [cols, rows]
// ============================================================

void transpose(Context& ctx, Tensor& src, Tensor& dst) {
    int rows = static_cast<int>(src.shape(0));
    int cols = static_cast<int>(src.shape(1));
    bf16* s_ptr = static_cast<bf16*>(src.data());
    bf16* d_ptr = static_cast<bf16*>(dst.data());

    // 使用 16x16 tile 优化访存合并 (Memory Coalescing)
    // A770 的 L3 cache 很大，简单的 tile 就能跑出极高性能
    constexpr int TILE_DIM = 16;

    ctx.queue().submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<2>(
            sycl::range<2>((rows + TILE_DIM - 1) / TILE_DIM * TILE_DIM, 
                           (cols + TILE_DIM - 1) / TILE_DIM * TILE_DIM),
            sycl::range<2>(TILE_DIM, TILE_DIM)), 
            [=](sycl::nd_item<2> item) {
                int r = item.get_global_id(0);
                int c = item.get_global_id(1);

                if (r < rows && c < cols) {
                    // dst[c, r] = src[r, c]
                    d_ptr[c * rows + r] = s_ptr[r * cols + c];
                }
            });
    });
}

} // namespace ops
