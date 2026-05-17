#pragma once

#include "../core/Context.hpp"
#include "../core/Tensor.hpp"

// Simple SYCL conv2d + GELU for audio encoder frontend.
// kernel=3, stride=2, pad=1 (same padding for odd input).

namespace ops {

// input:  [N, in_ch, in_h, in_w] bf16
// weight: [out_ch, in_ch, 3, 3] bf16 (OIHW)
// bias:   [out_ch] f32 or bf16
// output: [N, out_ch, out_h, out_w] bf16
// out_h = (in_h + 2*1 - 3) / 2 + 1 = ceil(in_h/2)
// out_w = (in_w + 2*1 - 3) / 2 + 1 = ceil(in_w/2)
void conv2d_gelu(Context& ctx,
                 Tensor& input, Tensor& weight, Tensor& bias,
                 Tensor& output,
                 int batch, int in_ch, int out_ch,
                 int in_h, int in_w, int out_h, int out_w);

} // namespace ops
