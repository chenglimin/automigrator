// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2026 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "deformableconv2d_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

#include "riscv_activation.h"
#include "riscv_usability.h"

#include "benchmark.h"
#include "cpu.h"
#include "layer_type.h"

namespace ncnn {

DeformableConv2D_riscv::DeformableConv2D_riscv()
{
#if __riscv_vector
    support_packing = true;
#else
    support_packing = false;
#endif
}

int DeformableConv2D_riscv::create_pipeline(const Option& opt)
{
    if (opt.lightmode)
        weight_data.release();
    return 0;
}

int DeformableConv2D_riscv::destroy_pipeline(const Option& opt)
{
    return 0;
}

int DeformableConv2D_riscv::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob_base = bottom_blobs[0];
    const Mat& offset_base = bottom_blobs[1];
    const bool has_mask = (bottom_blobs.size() == 3);
    const Mat& mask_base = has_mask ? bottom_blobs[2] : Mat();
    Mat& top_blob = top_blobs[0];

    int elempack = bottom_blob_base.elempack;

    Mat bottom_blob, offset, mask;
    if (elempack != 1)
    {
        convert_packing(bottom_blob_base, bottom_blob, 1, opt);
    }
    else
    {
        bottom_blob = bottom_blob_base;
    }
    convert_packing(offset_base, offset, 1, opt);
    if (has_mask)
        convert_packing(mask_base, mask, 1, opt);

    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const size_t elemsize = bottom_blob.elemsize;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;
    const int outw = (w + pad_left + pad_right - kernel_extent_w) / stride_w + 1;
    const int outh = (h + pad_top + pad_bottom - kernel_extent_h) / stride_h + 1;

    top_blob.create(outw, outh, num_output, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    const float* weight_ptr = weight_data;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int h_col = 0; h_col < outh; h_col++)
    {
        for (int w_col = 0; w_col < outw; w_col++)
        {
            int h_in = h_col * stride_h - pad_top;
            int w_in = w_col * stride_w - pad_left;
            for (int oc = 0; oc < num_output; oc++)
            {
                float sum = bias_term ? bias_data[oc] : 0.f;
                for (int i = 0; i < kernel_h; i++)
                {
                    for (int j = 0; j < kernel_w; j++)
                    {
                        const float offset_h = offset.channel((i * kernel_w + j) * 2).row(h_col)[w_col];
                        const float offset_w = offset.channel((i * kernel_w + j) * 2 + 1).row(h_col)[w_col];
                        const float mask_ = has_mask ? mask.channel(i * kernel_w + j).row(h_col)[w_col] : 1.f;
                        const float h_im = h_in + i * dilation_h + offset_h;
                        const float w_im = w_in + j * dilation_w + offset_w;

                        const bool cond = h_im > -1 && w_im > -1 && h_im < h && w_im < w;
                        int h_low = 0;
                        int w_low = 0;
                        int h_high = 0;
                        int w_high = 0;
                        float w1 = 0.f, w2 = 0.f, w3 = 0.f, w4 = 0.f;
                        bool v1_cond = false, v2_cond = false, v3_cond = false, v4_cond = false;
                        if (cond)
                        {
                            h_low = (int)floorf(h_im);
                            w_low = (int)floorf(w_im);
                            h_high = h_low + 1;
                            w_high = w_low + 1;
                            float lh = h_im - h_low;
                            float lw = w_im - w_low;
                            float hh = 1 - lh;
                            float hw = 1 - lw;
                            v1_cond = (h_low >= 0 && w_low >= 0);
                            v2_cond = (h_low >= 0 && w_high <= w - 1);
                            v3_cond = (h_high <= h - 1 && w_low >= 0);
                            v4_cond = (h_high <= h - 1 && w_high <= w - 1);
                            w1 = hh * hw;
                            w2 = hh * lw;
                            w3 = lh * hw;
                            w4 = lh * lw;
                        }

#if __riscv_vector
                        // RVV向量化按通道聚合：每次处理packn通道
                        if (cond && opt.use_packing_layout)
                        {
                            const int packn = cpu_riscv_vlenb() / 4;
                            int ic = 0;
                            for (; ic + (packn - 1) < channels; ic += packn)
                            {
                                // 逐通道计算双线性插值的四邻域加权求和
                                float tmpacc_buf[64];
                                size_t vl = __riscv_vsetvl_e32m1(packn);
                                for (int t = 0; t < packn; t++)
                                {
                                    int ci = ic + t;
                                    float v1 = v1_cond ? bottom_blob.channel(ci).row(h_low)[w_low] : 0.f;
                                    float v2 = v2_cond ? bottom_blob.channel(ci).row(h_low)[w_high] : 0.f;
                                    float v3 = v3_cond ? bottom_blob.channel(ci).row(h_high)[w_low] : 0.f;
                                    float v4 = v4_cond ? bottom_blob.channel(ci).row(h_high)[w_high] : 0.f;
                                    tmpacc_buf[t] = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
                                }
                                vfloat32m1_t vacc = __riscv_vle32_v_f32m1(tmpacc_buf, vl);

                                // 权重向量
                                float wbuf[64];
                                for (int t = 0; t < packn; t++)
                                {
                                    int ci = ic + t;
                                    wbuf[t] = weight_ptr[((oc * channels + ci) * kernel_h + i) * kernel_w + j] * mask_;
                                }
                                vfloat32m1_t wv = __riscv_vle32_v_f32m1(wbuf, vl);
                                vfloat32m1_t prod = __riscv_vfmul_vv_f32m1(vacc, wv, vl);
                                float partial = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m1_f32m1(prod, __riscv_vfmv_s_f_f32m1(0.f, vl), vl));
                                sum += partial;
                            }
                            // 处理剩余通道（不足packn）
                            for (; ic < channels; ic++)
                            {
                                float val = 0.f;
                                if (cond)
                                {
                                    float v1 = v1_cond ? bottom_blob.channel(ic).row(h_low)[w_low] : 0.f;
                                    float v2 = v2_cond ? bottom_blob.channel(ic).row(h_low)[w_high] : 0.f;
                                    float v3 = v3_cond ? bottom_blob.channel(ic).row(h_high)[w_low] : 0.f;
                                    float v4 = v4_cond ? bottom_blob.channel(ic).row(h_high)[w_high] : 0.f;
                                    val = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
                                }
                                sum += val * mask_ * weight_ptr[((oc * channels + ic) * kernel_h + i) * kernel_w + j];
                            }
                        }
                        else
#endif
                        {
                            for (int ic = 0; ic < channels; ic++)
                            {
                                float val = 0.f;
                                if (cond)
                                {
                                    float v1 = v1_cond ? bottom_blob.channel(ic).row(h_low)[w_low] : 0.f;
                                    float v2 = v2_cond ? bottom_blob.channel(ic).row(h_low)[w_high] : 0.f;
                                    float v3 = v3_cond ? bottom_blob.channel(ic).row(h_high)[w_low] : 0.f;
                                    float v4 = v4_cond ? bottom_blob.channel(ic).row(h_high)[w_high] : 0.f;
                                    val = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
                                }
                                sum += val * mask_ * weight_ptr[((oc * channels + ic) * kernel_h + i) * kernel_w + j];
                            }
                        }
                    }
                }
                top_blob.channel(oc).row(h_col)[w_col] = activation_ss(sum, activation_type, activation_params);
            }
        }
    }

    return 0;
}

} // namespace ncnn
