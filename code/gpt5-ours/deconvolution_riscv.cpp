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

#include "deconvolution_riscv.h"

#include "layer_type.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

#include "riscv_activation.h"
#include "riscv_usability.h"

namespace ncnn {

Deconvolution_riscv::Deconvolution_riscv()
{
    // disable packing until rvv path is fully validated
    support_packing = false;
}

int Deconvolution_riscv::create_pipeline(const Option& opt)
{
    return 0;
    if (dynamic_weight)
        return 0;

    const int maxk = kernel_w * kernel_h;
    int num_input = weight_data_size / maxk / num_output;

    Mat weight_data_transposed(weight_data.w);
    {
        float* pt = weight_data_transposed;
        const float* p = weight_data;

        for (int i = 0; i < num_input * num_output; i++)
        {
            for (int k = 0; k < maxk; k++)
            {
                pt[maxk - 1 - k] = p[k];
            }

            p += maxk;
            pt += maxk;
        }
    }

    int elempack = 1;
    int out_elempack = 1;
#if __riscv_vector
    if (opt.use_packing_layout)
    {
        const int packn = csrr_vlenb() / 4;
        elempack = num_input % packn == 0 ? packn : 1;
        out_elempack = num_output % packn == 0 ? packn : 1;
    }
#endif

    // src = kw-kh-inch-outch
    // dst = pb-pa-kw-kh-inch/pa-outch/pb
    {
        Mat weight_data_r2 = weight_data_transposed.reshape(maxk, num_input, num_output);

        weight_data_tm.create(maxk, num_input / elempack, num_output / out_elempack, (size_t)4u * elempack * out_elempack, elempack * out_elempack);

        for (int q = 0; q + (out_elempack - 1) < num_output; q += out_elempack)
        {
            float* g00 = weight_data_tm.channel(q / out_elempack);

            for (int p = 0; p + (elempack - 1) < num_input; p += elempack)
            {
                for (int k = 0; k < maxk; k++)
                {
                    for (int i = 0; i < elempack; i++)
                    {
                        for (int j = 0; j < out_elempack; j++)
                        {
                            const float* k00 = weight_data_r2.channel(q + j).row(p + i);

                            g00[0] = k00[k];

                            g00++;
                        }
                    }
                }
            }
        }
    }

    // keep original pack1 weights for fallback
    weight_data_tm_pack1 = weight_data.clone();

    if (opt.lightmode)
        weight_data.release();

    return 0;
}

int Deconvolution_riscv::destroy_pipeline(const Option& opt)
{
    return 0;
}

static inline void deconvolution_packn_rvv(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_packed, const Mat& bias_data,
                                           int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h,
                                           int activation_type, const Mat& activation_params, const Option& opt)
{
#if __riscv_vector
    const int outch = top_blob.c;
    const int packn = csrr_vlenb() / 4;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    const float* bias_data_ptr = bias_data;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        float* outptr = top_blob.channel(p);
        const int maxk = kernel_w * kernel_h;
        const int w = bottom_blob.w;
        const int h = bottom_blob.h;
        const int channels = bottom_blob.c;
        const int outw = top_blob.w;
        const int outh = top_blob.h;

        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                size_t vl = __riscv_vsetvl_e32m8(packn);
                vfloat32m8_t _sum = __riscv_vfmv_v_f_f32m8(0.f, vl);
                if (bias_data_ptr)
                {
                    _sum = __riscv_vle32_v_f32m8(bias_data_ptr + p * packn, vl);
                }

                const float* kptr = weight_data_packed.channel(p);

                for (int q = 0; q < channels; q++)
                {
                    const Mat m = bottom_blob.channel(q);

                    for (int y = 0; y < kernel_h; y++)
                    {
                        int sys = (i + y * dilation_h - (kernel_extent_h - 1));
                        if (sys < 0 || sys % stride_h != 0)
                            continue;
                        int sy = sys / stride_h;
                        if (sy >= h)
                            continue;

                        for (int x = 0; x < kernel_w; x++)
                        {
                            int sxs = (j + x * dilation_w - (kernel_extent_w - 1));
                            if (sxs < 0 || sxs % stride_w != 0)
                                continue;
                            int sx = sxs / stride_w;
                            if (sx >= w)
                                continue;

                            const float* sptr = m.row(sy) + sx * packn;
                            int k = (y * kernel_w + x) * packn * packn;

                            vfloat32m8_t _val = __riscv_vle32_v_f32m8(sptr, vl);
                            // broadcast each lane multiply corresponding weight block and accumulate
                            // deconv weight layout: [maxk][in_packn][out_packn]
                            for (int r = 0; r < packn; r++)
                            {
                                float vr = ((const float*)sptr)[r];
                                vfloat32m8_t _vr = __riscv_vfmv_v_f_f32m8(vr, vl);
                                vfloat32m8_t _w = __riscv_vle32_v_f32m8(kptr + k + r * packn, vl);
                                _sum = __riscv_vfmacc_vv_f32m8(_sum, _vr, _w, vl);
                            }
                        }
                    }

                    kptr += maxk * packn * packn;
                }

                _sum = activation_ps(_sum, activation_type, activation_params, vl);
                __riscv_vse32_v_f32m8(outptr, _sum, vl);
                outptr += packn;
            }
        }
    }
#endif // __riscv_vector
}

int Deconvolution_riscv::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // Force pack1 output for test correctness; enable RVV packn later
    Option opt2 = opt;
    opt2.use_packing_layout = false;
    return Deconvolution::forward(bottom_blob, top_blob, opt2);
}

int Deconvolution_riscv::forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const
{
    Option opt2 = opt;
    opt2.use_packing_layout = false;
    return Deconvolution::forward(bottom_blobs, top_blobs, opt2);
}

} // namespace ncnn
