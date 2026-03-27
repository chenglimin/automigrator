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

#ifndef LAYER_RISCV_CONVOLUTION1D_PACKED_H
#define LAYER_RISCV_CONVOLUTION1D_PACKED_H

#include "mat.h"
#include "option.h"
#include "riscv_activation.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

namespace ncnn {

// Transform kernel to packn-friendly layout
// src layout: [outh][inh][kw] (flattened to Mat with w=kw*inh*outh)
// dst layout: channel-major blocks over outh with elempack=packn; each row stores packn weights for given (q,k)
static inline void convolution1d_transform_kernel_packed(const Mat& kernel, Mat& kernel_tm, int inh, int outh, int kernel_w)
{
#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
    int out_elempack = outh % packn == 0 ? packn : 1;
    if (out_elempack == 1)
    {
        kernel_tm = kernel;
        return;
    }

    // reshape src to [kw, inh, outh]
    Mat kernel_r = kernel.reshape(kernel_w, inh, outh);

    // dst: w = kw*inh, h = outh/packn, elempack = packn
    kernel_tm.create(1, kernel_w * inh, outh / out_elempack, (size_t)4u * out_elempack, out_elempack);

    for (int ob = 0; ob < outh / out_elempack; ob++)
    {
        Mat gch = kernel_tm.channel(ob);
        for (int q = 0; q < inh; q++)
        {
            for (int k = 0; k < kernel_w; k++)
            {
                float* g = gch.row(q * kernel_w + k);
                for (int j = 0; j < out_elempack; j++)
                {
                    const float* src = kernel_r.channel(ob * out_elempack + j).row(q);
                    g[j] = src[k];
                }
            }
        }
    }
#else
    kernel_tm = kernel;
#endif // __riscv_vector
}

// RVV forward with output elempack=packn; input may be pack1
static inline void convolution1d_packed(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_tm, const Mat& bias_data, int kernel_w, int dilation_w, int stride_w, int activation_type, const Mat& activation_params, const Option& opt)
{
    const int outw = top_blob.w;
    const int out_elempack = top_blob.elempack;
    const int outh = top_blob.h * out_elempack;

    const int inh = bottom_blob.h * bottom_blob.elempack;
    const float* bias_data_ptr = bias_data;

#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
    (void)packn;

    const int out_blocks = outh / out_elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int ob = 0; ob < out_blocks; ob++)
    {
        const float* biasptr = 0;
        if (bias_data_ptr)
            biasptr = bias_data_ptr + ob * out_elempack;

        Mat wch = weight_data_tm.channel(ob);

        for (int j = 0; j < outw; j++)
        {
            size_t vl = __riscv_vsetvl_e32m1(out_elempack);
            vfloat32m1_t _sum = __riscv_vfmv_v_f_f32m1(0.f, vl);
            if (biasptr)
                _sum = __riscv_vle32_v_f32m1(biasptr, vl);

            // accumulate over inh and kw
            for (int q = 0; q < inh; q++)
            {
                const int q_c = q / bottom_blob.elempack;
                const int q_lane = q % bottom_blob.elempack;
                const float* sptr_row = bottom_blob.row(q_c);
                const float* sptr = sptr_row + j * stride_w * bottom_blob.elempack + q_lane;

                for (int k = 0; k < kernel_w; k++)
                {
                    const float val = sptr[0];
                    const float* wptr = wch.row(q * kernel_w + k);
                    vfloat32m1_t _w = __riscv_vle32_v_f32m1(wptr, vl);
                    _sum = __riscv_vfmacc_vf_f32m1(_sum, val, _w, vl);
                    sptr += dilation_w * bottom_blob.elempack;
                }
            }

            _sum = activation_ps(_sum, activation_type, activation_params, vl);
            float* outptr = top_blob.row(ob);
            __riscv_vse32_v_f32m1(outptr + (size_t)j * out_elempack, _sum, vl);
        }
    }
#else
    // fallback generic pack1
    const int bias_term = bias_data.empty() ? 0 : 1;
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outh; p++)
    {
        float* outptr = top_blob.row(p);
        for (int j = 0; j < outw; j++)
        {
            float sum = bias_term ? bias_data_ptr[p] : 0.f;
            const float* kptr = (const float*)weight_data_tm + kernel_w * bottom_blob.h * p;
            for (int q = 0; q < bottom_blob.h; q++)
            {
                const float* sptr = bottom_blob.row(q) + j * stride_w;
                for (int k = 0; k < kernel_w; k++)
                {
                    sum += sptr[0] * kptr[k];
                    sptr += dilation_w;
                }
                kptr += kernel_w;
            }
            sum = activation_ss(sum, activation_type, activation_params);
            outptr[j] = sum;
        }
    }
#endif // __riscv_vector
}

} // namespace ncnn

#endif // LAYER_RISCV_CONVOLUTION1D_PACKED_H
