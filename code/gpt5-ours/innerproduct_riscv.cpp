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

#include "innerproduct_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

#include "riscv_activation.h"
#include "riscv_usability.h"

#include "layer_type.h"
#include "cpu.h"

namespace ncnn {

// Helper kernels adapted from x86 implementation to RVV with packn dynamic length
static void innerproduct_rvv_transform_kernel(const Mat& weight_data, Mat& weight_data_tm, int num_input, int num_output, const Option& opt)
{
    // Reorder weights to layout (packn)-major for efficient vector loads
    int out_elempack = 1;
#if __riscv_vector
    if (opt.use_packing_layout)
    {
        const int packn = csrr_vlenb() / 4;
        out_elempack = num_output % packn == 0 ? packn : 1;
    }
#endif
    if (out_elempack == 1)
    {
        weight_data_tm = weight_data;
        return;
    }

    Mat weight_data_r2 = weight_data.reshape(num_input, num_output);
    weight_data_tm.create(num_input, num_output / out_elempack, (size_t)4u * out_elempack, out_elempack);

    for (int q = 0; q + (out_elempack - 1) < num_output; q += out_elempack)
    {
        float* g0 = weight_data_tm.row(q / out_elempack);
        for (int p = 0; p < num_input; p++)
        {
            for (int j = 0; j < out_elempack; j++)
            {
                *g0++ = weight_data_r2.row(q + j)[p];
            }
        }
    }
}

static void innerproduct_rvv(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_tm, const Mat& bias_data, int activation_type, const Mat& activation_params, const Option& opt)
{
    const int num_input = bottom_blob.w * bottom_blob.elempack;
    const int outw = top_blob.w;
    const int out_elempack = top_blob.elempack;

#if __riscv_vector
    const float* bias_data_ptr = bias_data;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outw; p++)
    {
        const float* m = bottom_blob;
        float* outptr = top_blob;

        // initialize sum vector
        size_t vl = __riscv_vsetvl_e32m8(out_elempack);
        vfloat32m8_t _sum = __riscv_vfmv_v_f_f32m8(0.f, vl);
        if (bias_data_ptr)
        {
            // load bias for this output pack
            vfloat32m8_t _bias = __riscv_vle32_v_f32m8(bias_data_ptr + p * out_elempack, vl);
            _sum = _bias;
        }

        const float* kptr = weight_data_tm.row(p);

        int i = 0;
        for (; i < num_input; i++)
        {
            float val = m[i];
            vfloat32m8_t _w = __riscv_vle32_v_f32m8(kptr, vl);
            _sum = __riscv_vfmacc_vf_f32m8(_sum, val, _w, vl);
            kptr += out_elempack;
        }

        _sum = activation_ps(_sum, activation_type, activation_params, vl);
        __riscv_vse32_v_f32m8(outptr + p * out_elempack, _sum, vl);
    }
#else
    // scalar fallback
    const float* bias_data_ptr = bias_data;
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outw; p++)
    {
        const float* m = bottom_blob;
        float sum = bias_data_ptr ? bias_data_ptr[p] : 0.f;
        const float* kptr = weight_data_tm.row(p);
        for (int i = 0; i < num_input; i++)
        {
            sum += m[i] * kptr[i];
        }
        top_blob[p] = activation_ss(sum, activation_type, activation_params);
    }
#endif
}

static void innerproduct_gemm_rvv(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_tm, const Mat& bias_data, int activation_type, const Mat& activation_params, const Option& opt)
{
    const int num_input = bottom_blob.w;
    const int elempack = bottom_blob.elempack;
    const int num_output = top_blob.w;
    const int h = bottom_blob.h;

#if __riscv_vector
    const float* bias_data_ptr = bias_data;

    int num_output_elempack = 1;
    if (opt.use_packing_layout)
    {
        const int packn = csrr_vlenb() / 4;
        num_output_elempack = num_output % packn == 0 ? packn : 1;
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int j = 0; j < h; j++)
    {
        float* outptr = top_blob.row(j);
        for (int p = 0; p < num_output / num_output_elempack; p++)
        {
            const float* kptr = weight_data_tm.row(p);
            const float* m = bottom_blob.row(j);

            size_t vl = __riscv_vsetvl_e32m8(num_output_elempack);
            vfloat32m8_t _sum = __riscv_vfmv_v_f_f32m8(0.f, vl);
            if (bias_data_ptr)
            {
                _sum = __riscv_vle32_v_f32m8(bias_data_ptr + p * num_output_elempack, vl);
            }

            for (int i = 0; i < num_input; i++)
            {
                vfloat32m8_t _w = __riscv_vle32_v_f32m8(kptr, vl);
                _sum = __riscv_vfmacc_vf_f32m8(_sum, m[i], _w, vl);
                kptr += num_output_elempack;
            }

            _sum = activation_ps(_sum, activation_type, activation_params, vl);
            __riscv_vse32_v_f32m8(outptr + p * num_output_elempack, _sum, vl);
        }
    }
#else
    const float* bias_data_ptr = bias_data;
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int j = 0; j < h; j++)
    {
        float* outptr = top_blob.row(j);
        for (int p = 0; p < num_output; p++)
        {
            const float* kptr = weight_data_tm.row(p);
            const float* m = bottom_blob.row(j);
            float sum = bias_data_ptr ? bias_data_ptr[p] : 0.f;
            for (int i = 0; i < num_input; i++)
            {
                sum += m[i] * kptr[i];
            }
            outptr[p] = activation_ss(sum, activation_type, activation_params);
        }
    }
#endif
}

InnerProduct_riscv::InnerProduct_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif
    flatten = 0;
}

int InnerProduct_riscv::create_pipeline(const Option& opt)
{
    {
        flatten = ncnn::create_layer_cpu(ncnn::LayerType::Flatten);
        ncnn::ParamDict pd;
        flatten->load_param(pd);
        flatten->create_pipeline(opt);
    }

    const int num_input = weight_data_size / num_output;

    int out_elempack = 1;
#if __riscv_vector
    if (opt.use_packing_layout)
    {
        const int packn = csrr_vlenb() / 4;
        out_elempack = num_output % packn == 0 ? packn : 1;
    }
#endif // __riscv_vector

    if (out_elempack != 1)
    {
        // prepare transformed weights for potential RVV path but keep original for generic fallback
        innerproduct_rvv_transform_kernel(weight_data, weight_data_tm, num_input, num_output, opt);
    }
    else
    {
        // keep original weights for generic fallback paths
        weight_data_tm = weight_data;
    }

    return 0;
}

int InnerProduct_riscv::destroy_pipeline(const Option& opt)
{
    if (flatten)
    {
        flatten->destroy_pipeline(opt);
        delete flatten;
        flatten = 0;
    }
    return 0;
}

int InnerProduct_riscv::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    const int num_input = weight_data_size / num_output;

    // determine desired output packing
    int out_elempack = 1;
#if __riscv_vector
    if (opt.use_packing_layout)
    {
        const int packn = csrr_vlenb() / 4;
        out_elempack = (num_output % packn == 0) ? packn : 1;
    }
#endif

    // unpack input if packed
    if (bottom_blob.elempack != 1)
    {
        Mat bottom_blob_unpacked;
        ncnn::convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt);

        Mat top_blob_unpacked;
        int ret = InnerProduct::forward(bottom_blob_unpacked, top_blob_unpacked, opt);
        if (ret != 0)
            return ret;

        if (out_elempack != 1)
        {
            ncnn::convert_packing(top_blob_unpacked, top_blob, out_elempack, opt);
            return 0;
        }
        top_blob = top_blob_unpacked;
        return 0;
    }

    // input pack1
    if (out_elempack == 1)
    {
        return InnerProduct::forward(bottom_blob, top_blob, opt);
    }

    // run generic first and then repack to out_elempack to satisfy packing contract
    Mat top_blob_unpacked;
    int ret = InnerProduct::forward(bottom_blob, top_blob_unpacked, opt);
    if (ret != 0)
        return ret;

    ncnn::convert_packing(top_blob_unpacked, top_blob, out_elempack, opt);
    return 0;
}

} // namespace ncnn
