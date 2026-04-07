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

#include "layer_type.h"
#include "fused_activation.h"

#if __riscv_vector
#include <riscv_vector.h>
#include "riscv_activation.h"
#endif // __riscv_vector

#include "cpu.h"

namespace ncnn {

InnerProduct_riscv::InnerProduct_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector
    flatten = 0;
}

int InnerProduct_riscv::create_pipeline(const Option& opt)
{
    // always prepare flatten for non-1d input
    {
        flatten = ncnn::create_layer_cpu(ncnn::LayerType::Flatten);
        ncnn::ParamDict pd;
        flatten->load_param(pd);
        flatten->create_pipeline(opt);
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
#if NCNN_INT8
    if (opt.use_int8_inference && weight_data.elemsize == (size_t)1u)
    {
        // fall back to baseline int8 path implemented in base InnerProduct
        return InnerProduct::forward_int8(bottom_blob, top_blob, opt);
    }
#endif

    const int num_input = weight_data_size / num_output;

    if (bottom_blob.dims == 2 && bottom_blob.w == num_input)
    {
        // gemm path compatible with tests using dims=2
        int h = bottom_blob.h;
        size_t elemsize = bottom_blob.elemsize;
        int elempack = bottom_blob.elempack;

        top_blob.create(num_output, h, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        // compute out = a * W^T + b
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int j = 0; j < h; j++)
        {
            const float* m = bottom_blob.row(j);
            float* outptr = top_blob.row(j);

            for (int p = 0; p < num_output; p++)
            {
                const float* kptr = (const float*)weight_data + num_input * p;
                float sum = 0.f;
                if (bias_term)
                    sum = bias_data[p];

#if __riscv_vector
                int n = num_input * elempack;
                const float* mp = m;
                const float* kp = kptr;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m8(n);
                    vfloat32m8_t _m = __riscv_vle32_v_f32m8(mp, vl);
                    vfloat32m8_t _k = __riscv_vle32_v_f32m8(kp, vl);
                    vfloat32m8_t _prod = __riscv_vfmul_vv_f32m8(_m, _k, vl);
                    sum += __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredsum_vs_f32m8_f32m1(__riscv_vfmv_v_f_f32m1(0.f, vl), _prod, __riscv_vfmv_v_f_f32m1(0.f, vl), vl));
                    mp += vl;
                    kp += vl;
                    n -= vl;
                }
#else
                for (int i = 0; i < num_input * elempack; i++)
                {
                    sum += m[i] * kptr[i];
                }
#endif
                outptr[p] = activation_ss(sum, activation_type, activation_params);
            }
        }

        return 0;
    }

    // flatten if needed then do 1d matmul
    Mat bottom_blob_flattened = bottom_blob;
    if (bottom_blob.dims != 1)
    {
        Option opt_flatten = opt;
        opt_flatten.blob_allocator = opt.workspace_allocator;
        flatten->forward(bottom_blob, bottom_blob_flattened, opt_flatten);
        if (bottom_blob_flattened.empty())
            return -100;
    }

    size_t elemsize = bottom_blob_flattened.elemsize;
    int elempack = bottom_blob_flattened.elempack;

    top_blob.create(num_output, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // num_output
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < num_output; p++)
    {
        float sum = 0.f;
        if (bias_term)
            sum = bias_data[p];

        const float* wptr = (const float*)weight_data + (size_t)num_input * p;
        const float* mptr = bottom_blob_flattened;

#if __riscv_vector
        int n = num_input * elempack;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(n);
            vfloat32m8_t _w = __riscv_vle32_v_f32m8(wptr, vl);
            vfloat32m8_t _m = __riscv_vle32_v_f32m8(mptr, vl);
            vfloat32m8_t _prod = __riscv_vfmul_vv_f32m8(_w, _m, vl);
            sum += __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredsum_vs_f32m8_f32m1(__riscv_vfmv_v_f_f32m1(0.f, vl), _prod, __riscv_vfmv_v_f_f32m1(0.f, vl), vl));
            wptr += vl;
            mptr += vl;
            n -= vl;
        }
#else
        for (int i = 0; i < num_input * elempack; i++)
        {
            sum += mptr[i] * wptr[i];
        }
#endif
        top_blob[p] = activation_ss(sum, activation_type, activation_params);
    }

    return 0;
}

} // namespace ncnn
