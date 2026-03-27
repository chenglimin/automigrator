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

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

#include "riscv_activation.h"
#include "riscv_usability.h"

#include "cpu.h"

namespace ncnn {

InnerProduct_riscv::InnerProduct_riscv()
{
#if __riscv_vector
    support_packing = true;
#if NCNN_ZFH
    support_fp16_storage = cpu_support_riscv_zvfh();
#endif
#endif // __riscv_vector

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

#if NCNN_INT8
    if (opt.use_int8_inference && weight_data.elemsize == (size_t)1u)
    {
        return create_pipeline_int8_riscv(opt);
    }
#endif

#if __riscv_vector
    if (opt.use_fp16_storage && support_fp16_storage)
    {
        return create_pipeline_fp16s(opt);
    }
#endif

    const int num_input = weight_data_size / num_output;

    int out_elempack = 1;
#if __riscv_vector
    if (opt.use_packing_layout)
    {
        const int packn = csrr_vlenb() / 4;
        out_elempack = num_output % packn == 0 ? packn : num_output % 4 == 0 ? 4 : 1;
    }
#endif // __riscv_vector

    if (out_elempack > 1)
    {
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
    else
    {
        weight_data_tm = weight_data;
    }

    if (opt.lightmode)
        weight_data.release();

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
    if (opt.use_int8_inference && int8_scale_term)
    {
        return forward_int8_riscv(bottom_blob, top_blob, opt);
    }
#endif

#if __riscv_vector
    if (support_fp16_storage && opt.use_fp16_storage)
    {
        return forward_fp16s(bottom_blob, top_blob, opt);
    }
#endif

    const int num_input = weight_data_size / num_output;

    if (bottom_blob.dims == 2 && bottom_blob.w == num_input)
    {
        int h = bottom_blob.h;
        size_t elemsize = bottom_blob.elemsize;
        int elempack = bottom_blob.elempack;

        top_blob.create(num_output, h, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        int num_output_elempack = 1;
#if __riscv_vector
        if (opt.use_packing_layout)
        {
            const int packn = csrr_vlenb() / 4;
            num_output_elempack = num_output % packn == 0 ? packn : num_output % 4 == 0 ? 4 : 1;
        }
#endif

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int j = 0; j < h; j++)
        {
#if __riscv_vector
            if (elempack == 4 && num_output_elempack == 4)
            {
                float* outptr = top_blob.row(j);

                for (int p = 0; p < num_output / num_output_elempack; p++)
                {
                    const float* kptr = weight_data_tm.row(p);
                    const float* m = bottom_blob.row(j);

                    vfloat32m1_t _sum0 = __riscv_vfmv_v_f_f32m1(0.f, __riscv_vsetvl_e32m1(4));
                    vfloat32m1_t _sum1 = __riscv_vfmv_v_f_f32m1(0.f, __riscv_vsetvl_e32m1(4));
                    vfloat32m1_t _sum2 = __riscv_vfmv_v_f_f32m1(0.f, __riscv_vsetvl_e32m1(4));
                    vfloat32m1_t _sum3 = __riscv_vfmv_v_f_f32m1(0.f, __riscv_vsetvl_e32m1(4));

                    if (bias_term)
                    {
                        _sum0 = __riscv_vle32_v_f32m1((const float*)bias_data + p * 4, __riscv_vsetvl_e32m1(4));
                    }

                    int i = 0;
                    for (; i < num_input; i++)
                    {
                        vfloat32m1_t _val = __riscv_vle32_v_f32m1(m, __riscv_vsetvl_e32m1(4));
                        vfloat32m1_t _w = __riscv_vle32_v_f32m1(kptr, __riscv_vsetvl_e32m1(4));
                        _sum0 = __riscv_vfmacc_vv_f32m1(_sum0, _val, __riscv_vfmv_v_f_f32m1(__riscv_vfmv_f_s_f32m1_f32(_w), __riscv_vsetvl_e32m1(4)), __riscv_vsetvl_e32m1(4));
                        _sum1 = __riscv_vfmacc_vv_f32m1(_sum1, _val, __riscv_vfmv_v_f_f32m1(__riscv_vfmv_f_s_f32m1_f32(_w), __riscv_vsetvl_e32m1(4)), __riscv_vsetvl_e32m1(4));
                        _sum2 = __riscv_vfmacc_vv_f32m1(_sum2, _val, __riscv_vfmv_v_f_f32m1(__riscv_vfmv_f_s_f32m1_f32(_w), __riscv_vsetvl_e32m1(4)), __riscv_vsetvl_e32m1(4));
                        _sum3 = __riscv_vfmacc_vv_f32m1(_sum3, _val, __riscv_vfmv_v_f_f32m1(__riscv_vfmv_f_s_f32m1_f32(_w), __riscv_vsetvl_e32m1(4)), __riscv_vsetvl_e32m1(4));
                        m += 4;
                        kptr += 4;
                    }

                    _sum0 = activation_ps(_sum0, activation_type, activation_params, __riscv_vsetvl_e32m1(4));
                    _sum1 = activation_ps(_sum1, activation_type, activation_params, __riscv_vsetvl_e32m1(4));
                    _sum2 = activation_ps(_sum2, activation_type, activation_params, __riscv_vsetvl_e32m1(4));
                    _sum3 = activation_ps(_sum3, activation_type, activation_params, __riscv_vsetvl_e32m1(4));

                    __riscv_vse32_v_f32m1(outptr, _sum0, __riscv_vsetvl_e32m1(4));
                    __riscv_vse32_v_f32m1(outptr + 4, _sum1, __riscv_vsetvl_e32m1(4));
                    __riscv_vse32_v_f32m1(outptr + 8, _sum2, __riscv_vsetvl_e32m1(4));
                    __riscv_vse32_v_f32m1(outptr + 12, _sum3, __riscv_vsetvl_e32m1(4));
                    outptr += 16;
                }
            }
#endif // __riscv_vector

            if (elempack == 1 && num_output_elempack == 1)
            {
                float* outptr = top_blob.row(j);

                for (int p = 0; p < num_output; p++)
                {
                    const float* kptr = (const float*)weight_data_tm + num_input * p;
                    const float* m = bottom_blob.row(j);

                    float sum = 0.f;
                    if (bias_term)
                        sum = bias_data[p];

                    for (int i = 0; i < num_input; i++)
                    {
                        sum += m[i] * kptr[i];
                    }

                    outptr[p] = activation_ss(sum, activation_type, activation_params);
                }
            }
        }

        return 0;
    }

    // flatten
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

    int out_elempack = 1;
#if __riscv_vector
    if (opt.use_packing_layout)
    {
        const int packn = csrr_vlenb() / 4;
        out_elempack = num_output % packn == 0 ? packn : num_output % 4 == 0 ? 4 : 1;
    }
#endif // __riscv_vector
    size_t out_elemsize = elemsize / elempack * out_elempack;

    top_blob.create(num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

#if __riscv_vector
    if (out_elempack == (csrr_vlenb() / 4))
    {
        const int packn = csrr_vlenb() / 4;
        const size_t vl = __riscv_vsetvl_e32m1(packn);
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < num_output / out_elempack; p++)
        {
            vfloat32m1_t _sum = bias_term ? __riscv_vle32_v_f32m1((const float*)bias_data + p * packn, vl) : __riscv_vfmv_v_f_f32m1(0.f, vl);

            const float* kptr = weight_data_tm.row(p);
            const float* sptr = bottom_blob_flattened;

            for (int i = 0; i < num_input; i++)
            {
                vfloat32m1_t _w = __riscv_vle32_v_f32m1(kptr, vl);
                vfloat32m1_t _val = __riscv_vfmv_v_f_f32m1(sptr[0], vl);
                _sum = __riscv_vfmacc_vv_f32m1(_sum, _w, _val, vl);
                sptr += 1;
                kptr += packn;
            }

            _sum = activation_ps(_sum, activation_type, activation_params, vl);

            float* outptr = top_blob;
            __riscv_vse32_v_f32m1(outptr + p * packn, _sum, vl);
        }
    }
    else
#endif // __riscv_vector
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < num_output; p++)
        {
            const float* kptr = (const float*)weight_data_tm + num_input * p;
            const float* sptr = bottom_blob_flattened;

            float sum = 0.f;
            if (bias_term)
                sum = bias_data[p];

            for (int i = 0; i < num_input; i++)
            {
                sum += sptr[i] * kptr[i];
            }

            float* outptr = top_blob;
            outptr[p] = activation_ss(sum, activation_type, activation_params);
        }
    }

    return 0;
}

#if __riscv_vector
int InnerProduct_riscv::create_pipeline_fp16s(const Option& opt)
{
    const int num_input = weight_data_size / num_output;

    int out_elempack = 1;
    if (opt.use_packing_layout)
    {
        const int packn = csrr_vlenb() / 2;
        out_elempack = num_output % packn == 0 ? packn : num_output % 4 == 0 ? 4 : 1;
    }

    // reuse float pipeline transform for simplicity
    return create_pipeline(opt);
}

int InnerProduct_riscv::forward_fp16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // fallback to float compute on fp16 storage for now
    return forward(bottom_blob, top_blob, opt);
}
#endif // __riscv_vector

#if NCNN_INT8
int InnerProduct_riscv::create_pipeline_int8_riscv(const Option& opt)
{
    // no special handling needed
    return 0;
}

int InnerProduct_riscv::forward_int8_riscv(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // use base implementation
    return InnerProduct::forward_int8(bottom_blob, top_blob, opt);
}
#endif // NCNN_INT8

} // namespace ncnn
