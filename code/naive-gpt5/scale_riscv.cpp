// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "scale_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector
#include "riscv_usability.h"

namespace ncnn {

Scale_riscv::Scale_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector
}

int Scale_riscv::forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& opt) const
{
    Mat& bottom_top_blob = bottom_top_blobs[0];
    const Mat& scale_blob = bottom_top_blobs[1];

    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int d = bottom_top_blob.d;
    const int channels = bottom_top_blob.c;
    const int dims = bottom_top_blob.dims;

    const int elempack = bottom_top_blob.elempack;

    const float* scale = scale_blob;
    const float* bias = bias_data;

    if (dims == 1)
    {
        float* ptr = (float*)bottom_top_blob;
        const int size = w * elempack;

        if (bias_term)
        {
#if __riscv_vector
            int n = size;
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e32m8(n);
                vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                vfloat32m8_t _s = __riscv_vle32_v_f32m8(scale, vl);
                vfloat32m8_t _b = __riscv_vle32_v_f32m8(bias, vl);
                vfloat32m8_t _ps = __riscv_vfmul_vv_f32m8(_p, _s, vl);
                _ps = __riscv_vfadd_vv_f32m8(_ps, _b, vl);
                __riscv_vse32_v_f32m8(ptr, _ps, vl);
                ptr += vl;
                scale += vl;
                bias += vl;
                n -= vl;
            }
#else
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < size; i++)
            {
                ptr[i] = ptr[i] * scale[i] + bias[i];
            }
#endif
        }
        else
        {
#if __riscv_vector
            int n = size;
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e32m8(n);
                vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                vfloat32m8_t _s = __riscv_vle32_v_f32m8(scale, vl);
                __riscv_vse32_v_f32m8(ptr, __riscv_vfmul_vv_f32m8(_p, _s, vl), vl);
                ptr += vl;
                scale += vl;
                n -= vl;
            }
#else
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < size; i++)
            {
                ptr[i] = ptr[i] * scale[i];
            }
#endif
        }
    }

    if (dims == 2)
    {
        const int size = w * elempack;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            float* ptr = bottom_top_blob.row(i);
            if (elempack == 1)
            {
                float s = scale[i];
                if (bias_term)
                {
                    float b = bias[i];
#if __riscv_vector
                    int n = size;
                    while (n > 0)
                    {
                        size_t vl = __riscv_vsetvl_e32m8(n);
                        vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                        vfloat32m8_t _s = __riscv_vfmv_v_f_f32m8(s, vl);
                        vfloat32m8_t _b = __riscv_vfmv_v_f_f32m8(b, vl);
                        vfloat32m8_t _ps = __riscv_vfmul_vv_f32m8(_p, _s, vl);
                        _ps = __riscv_vfadd_vv_f32m8(_ps, _b, vl);
                        __riscv_vse32_v_f32m8(ptr, _ps, vl);
                        ptr += vl;
                        n -= vl;
                    }
#else
                    for (int j = 0; j < size; j++)
                    {
                        ptr[j] = ptr[j] * s + b;
                    }
#endif
                }
                else
                {
#if __riscv_vector
                    int n = size;
                    while (n > 0)
                    {
                        size_t vl = __riscv_vsetvl_e32m8(n);
                        vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                        vfloat32m8_t _s = __riscv_vfmv_v_f_f32m8(s, vl);
                        __riscv_vse32_v_f32m8(ptr, __riscv_vfmul_vv_f32m8(_p, _s, vl), vl);
                        ptr += vl;
                        n -= vl;
                    }
#else
                    for (int j = 0; j < size; j++)
                    {
                        ptr[j] = ptr[j] * s;
                    }
#endif
                }
            }
            else
            {
                // elempack > 1, per-pack scale/bias
#if __riscv_vector
                size_t vl_pack = __riscv_vsetvl_e32m8(elempack);
                vfloat32m8_t _s_pack = __riscv_vle32_v_f32m8(scale + i * elempack, vl_pack);
                vfloat32m8_t _b_pack;
                bool has_bias = bias_term;
                if (has_bias)
                    _b_pack = __riscv_vle32_v_f32m8(bias + i * elempack, vl_pack);
                for (int j = 0; j < w; j++)
                {
                    vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl_pack);
                    vfloat32m8_t _ps = __riscv_vfmul_vv_f32m8(_p, _s_pack, vl_pack);
                    if (has_bias)
                        _ps = __riscv_vfadd_vv_f32m8(_ps, _b_pack, vl_pack);
                    __riscv_vse32_v_f32m8(ptr, _ps, vl_pack);
                    ptr += elempack;
                }
#else
                float s0 = scale[i * elempack + 0];
                float b0 = bias_term ? bias[i * elempack + 0] : 0.f;
                for (int j = 0; j < w * elempack; j += elempack)
                {
                    for (int k = 0; k < elempack; k++)
                    {
                        float s = scale[i * elempack + k];
                        float b = bias_term ? bias[i * elempack + k] : 0.f;
                        ptr[j + k] = ptr[j + k] * s + b;
                    }
                }
#endif
            }
        }
    }

    if (dims == 3 || dims == 4)
    {
        const int size = w * h * d * elempack;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);
            if (elempack == 1)
            {
                float s = scale[q];
                if (bias_term)
                {
                    float b = bias[q];
#if __riscv_vector
                    int n = size;
                    while (n > 0)
                    {
                        size_t vl = __riscv_vsetvl_e32m8(n);
                        vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                        vfloat32m8_t _s = __riscv_vfmv_v_f_f32m8(s, vl);
                        vfloat32m8_t _b = __riscv_vfmv_v_f_f32m8(b, vl);
                        vfloat32m8_t _ps = __riscv_vfmul_vv_f32m8(_p, _s, vl);
                        _ps = __riscv_vfadd_vv_f32m8(_ps, _b, vl);
                        __riscv_vse32_v_f32m8(ptr, _ps, vl);
                        ptr += vl;
                        n -= vl;
                    }
#else
                    for (int i = 0; i < size; i++)
                    {
                        ptr[i] = ptr[i] * s + b;
                    }
#endif
                }
                else
                {
#if __riscv_vector
                    int n = size;
                    while (n > 0)
                    {
                        size_t vl = __riscv_vsetvl_e32m8(n);
                        vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                        vfloat32m8_t _s = __riscv_vfmv_v_f_f32m8(s, vl);
                        __riscv_vse32_v_f32m8(ptr, __riscv_vfmul_vv_f32m8(_p, _s, vl), vl);
                        ptr += vl;
                        n -= vl;
                    }
#else
                    for (int i = 0; i < size; i++)
                    {
                        ptr[i] = ptr[i] * s;
                    }
#endif
                }
            }
            else
            {
#if __riscv_vector
                size_t vl_pack = __riscv_vsetvl_e32m8(elempack);
                vfloat32m8_t _s_pack = __riscv_vle32_v_f32m8(scale + q * elempack, vl_pack);
                vfloat32m8_t _b_pack;
                bool has_bias = bias_term;
                if (has_bias)
                    _b_pack = __riscv_vle32_v_f32m8(bias + q * elempack, vl_pack);
                int iterations = size / elempack;
                for (int i = 0; i < iterations; i++)
                {
                    vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl_pack);
                    vfloat32m8_t _ps = __riscv_vfmul_vv_f32m8(_p, _s_pack, vl_pack);
                    if (has_bias)
                        _ps = __riscv_vfadd_vv_f32m8(_ps, _b_pack, vl_pack);
                    __riscv_vse32_v_f32m8(ptr, _ps, vl_pack);
                    ptr += elempack;
                }
#else
                for (int i = 0; i < size; i += elempack)
                {
                    for (int k = 0; k < elempack; k++)
                    {
                        float s = scale[q * elempack + k];
                        float b = bias_term ? bias[q * elempack + k] : 0.f;
                        ptr[i + k] = ptr[i + k] * s + b;
                    }
                }
#endif
            }
        }
    }

    return 0;
}

} // namespace ncnn
