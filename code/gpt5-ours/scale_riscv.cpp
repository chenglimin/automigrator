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

int Scale_riscv::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    std::vector<Mat> bottom_top_blobs(2);
    bottom_top_blobs[0] = bottom_top_blob;
    bottom_top_blobs[1] = scale_data;
    return forward_inplace(bottom_top_blobs, opt);
}


static inline void scale_mul_add_row(float* ptr, int size, float s, float b)
{
#if __riscv_vector
    int n = size;
    while (n > 0)
    {
        size_t vl = __riscv_vsetvl_e32m8(n);
        vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
        vfloat32m8_t _b = __riscv_vfmv_v_f_f32m8(b, vl);
        // p = p * s + b
        _p = __riscv_vfmadd_vf_f32m8(_p, s, _b, vl);
        __riscv_vse32_v_f32m8(ptr, _p, vl);
        ptr += vl;
        n -= vl;
    }
#else
    for (int j = 0; j < size; j++)
        ptr[j] = ptr[j] * s + b;
#endif
}

static inline void scale_mul_row(float* ptr, int size, float s)
{
#if __riscv_vector
    int n = size;
    while (n > 0)
    {
        size_t vl = __riscv_vsetvl_e32m8(n);
        vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
        _p = __riscv_vfmul_vf_f32m8(_p, s, vl);
        __riscv_vse32_v_f32m8(ptr, _p, vl);
        ptr += vl;
        n -= vl;
    }
#else
    for (int j = 0; j < size; j++)
        ptr[j] = ptr[j] * s;
#endif
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
                // p = p * s + b
                _p = __riscv_vfmadd_vv_f32m8(_p, _s, _b, vl);
                __riscv_vse32_v_f32m8(ptr, _p, vl);
                ptr += vl;
                scale += vl;
                bias += vl;
                n -= vl;
            }
#else
            for (int i = 0; i < size; i++)
                ptr[i] = ptr[i] * scale[i] + bias[i];
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
                _p = __riscv_vfmul_vv_f32m8(_p, _s, vl);
                __riscv_vse32_v_f32m8(ptr, _p, vl);
                ptr += vl;
                scale += vl;
                n -= vl;
            }
#else
            for (int i = 0; i < size; i++)
                ptr[i] = ptr[i] * scale[i];
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
            float s = scale[i];
#if __riscv_vector
            if (elempack > 1)
            {
                const int packn = csrr_vlenb() / 4;
                size_t vl_packn = __riscv_vsetvl_e32m8(packn);
                const float* row_scale = (const float*)scale + i * elempack;
                vfloat32m8_t _s = __riscv_vle32_v_f32m8(row_scale, vl_packn);
                if (bias_term)
                {
                    const float* row_bias = (const float*)bias + i * elempack;
                    vfloat32m8_t _b = __riscv_vle32_v_f32m8(row_bias, vl_packn);
                    int n = size;
                    while (n > 0)
                    {
                        vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl_packn);
                        _p = __riscv_vfmadd_vv_f32m8(_p, _s, _b, vl_packn);
                        __riscv_vse32_v_f32m8(ptr, _p, vl_packn);
                        ptr += vl_packn;
                        n -= vl_packn;
                    }
                }
                else
                {
                    int n = size;
                    while (n > 0)
                    {
                        vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl_packn);
                        _p = __riscv_vfmul_vv_f32m8(_p, _s, vl_packn);
                        __riscv_vse32_v_f32m8(ptr, _p, vl_packn);
                        ptr += vl_packn;
                        n -= vl_packn;
                    }
                }
            }
            else
#endif // __riscv_vector
            {
                if (bias_term)
                {
                    float b = bias[i];
                    scale_mul_add_row(ptr, size, s, b);
                }
                else
                {
                    scale_mul_row(ptr, size, s);
                }
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
            float s = scale[q];
#if __riscv_vector
            if (elempack > 1)
            {
                const int packn = csrr_vlenb() / 4;
                size_t vl_packn = __riscv_vsetvl_e32m8(packn);
                const float* ch_scale = (const float*)scale + q * elempack;
                vfloat32m8_t _s = __riscv_vle32_v_f32m8(ch_scale, vl_packn);
                if (bias_term)
                {
                    const float* ch_bias = (const float*)bias + q * elempack;
                    vfloat32m8_t _b = __riscv_vle32_v_f32m8(ch_bias, vl_packn);
                    int n = size;
                    while (n > 0)
                    {
                        vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl_packn);
                        _p = __riscv_vfmadd_vv_f32m8(_p, _s, _b, vl_packn);
                        __riscv_vse32_v_f32m8(ptr, _p, vl_packn);
                        ptr += vl_packn;
                        n -= vl_packn;
                    }
                }
                else
                {
                    int n = size;
                    while (n > 0)
                    {
                        vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl_packn);
                        _p = __riscv_vfmul_vv_f32m8(_p, _s, vl_packn);
                        __riscv_vse32_v_f32m8(ptr, _p, vl_packn);
                        ptr += vl_packn;
                        n -= vl_packn;
                    }
                }
            }
            else
#endif // __riscv_vector
            {
                if (bias_term)
                {
                    float b = bias[q];
                    scale_mul_add_row(ptr, size, s, b);
                }
                else
                {
                    scale_mul_row(ptr, size, s);
                }
            }
        }
    }

    return 0;
}

} // namespace ncnn
