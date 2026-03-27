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

#include "scale_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

#include "cpu.h"

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
        int n = w * elempack;

#if __riscv_vector
        if (bias_term)
        {
            int remain = n;
            while (remain > 0)
            {
                size_t vl = __riscv_vsetvl_e32m8(remain);
                vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                vfloat32m8_t _s = __riscv_vle32_v_f32m8(scale, vl);
                vfloat32m8_t _b = __riscv_vle32_v_f32m8(bias, vl);
                vfloat32m8_t _res = __riscv_vfmadd_vvf_f32m8(_p, _s, _b, vl);
                __riscv_vse32_v_f32m8(ptr, _res, vl);
                ptr += vl;
                scale += vl;
                bias += vl;
                remain -= vl;
            }
        }
        else
        {
            int remain = n;
            while (remain > 0)
            {
                size_t vl = __riscv_vsetvl_e32m8(remain);
                vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                vfloat32m8_t _s = __riscv_vle32_v_f32m8(scale, vl);
                vfloat32m8_t _res = __riscv_vfmul_vv_f32m8(_p, _s, vl);
                __riscv_vse32_v_f32m8(ptr, _res, vl);
                ptr += vl;
                scale += vl;
                remain -= vl;
            }
        }
#else
        if (bias_term)
        {
            for (int i = 0; i < n; i++)
                ptr[i] = ptr[i] * scale[i] + bias[i];
        }
        else
        {
            for (int i = 0; i < n; i++)
                ptr[i] = ptr[i] * scale[i];
        }
#endif // __riscv_vector
    }

    if (dims == 2)
    {
        int size = w * elempack;
#pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            float* ptr = bottom_top_blob.row(i);
#if __riscv_vector
            if (elempack == 1)
            {
                float s = scale[i];
                float b = bias_term ? bias[i] : 0.f;
                int remain = size;
                while (remain > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m8(remain);
                    vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                    vfloat32m8_t _res = __riscv_vfmul_vf_f32m8(_p, s, vl);
                    if (bias_term)
                        _res = __riscv_vfadd_vf_f32m8(_res, b, vl);
                    __riscv_vse32_v_f32m8(ptr, _res, vl);
                    ptr += vl;
                    remain -= vl;
                }
            }
            else
            {
                const float* s_pack = scale + i * elempack;
                const float* b_pack = bias_term ? (bias + i * elempack) : 0;
                int remain = size;
                while (remain > 0)
                {
                    size_t vl = elempack;
                    vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                    vfloat32m8_t _s = __riscv_vle32_v_f32m8(s_pack, vl);
                    vfloat32m8_t _res = __riscv_vfmul_vv_f32m8(_p, _s, vl);
                    if (bias_term)
                    {
                        vfloat32m8_t _b = __riscv_vle32_v_f32m8(b_pack, vl);
                        _res = __riscv_vfadd_vv_f32m8(_res, _b, vl);
                    }
                    __riscv_vse32_v_f32m8(ptr, _res, vl);
                    ptr += vl;
                    remain -= vl;
                }
            }
#else
            if (bias_term)
            {
                float s = scale[i];
                float b = bias[i];
                for (int j = 0; j < size; j++)
                    ptr[j] = ptr[j] * s + b;
            }
            else
            {
                float s = scale[i];
                for (int j = 0; j < size; j++)
                    ptr[j] = ptr[j] * s;
            }
#endif // __riscv_vector
        }
    }

    if (dims == 3 || dims == 4)
    {
        int size = w * h * d * elempack;
#pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);
            const float* s_ptr = (elempack == 1) ? scale + q : scale + q * elempack;
            const float* b_ptr = (bias_term ? ((elempack == 1) ? bias + q : bias + q * elempack) : 0);
#if __riscv_vector
            int remain = size;
            if (bias_term)
            {
                while (remain > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m8(remain);
                    vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                    vfloat32m8_t _s = __riscv_vle32_v_f32m8(s_ptr, vl);
                    vfloat32m8_t _b = __riscv_vle32_v_f32m8(b_ptr, vl);
                    vfloat32m8_t _res = __riscv_vfmadd_vvf_f32m8(_p, _s, _b, vl);
                    __riscv_vse32_v_f32m8(ptr, _res, vl);
                    ptr += vl;
                    s_ptr += vl;
                    b_ptr += vl;
                    remain -= vl;
                }
            }
            else
            {
                while (remain > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m8(remain);
                    vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                    vfloat32m8_t _s = __riscv_vle32_v_f32m8(s_ptr, vl);
                    vfloat32m8_t _res = __riscv_vfmul_vv_f32m8(_p, _s, vl);
                    __riscv_vse32_v_f32m8(ptr, _res, vl);
                    ptr += vl;
                    s_ptr += vl;
                    remain -= vl;
                }
            }
#else
            if (bias_term)
            {
                float s = scale[q];
                float b = bias[q];
                for (int i = 0; i < size; i++)
                    ptr[i] = ptr[i] * s + b;
            }
            else
            {
                float s = scale[q];
                for (int i = 0; i < size; i++)
                    ptr[i] = ptr[i] * s;
            }
#endif // __riscv_vector
        }
    }

    return 0;
}

} // namespace ncnn
