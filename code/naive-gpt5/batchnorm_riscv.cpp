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

#include "batchnorm_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

#include "riscv_usability.h"

namespace ncnn {

BatchNorm_riscv::BatchNorm_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector
}

int BatchNorm_riscv::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int dims = bottom_top_blob.dims;
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int c = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;

    if (dims == 1)
    {
        float* ptr = bottom_top_blob;
        const float* aptr = a_data;
        const float* bptr = b_data;
        int size = w * elempack;

#if __riscv_vector
        int n = size;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(n);
            vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
            vfloat32m8_t _a = __riscv_vle32_v_f32m8(aptr, vl);
            vfloat32m8_t _b = __riscv_vle32_v_f32m8(bptr, vl);
            _p = __riscv_vfmadd_vv_f32m8(_p, _b, _a, vl);
            __riscv_vse32_v_f32m8(ptr, _p, vl);
            ptr += vl;
            aptr += vl;
            bptr += vl;
            n -= vl;
        }
#else
        for (int i = 0; i < size; i++)
        {
            *ptr = *bptr * *ptr + *aptr;
            ptr++;
            aptr++;
            bptr++;
        }
#endif // __riscv_vector
        return 0;
    }

    if (dims == 2)
    {
        const int size = w * elempack;
#if __riscv_vector
        const int packn = (int)(csrr_vlenb() / 4);
#endif
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            float* ptr = bottom_top_blob.row(i);
            float a = a_data[i];
            float b = b_data[i];

#if __riscv_vector
            if (elempack == packn)
            {
                size_t vl = __riscv_vsetvl_e32m1(packn);
                vfloat32m1_t _a = __riscv_vle32_v_f32m1((const float*)a_data + i * packn, vl);
                vfloat32m1_t _b = __riscv_vle32_v_f32m1((const float*)b_data + i * packn, vl);
                for (int j = 0; j < size; j += packn)
                {
                    vfloat32m1_t _p = __riscv_vle32_v_f32m1(ptr, vl);
                    _p = __riscv_vfmadd_vv_f32m1(_p, _b, _a, vl);
                    __riscv_vse32_v_f32m1(ptr, _p, vl);
                    ptr += packn;
                }
            }
            else
            {
                int n = size;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m8(n);
                    vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                    _p = __riscv_vfmul_vf_f32m8(_p, b, vl);
                    _p = __riscv_vfadd_vf_f32m8(_p, a, vl);
                    __riscv_vse32_v_f32m8(ptr, _p, vl);
                    ptr += vl;
                    n -= vl;
                }
            }
#else
            for (int j = 0; j < size; j++)
            {
                *ptr = b * *ptr + a;
                ptr++;
            }
#endif // __riscv_vector
        }
        return 0;
    }

    if (dims == 3 || dims == 4)
    {
        const int size = w * h * d * elempack;
#if __riscv_vector
        const int packn = (int)(csrr_vlenb() / 4);
#endif
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < c; q++)
        {
            float* ptr = bottom_top_blob.channel(q);
            float a = a_data[q];
            float b = b_data[q];

#if __riscv_vector
            if (elempack == packn)
            {
                size_t vl = __riscv_vsetvl_e32m1(packn);
                vfloat32m1_t _a = __riscv_vle32_v_f32m1((const float*)a_data + q * packn, vl);
                vfloat32m1_t _b = __riscv_vle32_v_f32m1((const float*)b_data + q * packn, vl);
                for (int i = 0; i < size; i += packn)
                {
                    vfloat32m1_t _p = __riscv_vle32_v_f32m1(ptr, vl);
                    _p = __riscv_vfmadd_vv_f32m1(_p, _b, _a, vl);
                    __riscv_vse32_v_f32m1(ptr, _p, vl);
                    ptr += packn;
                }
            }
            else
            {
                int n = size;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m8(n);
                    vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                    _p = __riscv_vfmul_vf_f32m8(_p, b, vl);
                    _p = __riscv_vfadd_vf_f32m8(_p, a, vl);
                    __riscv_vse32_v_f32m8(ptr, _p, vl);
                    ptr += vl;
                    n -= vl;
                }
            }
#else
            for (int i = 0; i < size; i++)
            {
                *ptr = b * *ptr + a;
                ptr++;
            }
#endif // __riscv_vector
        }
        return 0;
    }

    return 0;
}

} // namespace ncnn
