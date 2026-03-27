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

static inline void batchnorm_apply_vector(float* ptr, const float* a_data, const float* b_data, int n)
{
#if __riscv_vector
    while (n > 0)
    {
        size_t vl = __riscv_vsetvl_e32m1(n);
        vfloat32m1_t _p = __riscv_vle32_v_f32m1(ptr, vl);
        vfloat32m1_t _a = __riscv_vle32_v_f32m1(a_data, vl);
        vfloat32m1_t _b = __riscv_vle32_v_f32m1(b_data, vl);
        _p = __riscv_vfmadd_vv_f32m1(_p, _b, _a, vl);
        __riscv_vse32_v_f32m1(ptr, _p, vl);
        ptr += vl;
        a_data += vl;
        b_data += vl;
        n -= vl;
    }
#else
    for (int i = 0; i < n; i++)
    {
        ptr[i] = b_data[i] * ptr[i] + a_data[i];
    }
#endif
}

int BatchNorm_riscv::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int dims = bottom_top_blob.dims;
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int c = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;

    // Fallback to unpack->generic->repack for packed layout to ensure correctness across packn
    if (elempack != 1)
    {
        Mat bottom_top_blob_unpacked;
        convert_packing(bottom_top_blob, bottom_top_blob_unpacked, 1, opt);
        if (bottom_top_blob_unpacked.empty())
            return -100;

        // apply generic scalar batchnorm per original implementation on unpacked layout
        {
            int u_dims = bottom_top_blob_unpacked.dims;
            int u_w = bottom_top_blob_unpacked.w;
            int u_h = bottom_top_blob_unpacked.h;
            int u_d = bottom_top_blob_unpacked.d;
            int u_c = bottom_top_blob_unpacked.c;
            if (u_dims == 1)
            {
                float* ptr = bottom_top_blob_unpacked;
                for (int i = 0; i < u_w; i++)
                {
                    ptr[i] = b_data[i] * ptr[i] + a_data[i];
                }
            }
            else if (u_dims == 2)
            {
                for (int i = 0; i < u_h; i++)
                {
                    float* ptr = bottom_top_blob_unpacked.row(i);
                    float a = a_data[i];
                    float b = b_data[i];
                    for (int j = 0; j < u_w; j++) ptr[j] = b * ptr[j] + a;
                }
            }
            else // dims == 3 || dims == 4
            {
                int size = u_w * u_h * u_d;
                for (int q = 0; q < u_c; q++)
                {
                    float* ptr = bottom_top_blob_unpacked.channel(q);
                    float a = a_data[q];
                    float b = b_data[q];
                    for (int i = 0; i < size; i++) ptr[i] = b * ptr[i] + a;
                }
            }
        }

        // repack to original elempack
        Mat bottom_top_blob_repacked;
        convert_packing(bottom_top_blob_unpacked, bottom_top_blob_repacked, elempack, opt);
        if (bottom_top_blob_repacked.empty())
            return -100;
        bottom_top_blob = bottom_top_blob_repacked;
        return 0;
    }

    if (dims == 1)
    {
        float* ptr = bottom_top_blob;
        const float* aptr = a_data;
        const float* bptr = b_data;
        const int size = w;
        batchnorm_apply_vector(ptr, aptr, bptr, size);
    }

    if (dims == 2)
    {
        const int size = w;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            float* ptr = bottom_top_blob.row(i);
            float a = a_data[i];
            float b = b_data[i];
#if __riscv_vector
            int n = size;
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e32m8(n);
                vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                vfloat32m8_t _a = __riscv_vfmv_v_f_f32m8(a, vl);
                vfloat32m8_t _b = __riscv_vfmv_v_f_f32m8(b, vl);
                _p = __riscv_vfmadd_vv_f32m8(_p, _b, _a, vl);
                __riscv_vse32_v_f32m8(ptr, _p, vl);
                ptr += vl;
                n -= vl;
            }
#else
            for (int j = 0; j < w; j++)
            {
                ptr[j] = b * ptr[j] + a;
            }
#endif
        }
    }

    if (dims == 3 || dims == 4)
    {
        const int size = w * h * d;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < c; q++)
        {
            float* ptr = bottom_top_blob.channel(q);
            float a = a_data[q];
            float b = b_data[q];
#if __riscv_vector
            int n = size;
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e32m8(n);
                vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                vfloat32m8_t _a = __riscv_vfmv_v_f_f32m8(a, vl);
                vfloat32m8_t _b = __riscv_vfmv_v_f_f32m8(b, vl);
                _p = __riscv_vfmadd_vv_f32m8(_p, _b, _a, vl);
                __riscv_vse32_v_f32m8(ptr, _p, vl);
                ptr += vl;
                n -= vl;
            }
#else
            for (int i = 0; i < size; i++)
            {
                ptr[i] = b * ptr[i] + a;
            }
#endif
        }
    }

    return 0;
}

} // namespace ncnn
