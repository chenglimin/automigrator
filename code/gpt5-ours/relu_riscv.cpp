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

#include "relu_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

#include "cpu.h"

namespace ncnn {

ReLU_riscv::ReLU_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector
}

int ReLU_riscv::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int elembits = bottom_top_blob.elembits();

    if (elembits == 8)
        return forward_inplace_int8(bottom_top_blob, opt);

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;
    int size = w * h * d * elempack;

    if (slope == 0.f)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

#if __riscv_vector
            int n = size;
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e32m8(n);
                vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                _p = __riscv_vfmax_vf_f32m8(_p, 0.f, vl);
                __riscv_vse32_v_f32m8(ptr, _p, vl);
                ptr += vl;
                n -= vl;
            }
#else  // __riscv_vector
            for (int i = 0; i < size; i++)
            {
                if (ptr[i] < 0)
                    ptr[i] = 0.f;
            }
#endif // __riscv_vector
        }
    }
    else
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

#if __riscv_vector
            int n = size;
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e32m8(n);
                vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                // mask for values < 0
                vbool4_t _mask = __riscv_vmflt_vf_f32m8_b4(_p, 0.f, vl);
                // multiply by slope for negative lanes (merge operand required by RVV 1.0)
                _p = __riscv_vfmul_vf_f32m8_mu(_mask, _p, _p, slope, vl);
                __riscv_vse32_v_f32m8(ptr, _p, vl);
                ptr += vl;
                n -= vl;
            }
#else  // __riscv_vector
            for (int i = 0; i < size; i++)
            {
                if (ptr[i] < 0)
                    ptr[i] *= slope;
            }
#endif // __riscv_vector
        }
    }

    return 0;
}

int ReLU_riscv::forward_inplace_int8(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;
    int size = w * h * d * elempack;

    if (slope == 0.f)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            signed char* ptr = bottom_top_blob.channel(q);

#if __riscv_vector
            int n = size;
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e8m8(n);
                vint8m8_t _p = __riscv_vle8_v_i8m8(ptr, vl);
                vbool1_t _mask = __riscv_vmslt_vx_i8m8_b1(_p, 0, vl);
                vint8m8_t _zero = __riscv_vmv_v_x_i8m8(0, vl);
                // set negative lanes to zero
                _p = __riscv_vmerge_vvm_i8m8(_p, _zero, _mask, vl);
                __riscv_vse8_v_i8m8(ptr, _p, vl);
                ptr += vl;
                n -= vl;
            }
#else  // __riscv_vector
            for (int i = 0; i < size; i++)
            {
                if (*ptr < 0)
                    *ptr = 0;
                ptr++;
            }
#endif // __riscv_vector
        }
    }
    else
    {
        // TODO leakyrelu int8 (follow reduction_map and RVV_version_map when enabling int8 leakyrelu)
    }

    return 0;
}

} // namespace ncnn
