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

#include "prelu_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#include "riscv_activation.h"
#endif // __riscv_vector

#include "cpu.h"

namespace ncnn {

PReLU_riscv::PReLU_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector
}

int PReLU_riscv::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int dims = bottom_top_blob.dims;
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;

#if __riscv_vector
    // Use rvv vector when available
    if (dims == 1)
    {
        int size = w * elempack;
        float* ptr = bottom_top_blob;

        if (num_slope > 1)
        {
            const float* slope = slope_data;
            int n = size;
            int i = 0;
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e32m8(n);
                vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr + i, vl);
                vfloat32m8_t _slope = __riscv_vle32_v_f32m8(slope + i, vl);
                vbool4_t _mask = __riscv_vmflt_vf_f32m8_b4(_p, 0.f, vl);
                vfloat32m8_t _ps = __riscv_vfmul_vv_f32m8(_p, _slope, vl);
                _p = __riscv_vmerge_vvm_f32m8(_p, _ps, _mask, vl);
                __riscv_vse32_v_f32m8(ptr + i, _p, vl);
                i += vl;
                n -= vl;
            }
        }
        else
        {
            const float slope = slope_data[0];
            int n = size;
            int i = 0;
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e32m8(n);
                vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr + i, vl);
                vbool4_t _mask = __riscv_vmflt_vf_f32m8_b4(_p, 0.f, vl);
                vfloat32m8_t _ps = __riscv_vfmul_vf_f32m8(_p, slope, vl);
                _p = __riscv_vmerge_vvm_f32m8(_p, _ps, _mask, vl);
                __riscv_vse32_v_f32m8(ptr + i, _p, vl);
                i += vl;
                n -= vl;
            }
        }
    }

    if (dims == 2)
    {
        int size = w * elempack;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int irow = 0; irow < h; irow++)
        {
            float* ptr = bottom_top_blob.row(irow);
            const float slope = num_slope > 1 ? slope_data[irow] : slope_data[0];

            int n = size;
            int i = 0;
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e32m8(n);
                vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr + i, vl);
                vbool4_t _mask = __riscv_vmflt_vf_f32m8_b4(_p, 0.f, vl);
                vfloat32m8_t _ps;
                if (num_slope > 1 && (elempack == 4))
                {
                    _ps = __riscv_vfmul_vv_f32m8(_p, __riscv_vle32_v_f32m8((const float*)slope_data + irow * 4 + i, vl), vl);
                }
                else
                {
                    _ps = __riscv_vfmul_vf_f32m8(_p, slope, vl);
                }
                _p = __riscv_vmerge_vvm_f32m8(_p, _ps, _mask, vl);
                __riscv_vse32_v_f32m8(ptr + i, _p, vl);
                i += vl;
                n -= vl;
            }
        }
    }

    if (dims == 3)
    {
        int size = w * h * elempack;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);
            const float slope = num_slope > 1 ? slope_data[q] : slope_data[0];

            int n = size;
            int i = 0;
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e32m8(n);
                vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr + i, vl);
                vbool4_t _mask = __riscv_vmflt_vf_f32m8_b4(_p, 0.f, vl);
                vfloat32m8_t _ps;
                if (num_slope > 1 && (elempack == 4))
                {
                    _ps = __riscv_vfmul_vv_f32m8(_p, __riscv_vle32_v_f32m8((const float*)slope_data + q * 4 + i, vl), vl);
                }
                else
                {
                    _ps = __riscv_vfmul_vf_f32m8(_p, slope, vl);
                }
                _p = __riscv_vmerge_vvm_f32m8(_p, _ps, _mask, vl);
                __riscv_vse32_v_f32m8(ptr + i, _p, vl);
                i += vl;
                n -= vl;
            }
        }
    }

    return 0;
#else  // __riscv_vector
    // Fallback to base implementation when vector is not available
    return PReLU::forward_inplace(bottom_top_blob, opt);
#endif // __riscv_vector
}

} // namespace ncnn
