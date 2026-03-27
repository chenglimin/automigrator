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

#include "softmax_riscv.h"

#include <float.h>

#if __riscv_vector
#include <riscv_vector.h>
#include "rvv_mathfun.h"
#include "riscv_usability.h"
#endif // __riscv_vector

#include "cpu.h"
#include "packing.h"

namespace ncnn {

Softmax_riscv::Softmax_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector
}

#if __riscv_vector
static inline void softmax_scalar(float* ptr, int elemcount)
{
    float maxv = -FLT_MAX;
    for (int i = 0; i < elemcount; i++)
        maxv = std::max(maxv, ptr[i]);
    float sumv = 0.f;
    for (int i = 0; i < elemcount; i++)
    {
        float v = expf(ptr[i] - maxv);
        ptr[i] = v;
        sumv += v;
    }
    float inv = 1.f / sumv;
    for (int i = 0; i < elemcount; i++)
        ptr[i] *= inv;
}

static void softmax_packn_contiguous(float* ptr, int elemcount, int elempack)
{
    // contiguous memory with elempack elements interleaved
    int n = elemcount * elempack;
    while (n > 0)
    {
        size_t vl = __riscv_vsetvl_e32m8(n);
        vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
        // reduce max
        vfloat32m1_t _maxr = __riscv_vfmv_v_f_f32m1(-FLT_MAX, __riscv_vsetvlmax_e32m1());
        _maxr = __riscv_vfredmax_vs_f32m8_f32m1(_p, _maxr, vl);
        float maxv = __riscv_vfmv_f_s_f32m1_f32(_maxr);
        // exp and sum
        vfloat32m8_t _maxv = __riscv_vfmv_v_f_f32m8(maxv, vl);
        _p = __riscv_vfsub_vv_f32m8(_p, _maxv, vl);
        _p = exp_ps(_p, vl);
        __riscv_vse32_v_f32m8(ptr, _p, vl);
        vfloat32m1_t _sumr = __riscv_vfmv_v_f_f32m1(0.f, __riscv_vsetvlmax_e32m1());
        _sumr = __riscv_vfredusum_vs_f32m8_f32m1(_p, _sumr, vl);
        float inv = 1.f / __riscv_vfmv_f_s_f32m1_f32(_sumr);
        vfloat32m8_t _inv = __riscv_vfmv_v_f_f32m8(inv, vl);
        _p = __riscv_vle32_v_f32m8(ptr, vl);
        _p = __riscv_vfmul_vv_f32m8(_p, _inv, vl);
        __riscv_vse32_v_f32m8(ptr, _p, vl);
        ptr += vl;
        n -= vl;
    }
}

static void softmax_stride_packn(float* _ptr, int elemcount, int stride, int size1)
{
    // ptr layout: [elemcount x size1] with stride elements between rows
    // we compute along elemcount dimension for each column j in [0, size1)
    const int packn = csrr_vlenb() / 4;
    const size_t vl_pack1 = __riscv_vsetvl_e32m1(packn);

    // initialize workspace max and sum buffers of length size1
    // We reuse stack buffers in chunks to avoid large allocations.
    // Process columns by blocks of vector length to utilize RVV well.
    int j = 0;
    for (; j < size1; )
    {
        int remain = size1 - j;
        size_t vl = __riscv_vsetvl_e32m8(remain);
        // init max to -inf and sum to 0 for current block columns
        vfloat32m8_t vmaxv = __riscv_vfmv_v_f_f32m8(-FLT_MAX, vl);
        // reduction pass to get per-column max
        const float* rptr = _ptr + j;
        for (int i = 0; i < elemcount; i++)
        {
            vfloat32m8_t vcol = __riscv_vle32_v_f32m8(rptr, vl);
            vmaxv = __riscv_vfmax_vv_f32m8(vmaxv, vcol, vl);
            rptr += stride;
        }
        // second pass: exp(x-max) and accumulate sum, also write back
        vfloat32m8_t vsumv = __riscv_vfmv_v_f_f32m8(0.f, vl);
        float* wptr = _ptr + j;
        for (int i = 0; i < elemcount; i++)
        {
            vfloat32m8_t vcol = __riscv_vle32_v_f32m8(wptr, vl);
            vcol = __riscv_vfsub_vv_f32m8(vcol, vmaxv, vl);
            vcol = exp_ps(vcol, vl);
            __riscv_vse32_v_f32m8(wptr, vcol, vl);
            vsumv = __riscv_vfadd_vv_f32m8(vsumv, vcol, vl);
            wptr += stride;
        }
        // compute inv sum and final scale
        // avoid division by zero theoretically never happens due to exp
        // write back normalization
        // broadcast invsum per column
        // We need element-wise division of each column entry by sum
        // So loop rows again
        wptr = _ptr + j;
        for (int i = 0; i < elemcount; i++)
        {
            vfloat32m8_t vcol = __riscv_vle32_v_f32m8(wptr, vl);
            vcol = __riscv_vfdiv_vv_f32m8(vcol, vsumv, vl);
            __riscv_vse32_v_f32m8(wptr, vcol, vl);
            wptr += stride;
        }
        j += vl;
    }
}
#endif // __riscv_vector

int Softmax_riscv::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    const int dims = bottom_top_blob.dims;
    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int d = bottom_top_blob.d;
    const int channels = bottom_top_blob.c;
    const int elempack = bottom_top_blob.elempack;
    const int positive_axis = axis < 0 ? dims + axis : axis;

#if __riscv_vector
    if (elempack == 1)
    {
        // For scalar-packing, use generic implementation to preserve correctness and stride handling
        return Softmax::forward_inplace(bottom_top_blob, opt);
    }

    // elempack > 1 path
    if (elempack > 1)
    {
        // Robust fallback: unpack to elempack=1, run generic softmax, then repack back
        // This ensures correctness across all axis cases and varying VLEN
        Mat planar;
        {
            Packing pack_to1;
            pack_to1.out_elempack = 1;
            pack_to1.use_padding = 0;
            int ret = pack_to1.forward(bottom_top_blob, planar, opt);
            if (ret != 0)
                return ret;
        }
        int ret = Softmax::forward_inplace(planar, opt);
        if (ret != 0)
            return ret;
        {
            Packing pack_back;
            pack_back.out_elempack = elempack;
            pack_back.use_padding = 0;
            Mat restored;
            ret = pack_back.forward(planar, restored, opt);
            if (ret != 0)
                return ret;
            bottom_top_blob = restored;
        }
        return 0;
    }
#endif // __riscv_vector

    // fallback generic
    return Softmax::forward_inplace(bottom_top_blob, opt);
}

} // namespace ncnn
