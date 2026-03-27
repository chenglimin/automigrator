// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "rmsnorm_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#include "rvv_mathfun.h"
#include "riscv_usability.h"
#endif // __riscv_vector

namespace ncnn {

RMSNorm_riscv::RMSNorm_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector
}

static inline void rmsnorm_scalar(float* ptr, const float* gamma_ptr, float eps, int elemcount, int elempack)
{
    const int size = elemcount * elempack;
    float sqsum = 0.f;
    for (int i = 0; i < size; i++) sqsum += ptr[i] * ptr[i];
    float inv_rms = 1.f / sqrtf(sqsum / elemcount + eps);
    if (gamma_ptr)
    {
        if (elempack == 1)
        {
            for (int i = 0; i < size; i++) ptr[i] = (ptr[i] * inv_rms) * gamma_ptr[i];
        }
        else
        {
            // gamma broadcast per-pack element
            for (int i = 0; i < elemcount; i++)
            {
                float g = gamma_ptr[i];
                for (int k = 0; k < elempack; k++)
                {
                    ptr[i * elempack + k] = (ptr[i * elempack + k] * inv_rms) * g;
                }
            }
        }
    }
    else
    {
        for (int i = 0; i < size; i++) ptr[i] = ptr[i] * inv_rms;
    }
}

#if __riscv_vector
static inline void rmsnorm_v(float* ptr, const float* gamma_ptr, float eps, int elemcount, int elempack)
{
    const int size = elemcount * elempack;

    if (elempack == 1)
    {
        // scalar reduction with RVV assisted chunking
        float inv_rms = 0.f;
        {
            int n = size;
            float* p = ptr;
            vfloat32m1_t vacc0 = __riscv_vfmv_v_f_f32m1(0.f, 1);
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e32m8(n);
                vfloat32m8_t _p = __riscv_vle32_v_f32m8(p, vl);
                vfloat32m8_t _pp = __riscv_vfmul_vv_f32m8(_p, _p, vl);
                vfloat32m1_t vzero = __riscv_vfmv_v_f_f32m1(0.f, vl);
                vfloat32m1_t vsum = __riscv_vfredusum_vs_f32m8_f32m1(_pp, vzero, vl);
                float acc = __riscv_vfmv_f_s_f32m1_f32(vacc0) + __riscv_vfmv_f_s_f32m1_f32(vsum);
                vacc0 = __riscv_vfmv_s_f_f32m1(acc, 1);
                p += vl;
                n -= vl;
            }
            float sum = __riscv_vfmv_f_s_f32m1_f32(vacc0);
            inv_rms = 1.f / sqrtf(sum / elemcount + eps);
        }

        // scale by inv_rms and optional gamma
        int n = size;
        float* p = ptr;
        if (gamma_ptr)
        {
            const float* g = gamma_ptr;
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e32m8(n);
                vfloat32m8_t _p = __riscv_vle32_v_f32m8(p, vl);
                vfloat32m8_t _g = __riscv_vle32_v_f32m8(g, vl);
                _p = __riscv_vfmul_vf_f32m8(_p, inv_rms, vl);
                _p = __riscv_vfmul_vv_f32m8(_p, _g, vl);
                __riscv_vse32_v_f32m8(p, _p, vl);
                p += vl;
                g += vl;
                n -= vl;
            }
        }
        else
        {
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e32m8(n);
                vfloat32m8_t _p = __riscv_vle32_v_f32m8(p, vl);
                _p = __riscv_vfmul_vf_f32m8(_p, inv_rms, vl);
                __riscv_vse32_v_f32m8(p, _p, vl);
                p += vl;
                n -= vl;
            }
        }
        return;
    }

    // elempack > 1, accumulate per-lane sum of squares
    const int packn = csrr_vlenb() / 4;
    size_t vlp = __riscv_vsetvl_e32m8(packn);
    vfloat32m8_t _rms = __riscv_vfmv_v_f_f32m8(0.f, vlp);

    int i = 0;
    while (i < size)
    {
        vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr + i, vlp);
        _rms = __riscv_vfadd_vv_f32m8(_rms, __riscv_vfmul_vv_f32m8(_p, _p, vlp), vlp);
        i += packn;
    }

    // compute inv_rms vector = 1 / sqrt((_rms / elemcount) + eps)
    _rms = __riscv_vfdiv_vf_f32m8(_rms, (float)elemcount, vlp);
    _rms = __riscv_vfadd_vf_f32m8(_rms, eps, vlp);
    vfloat32m8_t _sqrt = __riscv_vfsqrt_v_f32m8(_rms, vlp);
    vfloat32m8_t _inv_rms = __riscv_vfrdiv_vf_f32m8(_sqrt, 1.f, vlp); // 1.f / sqrt

    // apply scaling and gamma
    i = 0;
    if (gamma_ptr)
    {
        const float* g = gamma_ptr;
        while (i < size)
        {
            vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr + i, vlp);
            vfloat32m8_t _g = __riscv_vfmv_v_f_f32m8(g[0], vlp);
            _p = __riscv_vfmul_vv_f32m8(_p, _inv_rms, vlp);
            _p = __riscv_vfmul_vv_f32m8(_p, _g, vlp);
            __riscv_vse32_v_f32m8(ptr + i, _p, vlp);
            i += packn;
            g += 1;
        }
    }
    else
    {
        while (i < size)
        {
            vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr + i, vlp);
            _p = __riscv_vfmul_vv_f32m8(_p, _inv_rms, vlp);
            __riscv_vse32_v_f32m8(ptr + i, _p, vlp);
            i += packn;
        }
    }
}
#endif // __riscv_vector

int RMSNorm_riscv::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    const int dims = bottom_top_blob.dims;
    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int channels = bottom_top_blob.c;
    const int elempack = bottom_top_blob.elempack;

    if (dims == 1)
    {
        float* ptr = bottom_top_blob;
#if __riscv_vector
        rmsnorm_v(ptr, gamma_data, eps, w * elempack, 1);
#else
        rmsnorm_scalar(ptr, gamma_data, eps, w * elempack, 1);
#endif
    }

    if (dims == 2)
    {
        // assert affine_size == w
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            float* ptr = bottom_top_blob.row(i);
#if __riscv_vector
            rmsnorm_v(ptr, gamma_data, eps, w, elempack);
#else
            rmsnorm_scalar(ptr, gamma_data, eps, w, elempack);
#endif
        }
    }

    if (dims == 3)
    {
        if (affine_size == w)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                for (int i = 0; i < h; i++)
                {
                    float* ptr = bottom_top_blob.channel(q).row(i);
#if __riscv_vector
                    rmsnorm_v(ptr, gamma_data, eps, w, elempack);
#else
                    rmsnorm_scalar(ptr, gamma_data, eps, w, elempack);
#endif
                }
            }
        }
        else // if (affine_size == w * h)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);
#if __riscv_vector
                rmsnorm_v(ptr, gamma_data, eps, w * h, elempack);
#else
                rmsnorm_scalar(ptr, gamma_data, eps, w * h, elempack);
#endif
            }
        }
    }

    return 0;
}

} // namespace ncnn
