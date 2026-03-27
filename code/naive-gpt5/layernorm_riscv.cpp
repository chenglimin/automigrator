// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "layernorm_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

#include "riscv_usability.h"

namespace ncnn {

LayerNorm_riscv::LayerNorm_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector
}

static inline void layernorm_rvv(float* ptr, const float* gamma_ptr, const float* beta_ptr, float eps, int elemcount, int elempack)
{
    const int size = elemcount * elempack;

#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
    if (elempack == packn)
    {
        const size_t vl = __riscv_vsetvl_e32m1(packn);
        // mean
        vfloat32m1_t _sum = __riscv_vfmv_v_f_f32m1(0.f, vl);
        for (int i = 0; i < elemcount; i++)
        {
            vfloat32m1_t _p = __riscv_vle32_v_f32m1(ptr + i * packn, vl);
            _sum = __riscv_vfadd_vv_f32m1(_sum, _p, vl);
        }
        vfloat32m1_t _mean = __riscv_vfdiv_vf_f32m1(_sum, (float)elemcount, vl);

        // variance
        vfloat32m1_t _sqsum = __riscv_vfmv_v_f_f32m1(0.f, vl);
        for (int i = 0; i < elemcount; i++)
        {
            vfloat32m1_t _p = __riscv_vle32_v_f32m1(ptr + i * packn, vl);
            _p = __riscv_vfsub_vv_f32m1(_p, _mean, vl);
            _sqsum = __riscv_vfmadd_vv_f32m1(_p, _p, _sqsum, vl);
        }
        vfloat32m1_t _var = __riscv_vfdiv_vf_f32m1(_sqsum, (float)elemcount, vl);
        _var = __riscv_vfadd_vf_f32m1(_var, eps, vl);
        // inv std
        vfloat32m1_t _inv_std = __riscv_vfrdiv_vf_f32m1(__riscv_vfsqrt_v_f32m1(_var, vl), 1.f, vl);
        vfloat32m1_t _mean_invstd = __riscv_vfmul_vv_f32m1(_mean, _inv_std, vl);

        if (gamma_ptr && beta_ptr)
        {
            for (int i = 0; i < elemcount; i++)
            {
                vfloat32m1_t _p = __riscv_vle32_v_f32m1(ptr + i * packn, vl);
                _p = __riscv_vfmul_vv_f32m1(_p, _inv_std, vl);
                _p = __riscv_vfsub_vv_f32m1(_p, _mean_invstd, vl);
                // broadcast gamma/beta for this group
                vfloat32m1_t _gamma = __riscv_vfmv_v_f_f32m1(gamma_ptr[i], vl);
                vfloat32m1_t _beta = __riscv_vfmv_v_f_f32m1(beta_ptr[i], vl);
                _p = __riscv_vfmul_vv_f32m1(_p, _gamma, vl);
                _p = __riscv_vfadd_vv_f32m1(_p, _beta, vl);
                __riscv_vse32_v_f32m1(ptr + i * packn, _p, vl);
            }
        }
        else
        {
            for (int i = 0; i < elemcount; i++)
            {
                vfloat32m1_t _p = __riscv_vle32_v_f32m1(ptr + i * packn, vl);
                _p = __riscv_vfmul_vv_f32m1(_p, _inv_std, vl);
                _p = __riscv_vfsub_vv_f32m1(_p, _mean_invstd, vl);
                __riscv_vse32_v_f32m1(ptr + i * packn, _p, vl);
            }
        }
        return;
    }
#endif // __riscv_vector

    // fallback for elempack == 1 or others
    // compute scalar mean and variance
    float mean = 0.f;
#if __riscv_vector
    // reduction with vector where possible
    {
        int n = size;
        float* p = ptr;
        vfloat32m1_t _sum = __riscv_vfmv_s_f_f32m1(0.f, __riscv_vsetvlmax_e32m1());
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(n);
            vfloat32m8_t _v = __riscv_vle32_v_f32m8(p, vl);
            _sum = __riscv_vfredusum_vs_f32m8_f32m1(_v, _sum, vl);
            p += vl;
            n -= vl;
        }
        mean = __riscv_vfmv_f_s_f32m1_f32(_sum) / size;
    }
#else
    for (int i = 0; i < size; i++)
        mean += ptr[i];
    mean /= size;
#endif

    float sqsum = 0.f;
#if __riscv_vector
    {
        int n = size;
        float* p = ptr;
        vfloat32m1_t _sq = __riscv_vfmv_s_f_f32m1(0.f, __riscv_vsetvlmax_e32m1());
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(n);
            vfloat32m8_t _v = __riscv_vle32_v_f32m8(p, vl);
            _v = __riscv_vfsub_vf_f32m8(_v, mean, vl);
            _sq = __riscv_vfredosum_vs_f32m8_f32m1(__riscv_vfmul_vv_f32m8(_v, _v, vl), _sq, vl);
            p += vl;
            n -= vl;
        }
        sqsum = __riscv_vfmv_f_s_f32m1_f32(_sq);
    }
#else
    for (int i = 0; i < size; i++)
    {
        float v = ptr[i] - mean;
        sqsum += v * v;
    }
#endif

    float inv_std = 1.f / sqrtf(sqsum / size + eps);
    float mean_invstd = mean * inv_std;

#if __riscv_vector
    {
        int n = size;
        float* p = ptr;
        const float* g = gamma_ptr;
        const float* b = beta_ptr;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(n);
            vfloat32m8_t _v = __riscv_vle32_v_f32m8(p, vl);
            _v = __riscv_vfmul_vf_f32m8(_v, inv_std, vl);
            _v = __riscv_vfsub_vf_f32m8(_v, mean_invstd, vl);
            if (g && b)
            {
                vfloat32m8_t _g = __riscv_vle32_v_f32m8(g, vl);
                vfloat32m8_t _b = __riscv_vle32_v_f32m8(b, vl);
                _v = __riscv_vfmul_vv_f32m8(_v, _g, vl);
                _v = __riscv_vfadd_vv_f32m8(_v, _b, vl);
                g += vl;
                b += vl;
            }
            __riscv_vse32_v_f32m8(p, _v, vl);
            p += vl;
            n -= vl;
        }
    }
#else
    if (gamma_ptr && beta_ptr)
    {
        for (int i = 0; i < size; i++)
            ptr[i] = (ptr[i] * inv_std - mean_invstd) * gamma_ptr[i] + beta_ptr[i];
    }
    else
    {
        for (int i = 0; i < size; i++)
            ptr[i] = (ptr[i] * inv_std - mean_invstd);
    }
#endif
}

int LayerNorm_riscv::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    const int dims = bottom_top_blob.dims;
    const int elempack = bottom_top_blob.elempack;
    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int channels = bottom_top_blob.c;

    if (dims == 1)
    {
        float* ptr = bottom_top_blob;
        // treat as contiguous vector, apply per-element gamma/beta
        layernorm_rvv(ptr, (const float*)gamma_data, (const float*)beta_data, eps, w * elempack, 1);
        return 0;
    }

    if (dims == 2)
    {
        // assert affine_size == w
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            float* ptr = bottom_top_blob.row(i);
            layernorm_rvv(ptr, (const float*)gamma_data, (const float*)beta_data, eps, w, elempack);
        }
        return 0;
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
                    layernorm_rvv(ptr, (const float*)gamma_data, (const float*)beta_data, eps, w, elempack);
                }
            }
        }
        else // if (affine_size == w * h)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);
                layernorm_rvv(ptr, (const float*)gamma_data, (const float*)beta_data, eps, w * h, elempack);
            }
        }
        return 0;
    }

    return 0;
}

} // namespace ncnn
