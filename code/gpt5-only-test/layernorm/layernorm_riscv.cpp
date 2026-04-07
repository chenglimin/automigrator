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

#include "layernorm_riscv.h"

#include <math.h>

#if __riscv_vector
#include <riscv_vector.h>
#include "rvv_mathfun.h"
#endif // __riscv_vector

#include "cpu.h"

namespace ncnn {

LayerNorm_riscv::LayerNorm_riscv()
{
#if __riscv_vector
    support_packing = true;
#if NCNN_ZFH
    support_fp16_storage = cpu_support_riscv_zvfh();
#endif
#endif // __riscv_vector
}

#if __riscv_vector
static void layernorm(float* ptr, const float* gamma_ptr, const float* beta_ptr, float eps, int elemcount, int elempack)
{
    const int size = elemcount * elempack;

    // compute mean
    vfloat32m8_t _mean = __riscv_vfmv_v_f_f32m8(0.f, __riscv_vsetvl_e32m8(size));
    float mean = 0.f;
    {
        const float* ptr0 = ptr;
        int n = size;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(n);
            vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr0, vl);
            _mean = __riscv_vfadd_vv_f32m8(_mean, _p, vl);
            ptr0 += vl;
            n -= vl;
        }
    }

    if (elempack == 1)
    {
        // reduce vector sum to scalar
        // fallback reduction by scalar loop to avoid vfredsum portability issues
        // re-scan to accumulate scalar sum into mean
        const float* p = ptr;
        for (int i = 0; i < size; i++) mean += p[i];
        mean = mean / elemcount;
        _mean = __riscv_vfmv_v_f_f32m8(mean, __riscv_vsetvl_e32m8(1));
    }
    else
    {
        // pack4: per-lane mean
        // divide by elemcount
        size_t vl1 = __riscv_vsetvl_e32m8(elempack);
        vfloat32m8_t _elemcount = __riscv_vfmv_v_f_f32m8((float)elemcount, vl1);
        _mean = __riscv_vfdiv_vv_f32m8(_mean, _elemcount, vl1);
    }

    // compute variance and rsqrt
    vfloat32m8_t _var = __riscv_vfmv_v_f_f32m8(0.f, __riscv_vsetvl_e32m8(size));
    float var = 0.f;
    {
        const float* ptr0 = ptr;
        int n = size;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(n);
            vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr0, vl);
            // broadcast mean to current vl
            vfloat32m8_t _mean_vl = __riscv_vfmv_v_f_f32m8(mean, vl);
            vfloat32m8_t _d = __riscv_vfsub_vv_f32m8(_p, _mean_vl, vl);
            _var = __riscv_vfmadd_vv_f32m8(_var, _d, _d, vl);
            ptr0 += vl;
            n -= vl;
        }
    }

    if (elempack == 1)
    {
        // scalar normalization factors
        // accumulate var scalar
        const float* p = ptr;
        for (int i = 0; i < size; i++)
        {
            float d = p[i] - mean;
            var += d * d;
        }
        var = 1.f / sqrtf(var / elemcount + eps);
        mean = -mean * var;
    }
    else
    {
        size_t vl1 = __riscv_vsetvl_e32m8(elempack);
        vfloat32m8_t _elemcount = __riscv_vfmv_v_f_f32m8((float)elemcount, vl1);
        vfloat32m8_t _eps = __riscv_vfmv_v_f_f32m8(eps, vl1);
        _var = __riscv_vfdiv_vv_f32m8(_var, _elemcount, vl1);
        _var = __riscv_vfadd_vv_f32m8(_var, _eps, vl1);
        // approximate rsqrt via mathfun
        // rvv_mathfun provides rsqrt via 1/sqrt(x)
        vfloat32m8_t _rsqrt = __riscv_vfrdiv_vf_f32m8(__riscv_vfsqrt_v_f32m8(_var, vl1), 1.f, vl1);
        _mean = __riscv_vfmul_vv_f32m8(_mean, _rsqrt, vl1);
        _mean = __riscv_vfneg_v_f32m8(_mean, vl1);
        _var = _rsqrt;
    }

    if (gamma_ptr && beta_ptr)
    {
        int n = size;
        float* p = ptr;
        if (elempack == 1)
        {
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e32m8(n);
                vfloat32m8_t _p = __riscv_vle32_v_f32m8(p, vl);
                vfloat32m8_t _gamma = __riscv_vle32_v_f32m8(gamma_ptr, vl);
                vfloat32m8_t _beta = __riscv_vle32_v_f32m8(beta_ptr, vl);
                vfloat32m8_t _var_vl = __riscv_vfmv_v_f_f32m8(var, vl);
                vfloat32m8_t _mean_vl = __riscv_vfmv_v_f_f32m8(mean, vl);
                _p = __riscv_vfmadd_vv_f32m8(_mean_vl, _p, _var_vl, vl);
                _p = __riscv_vfmadd_vv_f32m8(_beta, _p, _gamma, vl);
                __riscv_vse32_v_f32m8(p, _p, vl);
                p += vl;
                gamma_ptr += vl;
                beta_ptr += vl;
                n -= vl;
            }
        }
        else
        {
            // elempack lanes share same gamma/beta when affine_size equals elemcount
            int i = 0;
            while (i + elempack - 1 < size)
            {
                size_t vl = __riscv_vsetvl_e32m8(elempack);
                vfloat32m8_t _p = __riscv_vle32_v_f32m8(p, vl);
                vfloat32m8_t _gamma = __riscv_vfmv_v_f_f32m8(gamma_ptr[0], vl);
                vfloat32m8_t _beta = __riscv_vfmv_v_f_f32m8(beta_ptr[0], vl);
                _p = __riscv_vfmadd_vv_f32m8(_mean, _p, _var, vl);
                _p = __riscv_vfmadd_vv_f32m8(_beta, _p, _gamma, vl);
                __riscv_vse32_v_f32m8(p, _p, vl);
                p += vl;
                gamma_ptr += 1;
                beta_ptr += 1;
                i += vl;
            }
            for (; i < size; i++)
            {
                p[0] = (p[0] * ((float)__riscv_vfmv_f_s_f32m1(_var)) + (float)__riscv_vfmv_f_s_f32m1(_mean)) * gamma_ptr[0] + beta_ptr[0];
                p++;
                gamma_ptr++;
                beta_ptr++;
            }
        }
    }
    else
    {
        int n = size;
        float* p = ptr;
        if (elempack == 1)
        {
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e32m8(n);
                vfloat32m8_t _p = __riscv_vle32_v_f32m8(p, vl);
                vfloat32m8_t _var_vl = __riscv_vfmv_v_f_f32m8(var, vl);
                vfloat32m8_t _mean_vl = __riscv_vfmv_v_f_f32m8(mean, vl);
                _p = __riscv_vfmadd_vv_f32m8(_mean_vl, _p, _var_vl, vl);
                __riscv_vse32_v_f32m8(p, _p, vl);
                p += vl;
                n -= vl;
            }
        }
        else
        {
            int i = 0;
            while (i + elempack - 1 < size)
            {
                size_t vl = __riscv_vsetvl_e32m8(elempack);
                vfloat32m8_t _p = __riscv_vle32_v_f32m8(p, vl);
                _p = __riscv_vfmadd_vv_f32m8(_mean, _p, _var, vl);
                __riscv_vse32_v_f32m8(p, _p, vl);
                p += vl;
                i += vl;
            }
            for (; i < size; i++)
            {
                p[0] = p[0] * ((float)__riscv_vfmv_f_s_f32m1(_var)) + (float)__riscv_vfmv_f_s_f32m1(_mean);
                p++;
            }
        }
    }
}
#endif // __riscv_vector

int LayerNorm_riscv::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    const int dims = bottom_top_blob.dims;
    const int elempack = bottom_top_blob.elempack;
    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int channels = bottom_top_blob.c;

#if __riscv_vector
    if (dims == 1)
    {
        float* ptr = bottom_top_blob;
        layernorm(ptr, gamma_data, beta_data, eps, w * elempack, 1);
    }
    else if (dims == 2)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            float* ptr = bottom_top_blob.row(i);
            layernorm(ptr, gamma_data, beta_data, eps, w, elempack);
        }
    }
    else if (dims == 3)
    {
        if (affine_size == w)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                for (int i = 0; i < h; i++)
                {
                    float* ptr = bottom_top_blob.channel(q).row(i);
                    layernorm(ptr, gamma_data, beta_data, eps, w, elempack);
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);
                layernorm(ptr, gamma_data, beta_data, eps, w * h, elempack);
            }
        }
    }
    else
    {
        // fallback to base implementation when vector not available
        return LayerNorm::forward_inplace(bottom_top_blob, opt);
    }
#else
    // no rvv, use base scalar implementation
    return LayerNorm::forward_inplace(bottom_top_blob, opt);
#endif

    return 0;
}

} // namespace ncnn
