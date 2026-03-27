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

#if __riscv_vector
static inline void layernorm_rvv(float* ptr, const float* gamma_ptr, const float* beta_ptr, float eps, int elemcount, int elempack)
{
    const int size = elemcount * elempack;

    if (elempack > 1)
    {
        // accumulate lane-wise mean across elemcount groups
        size_t vl_pack = __riscv_vsetvl_e32m8(elempack);
        vfloat32m8_t vmean = __riscv_vfmv_v_f_f32m8(0.f, vl_pack);

        const float* ptr0 = ptr;
        for (int i = 0; i < elemcount; i++)
        {
            vfloat32m8_t vp = __riscv_vle32_v_f32m8(ptr0, vl_pack);
            vmean = __riscv_vfadd_vv_f32m8(vmean, vp, vl_pack);
            ptr0 += elempack;
        }
        // mean per lane
        vmean = __riscv_vfdiv_vf_f32m8(vmean, (float)elemcount, vl_pack);

        // accumulate lane-wise variance
        vfloat32m8_t vvar = __riscv_vfmv_v_f_f32m8(0.f, vl_pack);
        ptr0 = ptr;
        for (int i = 0; i < elemcount; i++)
        {
            vfloat32m8_t vp = __riscv_vle32_v_f32m8(ptr0, vl_pack);
            vp = __riscv_vfsub_vv_f32m8(vp, vmean, vl_pack);
            vvar = __riscv_vfmacc_vv_f32m8(vvar, vp, vp, vl_pack); // vvar += vp*vp
            ptr0 += elempack;
        }
        vvar = __riscv_vfdiv_vf_f32m8(vvar, (float)elemcount, vl_pack);
        vvar = __riscv_vfadd_vf_f32m8(vvar, eps, vl_pack);
        // inv std = 1 / sqrt(var + eps)
        vfloat32m8_t vinvstd = __riscv_vfrdiv_vf_f32m8(__riscv_vfsqrt_v_f32m8(vvar, vl_pack), 1.f, vl_pack);
        // precompute mean * invstd
        vfloat32m8_t vmean_inv = __riscv_vfmul_vv_f32m8(vmean, vinvstd, vl_pack);

        // normalize and apply affine
        float* p = ptr;
        if (gamma_ptr && beta_ptr)
        {
            for (int i = 0; i < elemcount; i++)
            {
                vfloat32m8_t vp = __riscv_vle32_v_f32m8(p, vl_pack);
                // z = p * invstd - mean*invstd
                vp = __riscv_vfsub_vv_f32m8(__riscv_vfmul_vv_f32m8(vp, vinvstd, vl_pack), vmean_inv, vl_pack);
                // gamma/beta are scalar per elemcount, broadcast
                vfloat32m8_t vgamma = __riscv_vfmv_v_f_f32m8(gamma_ptr[i], vl_pack);
                vfloat32m8_t vbeta = __riscv_vfmv_v_f_f32m8(beta_ptr[i], vl_pack);
                vp = __riscv_vfmadd_vv_f32m8(vp, vgamma, vbeta, vl_pack); // vp = vp*gamma + beta
                __riscv_vse32_v_f32m8(p, vp, vl_pack);
                p += elempack;
            }
        }
        else
        {
            for (int i = 0; i < elemcount; i++)
            {
                vfloat32m8_t vp = __riscv_vle32_v_f32m8(p, vl_pack);
                vp = __riscv_vfsub_vv_f32m8(__riscv_vfmul_vv_f32m8(vp, vinvstd, vl_pack), vmean_inv, vl_pack);
                __riscv_vse32_v_f32m8(p, vp, vl_pack);
                p += elempack;
            }
        }
        return;
    }

    // elempack == 1 path: reduce to scalar mean/var, then vectorize application
    int n = size;
    vfloat32m8_t vsum = __riscv_vfmv_v_f_f32m8(0.f, __riscv_vsetvl_e32m8(n));
    const float* pscan = ptr;
    while (n > 0)
    {
        size_t vl = __riscv_vsetvl_e32m8(n);
        vfloat32m8_t vp = __riscv_vle32_v_f32m8(pscan, vl);
        vsum = __riscv_vfadd_vv_f32m8(vsum, vp, vl);
        pscan += vl;
        n -= vl;
    }
    // reduce to scalar
    size_t vl_all = __riscv_vsetvl_e32m8(size);
    vfloat32m1_t vzero = __riscv_vfmv_v_f_f32m1(0.f, vl_all);
    vfloat32m1_t vmean1 = __riscv_vfredusum_vs_f32m8_f32m1(vsum, vzero, vl_all);
    float mean = __riscv_vfmv_f_s_f32m1_f32(vmean1) / (float)size;

    // variance
    n = size;
    pscan = ptr;
    vfloat32m8_t vvarsum = __riscv_vfmv_v_f_f32m8(0.f, __riscv_vsetvl_e32m8(n));
    while (n > 0)
    {
        size_t vl = __riscv_vsetvl_e32m8(n);
        vfloat32m8_t vp = __riscv_vle32_v_f32m8(pscan, vl);
        vp = __riscv_vfsub_vf_f32m8(vp, mean, vl);
        vvarsum = __riscv_vfmacc_vv_f32m8(vvarsum, vp, vp, vl);
        pscan += vl;
        n -= vl;
    }
    vfloat32m1_t vvar1 = __riscv_vfredusum_vs_f32m8_f32m1(vvarsum, vzero, vl_all);
    float var = __riscv_vfmv_f_s_f32m1_f32(vvar1) / (float)size;
    float invstd = 1.f / sqrtf(var + eps);
    float mean_inv = mean * invstd;

    // apply normalization and affine
    n = size;
    float* p = ptr;
    if (gamma_ptr && beta_ptr)
    {
        const float* g = gamma_ptr;
        const float* b = beta_ptr;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(n);
            vfloat32m8_t vp = __riscv_vle32_v_f32m8(p, vl);
            vp = __riscv_vfsub_vf_f32m8(__riscv_vfmul_vf_f32m8(vp, invstd, vl), mean_inv, vl);
            vfloat32m8_t vgamma = __riscv_vle32_v_f32m8(g, vl);
            vfloat32m8_t vbeta = __riscv_vle32_v_f32m8(b, vl);
            vp = __riscv_vfmadd_vv_f32m8(vp, vgamma, vbeta, vl);
            __riscv_vse32_v_f32m8(p, vp, vl);
            p += vl;
            g += vl;
            b += vl;
            n -= vl;
        }
    }
    else
    {
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(n);
            vfloat32m8_t vp = __riscv_vle32_v_f32m8(p, vl);
            vp = __riscv_vfsub_vf_f32m8(__riscv_vfmul_vf_f32m8(vp, invstd, vl), mean_inv, vl);
            __riscv_vse32_v_f32m8(p, vp, vl);
            p += vl;
            n -= vl;
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

    if (dims == 1)
    {
        // assert affine_size == w
        float* ptr = bottom_top_blob;
#if __riscv_vector
        layernorm_rvv(ptr, (const float*)gamma_data, (const float*)beta_data, eps, w * elempack, 1);
#else
        // fallback to generic
        return LayerNorm::forward_inplace(bottom_top_blob, opt);
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
            layernorm_rvv(ptr, (const float*)gamma_data, (const float*)beta_data, eps, w, elempack);
#else
            Mat m = bottom_top_blob.row(i);
            LayerNorm::forward_inplace(m, opt);
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
                    layernorm_rvv(ptr, (const float*)gamma_data, (const float*)beta_data, eps, w, elempack);
#else
                    Mat m = bottom_top_blob.channel(q).row(i);
                    LayerNorm::forward_inplace(m, opt);
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
                layernorm_rvv(ptr, (const float*)gamma_data, (const float*)beta_data, eps, w * h, elempack);
#else
                Mat m = bottom_top_blob.channel(q);
                LayerNorm::forward_inplace(m, opt);
#endif
            }
        }
    }

    return 0;
}

} // namespace ncnn
