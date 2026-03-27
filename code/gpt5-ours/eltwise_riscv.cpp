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

#include "eltwise_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

#include "riscv_usability.h"

namespace ncnn {

Eltwise_riscv::Eltwise_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector
}

static inline void eltwise_prod_f32(const float* a, const float* b, float* out, int n)
{
#if __riscv_vector
    int remaining = n;
    while (remaining > 0)
    {
        size_t vl = __riscv_vsetvl_e32m1(remaining);
        vfloat32m1_t va = __riscv_vle32_v_f32m1(a, vl);
        vfloat32m1_t vb = __riscv_vle32_v_f32m1(b, vl);
        vfloat32m1_t vc = __riscv_vfmul_vv_f32m1(va, vb, vl);
        __riscv_vse32_v_f32m1(out, vc, vl);
        a += vl; b += vl; out += vl; remaining -= vl;
    }
#else
    for (int i = 0; i < n; i++) out[i] = a[i] * b[i];
#endif
}

static inline void eltwise_sum_f32(const float* a, const float* b, float* out, int n)
{
#if __riscv_vector
    int remaining = n;
    while (remaining > 0)
    {
        size_t vl = __riscv_vsetvl_e32m1(remaining);
        vfloat32m1_t va = __riscv_vle32_v_f32m1(a, vl);
        vfloat32m1_t vb = __riscv_vle32_v_f32m1(b, vl);
        vfloat32m1_t vc = __riscv_vfadd_vv_f32m1(va, vb, vl);
        __riscv_vse32_v_f32m1(out, vc, vl);
        a += vl; b += vl; out += vl; remaining -= vl;
    }
#else
    for (int i = 0; i < n; i++) out[i] = a[i] + b[i];
#endif
}

static inline void eltwise_sum_coeff2_f32(const float* a, const float* b, float* out, int n, float coeff0, float coeff1)
{
#if __riscv_vector
    int remaining = n;
    while (remaining > 0)
    {
        size_t vl = __riscv_vsetvl_e32m1(remaining);
        vfloat32m1_t va = __riscv_vle32_v_f32m1(a, vl);
        vfloat32m1_t vb = __riscv_vle32_v_f32m1(b, vl);
        va = __riscv_vfmul_vf_f32m1(va, coeff0, vl);
        vb = __riscv_vfmul_vf_f32m1(vb, coeff1, vl);
        vfloat32m1_t vc = __riscv_vfadd_vv_f32m1(va, vb, vl);
        __riscv_vse32_v_f32m1(out, vc, vl);
        a += vl; b += vl; out += vl; remaining -= vl;
    }
#else
    for (int i = 0; i < n; i++) out[i] = a[i] * coeff0 + b[i] * coeff1;
#endif
}

static inline void eltwise_sum_acc_f32(float* out, const float* b, int n)
{
#if __riscv_vector
    int remaining = n;
    while (remaining > 0)
    {
        size_t vl = __riscv_vsetvl_e32m1(remaining);
        vfloat32m1_t vo = __riscv_vle32_v_f32m1(out, vl);
        vfloat32m1_t vb = __riscv_vle32_v_f32m1(b, vl);
        vo = __riscv_vfadd_vv_f32m1(vo, vb, vl);
        __riscv_vse32_v_f32m1(out, vo, vl);
        b += vl; out += vl; remaining -= vl;
    }
#else
    for (int i = 0; i < n; i++) out[i] += b[i];
#endif
}

static inline void eltwise_sum_acc_coeff_f32(float* out, const float* b, int n, float coeff)
{
#if __riscv_vector
    int remaining = n;
    while (remaining > 0)
    {
        size_t vl = __riscv_vsetvl_e32m1(remaining);
        vfloat32m1_t vo = __riscv_vle32_v_f32m1(out, vl);
        vfloat32m1_t vb = __riscv_vle32_v_f32m1(b, vl);
        vb = __riscv_vfmul_vf_f32m1(vb, coeff, vl);
        vo = __riscv_vfadd_vv_f32m1(vo, vb, vl);
        __riscv_vse32_v_f32m1(out, vo, vl);
        b += vl; out += vl; remaining -= vl;
    }
#else
    for (int i = 0; i < n; i++) out[i] += b[i] * coeff;
#endif
}

static inline void eltwise_max_f32(const float* a, const float* b, float* out, int n)
{
#if __riscv_vector
    int remaining = n;
    while (remaining > 0)
    {
        size_t vl = __riscv_vsetvl_e32m1(remaining);
        vfloat32m1_t va = __riscv_vle32_v_f32m1(a, vl);
        vfloat32m1_t vb = __riscv_vle32_v_f32m1(b, vl);
        vfloat32m1_t vc = __riscv_vfmax_vv_f32m1(va, vb, vl);
        __riscv_vse32_v_f32m1(out, vc, vl);
        a += vl; b += vl; out += vl; remaining -= vl;
    }
#else
    for (int i = 0; i < n; i++) out[i] = a[i] > b[i] ? a[i] : b[i];
#endif
}

int Eltwise_riscv::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    int elempack = bottom_blob.elempack;
    int size = w * h * d * elempack;

    Mat& top_blob = top_blobs[0];
    top_blob.create_like(bottom_blob, opt.blob_allocator);
    if (top_blob.empty()) return -100;

    if (elempack == 1)
    {
        // fallback to generic implementation
        return Eltwise::forward(bottom_blobs, top_blobs, opt);
    }

    if (op_type == Operation_PROD)
    {
        const Mat& b1 = bottom_blobs[1];
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const float* a = bottom_blob.channel(q);
            const float* b = b1.channel(q);
            float* out = top_blob.channel(q);
            eltwise_prod_f32(a, b, out, size);
        }
        for (size_t b = 2; b < bottom_blobs.size(); b++)
        {
            const Mat& bi = bottom_blobs[b];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* p = bi.channel(q);
                float* out = top_blob.channel(q);
            #if __riscv_vector
                int remaining = size;
                while (remaining > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m1(remaining);
                    vfloat32m1_t vo = __riscv_vle32_v_f32m1(out, vl);
                    vfloat32m1_t vp = __riscv_vle32_v_f32m1(p, vl);
                    vo = __riscv_vfmul_vv_f32m1(vo, vp, vl);
                    __riscv_vse32_v_f32m1(out, vo, vl);
                    p += vl; out += vl; remaining -= vl;
                }
            #else
                for (int i = 0; i < size; i++) out[i] *= p[i];
            #endif
            }
        }
    }
    else if (op_type == Operation_SUM)
    {
        if (coeffs.w == 0)
        {
            const Mat& b1 = bottom_blobs[1];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* a = bottom_blob.channel(q);
                const float* b = b1.channel(q);
                float* out = top_blob.channel(q);
                eltwise_sum_f32(a, b, out, size);
            }
            for (size_t b = 2; b < bottom_blobs.size(); b++)
            {
                const Mat& bi = bottom_blobs[b];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const float* p = bi.channel(q);
                    float* out = top_blob.channel(q);
                    eltwise_sum_acc_f32(out, p, size);
                }
            }
        }
        else
        {
            const Mat& b1 = bottom_blobs[1];
            float coeff0 = coeffs[0];
            float coeff1 = coeffs[1];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* a = bottom_blob.channel(q);
                const float* b = b1.channel(q);
                float* out = top_blob.channel(q);
                eltwise_sum_coeff2_f32(a, b, out, size, coeff0, coeff1);
            }
            for (size_t b = 2; b < bottom_blobs.size(); b++)
            {
                const Mat& bi = bottom_blobs[b];
                float coeff = coeffs[b];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const float* p = bi.channel(q);
                    float* out = top_blob.channel(q);
                    eltwise_sum_acc_coeff_f32(out, p, size, coeff);
                }
            }
        }
    }
    else if (op_type == Operation_MAX)
    {
        const Mat& b1 = bottom_blobs[1];
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const float* a = bottom_blob.channel(q);
            const float* b = b1.channel(q);
            float* out = top_blob.channel(q);
            eltwise_max_f32(a, b, out, size);
        }
        for (size_t b = 2; b < bottom_blobs.size(); b++)
        {
            const Mat& bi = bottom_blobs[b];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* p = bi.channel(q);
                float* out = top_blob.channel(q);
            #if __riscv_vector
                int remaining = size;
                while (remaining > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m1(remaining);
                    vfloat32m1_t vo = __riscv_vle32_v_f32m1(out, vl);
                    vfloat32m1_t vp = __riscv_vle32_v_f32m1(p, vl);
                    vo = __riscv_vfmax_vv_f32m1(vo, vp, vl);
                    __riscv_vse32_v_f32m1(out, vo, vl);
                    p += vl; out += vl; remaining -= vl;
                }
            #else
                for (int i = 0; i < size; i++) out[i] = std::max(out[i], p[i]);
            #endif
            }
        }
    }

    return 0;
}

} // namespace ncnn
