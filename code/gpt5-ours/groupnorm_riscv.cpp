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

#include "groupnorm_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

#include <math.h>

namespace ncnn {

GroupNorm_riscv::GroupNorm_riscv()
{
    // groupnorm solved in packing dimension; keep consistent behavior
    support_packing = false;
}

static inline float rvv_sum_f32(const float* ptr, int n)
{
#if __riscv_vector
    float sum = 0.f;
    while (n > 0)
    {
        size_t vl = __riscv_vsetvl_e32m8(n);
        vfloat32m8_t v = __riscv_vle32_v_f32m8(ptr, vl);
        vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.f, vl);
        acc = __riscv_vfredusum_vs_f32m8_f32m1(v, acc, vl);
        sum += __riscv_vfmv_f_s_f32m1_f32(acc);
        ptr += vl;
        n -= (int)vl;
    }
    return sum;
#else
    float sum = 0.f;
    for (int i = 0; i < n; i++) sum += ptr[i];
    return sum;
#endif
}

int GroupNorm_riscv::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    const int dims = bottom_top_blob.dims;
    const int channels_per_group = channels / group;

    if (dims == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int g = 0; g < group; g++)
        {
            Mat bottom_top_blob_g = bottom_top_blob.range(g * channels_per_group, channels_per_group);
            const Mat gamma_data_g = gamma_data.range(g * channels_per_group, channels_per_group);
            const Mat beta_data_g = beta_data.range(g * channels_per_group, channels_per_group);

            float* ptr = bottom_top_blob_g;
            float sum = rvv_sum_f32(ptr, channels_per_group);
            float mean = sum / channels_per_group;

            // variance
            float sqsum = 0.f;
#if __riscv_vector
            {
                int n = channels_per_group;
                const float* rptr = bottom_top_blob_g;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m8(n);
                    vfloat32m8_t v = __riscv_vle32_v_f32m8(rptr, vl);
                    vfloat32m8_t vmean = __riscv_vfmv_v_f_f32m8(mean, vl);
                    vfloat32m8_t vd = __riscv_vfsub_vv_f32m8(v, vmean, vl);
                    vfloat32m8_t vd2 = __riscv_vfmul_vv_f32m8(vd, vd, vl);
                    vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.f, vl);
                    acc = __riscv_vfredusum_vs_f32m8_f32m1(vd2, acc, vl);
                    sqsum += __riscv_vfmv_f_s_f32m1_f32(acc);
                    rptr += vl;
                    n -= (int)vl;
                }
            }
#else
            for (int i = 0; i < channels_per_group; i++)
            {
                float tmp = ptr[i] - mean;
                sqsum += tmp * tmp;
            }
#endif
            float scale1 = 1.f / sqrtf(sqsum / channels_per_group + eps);
            float scale2 = -mean * scale1;

            if (affine)
            {
                const float* gamma = gamma_data_g;
                const float* beta = beta_data_g;
                int n = channels_per_group;
                float* wptr = bottom_top_blob_g;
#if __riscv_vector
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m8(n);
                    vfloat32m8_t vgamma = __riscv_vle32_v_f32m8(gamma, vl);
                    vfloat32m8_t vbeta = __riscv_vle32_v_f32m8(beta, vl);
                    vfloat32m8_t vp = __riscv_vle32_v_f32m8(wptr, vl);
                    vfloat32m8_t vscale1 = __riscv_vfmv_v_f_f32m8(scale1, vl);
                    vfloat32m8_t vscale2 = __riscv_vfmv_v_f_f32m8(scale2, vl);
                    vfloat32m8_t va = __riscv_vfmul_vv_f32m8(vgamma, vscale1, vl);
                    vfloat32m8_t vb = __riscv_vfadd_vv_f32m8(__riscv_vfmul_vv_f32m8(vgamma, vscale2, vl), vbeta, vl);
                    // vp = vp * va + vb
                    vp = __riscv_vfmadd_vv_f32m8(vp, va, vb, vl);
                    __riscv_vse32_v_f32m8(wptr, vp, vl);
                    gamma += vl;
                    beta += vl;
                    wptr += vl;
                    n -= (int)vl;
                }
#else
                for (int i = 0; i < channels_per_group; i++)
                {
                    float a = gamma[i] * scale1;
                    float b = gamma[i] * scale2 + beta[i];
                    wptr[i] = wptr[i] * a + b;
                }
#endif
            }
            else
            {
                int n = channels_per_group;
                float* wptr = bottom_top_blob_g;
#if __riscv_vector
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m8(n);
                    vfloat32m8_t vp = __riscv_vle32_v_f32m8(wptr, vl);
                    vfloat32m8_t vscale1 = __riscv_vfmv_v_f_f32m8(scale1, vl);
                    vfloat32m8_t vscale2 = __riscv_vfmv_v_f_f32m8(scale2, vl);
                    vp = __riscv_vfmadd_vv_f32m8(vp, vscale1, vscale2, vl);
                    __riscv_vse32_v_f32m8(wptr, vp, vl);
                    wptr += vl;
                    n -= (int)vl;
                }
#else
                for (int i = 0; i < channels_per_group; i++)
                {
                    wptr[i] = wptr[i] * scale1 + scale2;
                }
#endif
            }
        }

        return 0;
    }

    if (dims == 2)
    {
        int w = bottom_top_blob.w;
        int size = channels_per_group * w;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int g = 0; g < group; g++)
        {
            Mat bottom_top_blob_g = bottom_top_blob.row_range(g * channels_per_group, channels_per_group);
            const Mat gamma_data_g = gamma_data.range(g * channels_per_group, channels_per_group);
            const Mat beta_data_g = beta_data.range(g * channels_per_group, channels_per_group);

            float* ptr = bottom_top_blob_g;
            float sum = rvv_sum_f32(ptr, size);
            float mean = sum / size;

            float sqsum = 0.f;
#if __riscv_vector
            {
                int n = size;
                const float* rptr = bottom_top_blob_g;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m8(n);
                    vfloat32m8_t v = __riscv_vle32_v_f32m8(rptr, vl);
                    vfloat32m8_t vmean = __riscv_vfmv_v_f_f32m8(mean, vl);
                    vfloat32m8_t vd = __riscv_vfsub_vv_f32m8(v, vmean, vl);
                    vfloat32m8_t vd2 = __riscv_vfmul_vv_f32m8(vd, vd, vl);
                    vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.f, vl);
                    acc = __riscv_vfredusum_vs_f32m8_f32m1(vd2, acc, vl);
                    sqsum += __riscv_vfmv_f_s_f32m1_f32(acc);
                    rptr += vl;
                    n -= (int)vl;
                }
            }
#else
            for (int i = 0; i < size; i++)
            {
                float tmp = ptr[i] - mean;
                sqsum += tmp * tmp;
            }
#endif
            float scale1 = 1.f / sqrtf(sqsum / size + eps);
            float scale2 = -mean * scale1;

            float* wptr = bottom_top_blob_g;
            if (affine)
            {
                const float* gamma = gamma_data_g;
                const float* beta = beta_data_g;
                for (int q = 0; q < channels_per_group; q++)
                {
                    float a = gamma[q] * scale1;
                    float b = gamma[q] * scale2 + beta[q];
#if __riscv_vector
                    int n = w;
                    float* cptr = wptr + q * w;
                    while (n > 0)
                    {
                        size_t vl = __riscv_vsetvl_e32m8(n);
                        vfloat32m8_t vp = __riscv_vle32_v_f32m8(cptr, vl);
                        vfloat32m8_t va = __riscv_vfmv_v_f_f32m8(a, vl);
                        vfloat32m8_t vb = __riscv_vfmv_v_f_f32m8(b, vl);
                        vp = __riscv_vfmadd_vv_f32m8(vp, va, vb, vl);
                        __riscv_vse32_v_f32m8(cptr, vp, vl);
                        cptr += vl;
                        n -= (int)vl;
                    }
#else
                    float* cptr = wptr + q * w;
                    for (int i = 0; i < w; i++) cptr[i] = cptr[i] * a + b;
#endif
                }
            }
            else
            {
#if __riscv_vector
                int n = size;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m8(n);
                    vfloat32m8_t vp = __riscv_vle32_v_f32m8(wptr, vl);
                    vfloat32m8_t vscale1 = __riscv_vfmv_v_f_f32m8(scale1, vl);
                    vfloat32m8_t vscale2 = __riscv_vfmv_v_f_f32m8(scale2, vl);
                    vp = __riscv_vfmadd_vv_f32m8(vp, vscale1, vscale2, vl);
                    __riscv_vse32_v_f32m8(wptr, vp, vl);
                    wptr += vl;
                    n -= (int)vl;
                }
#else
                for (int i = 0; i < size; i++) wptr[i] = wptr[i] * scale1 + scale2;
#endif
            }
        }

        return 0;
    }

    if (dims == 3 || dims == 4)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int d = bottom_top_blob.d;
        int size = w * h * d;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int g = 0; g < group; g++)
        {
            Mat bottom_top_blob_g = bottom_top_blob.channel_range(g * channels_per_group, channels_per_group);
            const Mat gamma_data_g = gamma_data.range(g * channels_per_group, channels_per_group);
            const Mat beta_data_g = beta_data.range(g * channels_per_group, channels_per_group);

            float sum = 0.f;
            for (int q = 0; q < channels_per_group; q++)
            {
                const float* cptr = bottom_top_blob_g.channel(q);
                sum += rvv_sum_f32(cptr, size);
            }
            float mean = sum / (channels_per_group * size);

            float sqsum = 0.f;
            for (int q = 0; q < channels_per_group; q++)
            {
                const float* cptr = bottom_top_blob_g.channel(q);
#if __riscv_vector
                int n = size;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m8(n);
                    vfloat32m8_t v = __riscv_vle32_v_f32m8(cptr, vl);
                    vfloat32m8_t vmean = __riscv_vfmv_v_f_f32m8(mean, vl);
                    vfloat32m8_t vd = __riscv_vfsub_vv_f32m8(v, vmean, vl);
                    vfloat32m8_t vd2 = __riscv_vfmul_vv_f32m8(vd, vd, vl);
                    vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.f, vl);
                    acc = __riscv_vfredusum_vs_f32m8_f32m1(vd2, acc, vl);
                    sqsum += __riscv_vfmv_f_s_f32m1_f32(acc);
                    cptr += vl;
                    n -= (int)vl;
                }
#else
                for (int i = 0; i < size; i++)
                {
                    float tmp = cptr[i] - mean;
                    sqsum += tmp * tmp;
                }
#endif
            }

            float scale1 = 1.f / sqrtf(sqsum / (channels_per_group * size) + eps);
            float scale2 = -mean * scale1;

            const float* gamma = gamma_data_g;
            const float* beta = beta_data_g;
            for (int q = 0; q < channels_per_group; q++)
            {
                float a = scale1;
                float b = scale2;
                if (affine)
                {
                    a = gamma[q] * a;
                    b = gamma[q] * b + beta[q];
                }

                float* wptr = bottom_top_blob_g.channel(q);
#if __riscv_vector
                int n = size;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m8(n);
                    vfloat32m8_t vp = __riscv_vle32_v_f32m8(wptr, vl);
                    vfloat32m8_t va = __riscv_vfmv_v_f_f32m8(a, vl);
                    vfloat32m8_t vb = __riscv_vfmv_v_f_f32m8(b, vl);
                    vp = __riscv_vfmadd_vv_f32m8(vp, va, vb, vl);
                    __riscv_vse32_v_f32m8(wptr, vp, vl);
                    wptr += vl;
                    n -= (int)vl;
                }
#else
                for (int i = 0; i < size; i++) wptr[i] = wptr[i] * a + b;
#endif
            }
        }

        return 0;
    }

    return 0;
}

} // namespace ncnn
