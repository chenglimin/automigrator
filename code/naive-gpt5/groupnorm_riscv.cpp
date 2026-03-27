// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2023 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "riscv_usability.h"

namespace ncnn {

GroupNorm_riscv::GroupNorm_riscv()
{
#if __riscv_vector
    // groupnorm solved in normal dimension, but rvv can still accelerate
    support_packing = false;
#else
    support_packing = false;
#endif
}

int GroupNorm_riscv::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    const int dims = bottom_top_blob.dims;
    const int channels_per_group = channels / group;

#if !__riscv_vector
    // Fallback to reference implementation when rvv not available
    return GroupNorm::forward_inplace(bottom_top_blob, opt);
#else
    if (dims == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int g = 0; g < group; g++)
        {
            Mat bottom_top_blob_g = bottom_top_blob.range(g * channels_per_group, channels_per_group);
            const Mat gamma_data_g = gamma_data.range(g * channels_per_group, channels_per_group);
            const Mat beta_data_g = beta_data.range(g * channels_per_group, channels_per_group);

            float* ptr = bottom_top_blob_g;

            // sum
            float sum = 0.f;
            {
                int n = channels_per_group;
                vfloat32m1_t _sum = __riscv_vfmv_s_f_f32m1(0.f, __riscv_vsetvlmax_e32m1());
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m8(n);
                    vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                    _sum = __riscv_vfredusum_vs_f32m8_f32m1(_p, _sum, vl);
                    ptr += vl;
                    n -= vl;
                }
                sum = __riscv_vfmv_f_s_f32m1_f32(_sum);
            }

            float mean = sum / channels_per_group;

            // sqsum
            float sqsum = 0.f;
            ptr = bottom_top_blob_g;
            {
                int n = channels_per_group;
                vfloat32m1_t _sqsum = __riscv_vfmv_s_f_f32m1(0.f, __riscv_vsetvlmax_e32m1());
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m8(n);
                    vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                    _p = __riscv_vfsub_vf_f32m8(_p, mean, vl);
                    _sqsum = __riscv_vfredosum_vs_f32m8_f32m1(__riscv_vfmul_vv_f32m8(_p, _p, vl), _sqsum, vl);
                    ptr += vl;
                    n -= vl;
                }
                sqsum = __riscv_vfmv_f_s_f32m1_f32(_sqsum);
            }

            float scale1 = 1.f / sqrtf(sqsum / channels_per_group + eps);
            float scale2 = -mean * scale1;

            ptr = bottom_top_blob_g;
            if (affine)
            {
                const float* gamma = gamma_data_g;
                const float* beta = beta_data_g;
                int n = channels_per_group;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m8(n);
                    vfloat32m8_t _gamma = __riscv_vle32_v_f32m8(gamma, vl);
                    vfloat32m8_t _beta = __riscv_vle32_v_f32m8(beta, vl);
                    vfloat32m8_t _a = __riscv_vfmul_vf_f32m8(_gamma, scale1, vl);
                    vfloat32m8_t _b = __riscv_vfadd_vv_f32m8(__riscv_vfmul_vf_f32m8(_gamma, scale2, vl), _beta, vl);
                    vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                    _p = __riscv_vfadd_vv_f32m8(__riscv_vfmul_vv_f32m8(_p, _a, vl), _b, vl);
                    __riscv_vse32_v_f32m8(ptr, _p, vl);
                    gamma += vl;
                    beta += vl;
                    ptr += vl;
                    n -= vl;
                }
            }
            else
            {
                int n = channels_per_group;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m8(n);
                    vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                    _p = __riscv_vfadd_vf_f32m8(__riscv_vfmul_vf_f32m8(_p, scale1, vl), scale2, vl);
                    __riscv_vse32_v_f32m8(ptr, _p, vl);
                    ptr += vl;
                    n -= vl;
                }
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

            // sum
            float sum = 0.f;
            {
                int n = size;
                vfloat32m1_t _sum = __riscv_vfmv_s_f_f32m1(0.f, __riscv_vsetvlmax_e32m1());
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m8(n);
                    vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                    _sum = __riscv_vfredusum_vs_f32m8_f32m1(_p, _sum, vl);
                    ptr += vl;
                    n -= vl;
                }
                sum = __riscv_vfmv_f_s_f32m1_f32(_sum);
            }

            float mean = sum / size;

            // sqsum
            float sqsum = 0.f;
            ptr = bottom_top_blob_g;
            {
                int n = size;
                vfloat32m1_t _sqsum = __riscv_vfmv_s_f_f32m1(0.f, __riscv_vsetvlmax_e32m1());
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m8(n);
                    vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                    _p = __riscv_vfsub_vf_f32m8(_p, mean, vl);
                    _sqsum = __riscv_vfredosum_vs_f32m8_f32m1(__riscv_vfmul_vv_f32m8(_p, _p, vl), _sqsum, vl);
                    ptr += vl;
                    n -= vl;
                }
                sqsum = __riscv_vfmv_f_s_f32m1_f32(_sqsum);
            }

            float scale1 = 1.f / sqrtf(sqsum / size + eps);
            float scale2 = -mean * scale1;

            ptr = bottom_top_blob_g;
            if (affine)
            {
                const float* gamma = gamma_data_g;
                const float* beta = beta_data_g;
                for (int q = 0; q < channels_per_group; q++)
                {
                    float a = gamma[q] * scale1;
                    float b = gamma[q] * scale2 + beta[q];

                    float* rowptr = bottom_top_blob_g.row(q);
                    int n = w;
                    while (n > 0)
                    {
                        size_t vl = __riscv_vsetvl_e32m8(n);
                        vfloat32m8_t _p = __riscv_vle32_v_f32m8(rowptr, vl);
                        _p = __riscv_vfadd_vf_f32m8(__riscv_vfmul_vf_f32m8(_p, a, vl), b, vl);
                        __riscv_vse32_v_f32m8(rowptr, _p, vl);
                        rowptr += vl;
                        n -= vl;
                    }
                }
            }
            else
            {
                int n = size;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m8(n);
                    vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                    _p = __riscv_vfadd_vf_f32m8(__riscv_vfmul_vf_f32m8(_p, scale1, vl), scale2, vl);
                    __riscv_vse32_v_f32m8(ptr, _p, vl);
                    ptr += vl;
                    n -= vl;
                }
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

            // mean
            float sum = 0.f;
            for (int q = 0; q < channels_per_group; q++)
            {
                const float* ptr = bottom_top_blob_g.channel(q);
                int n = size;
                vfloat32m1_t _sum = __riscv_vfmv_s_f_f32m1(0.f, __riscv_vsetvlmax_e32m1());
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m8(n);
                    vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                    _sum = __riscv_vfredusum_vs_f32m8_f32m1(_p, _sum, vl);
                    ptr += vl;
                    n -= vl;
                }
                sum += __riscv_vfmv_f_s_f32m1_f32(_sum);
            }
            float mean = sum / (channels_per_group * size);

            // sqsum
            float sqsum = 0.f;
            for (int q = 0; q < channels_per_group; q++)
            {
                const float* ptr = bottom_top_blob_g.channel(q);
                int n = size;
                vfloat32m1_t _sqsum = __riscv_vfmv_s_f_f32m1(0.f, __riscv_vsetvlmax_e32m1());
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m8(n);
                    vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                    _p = __riscv_vfsub_vf_f32m8(_p, mean, vl);
                    _sqsum = __riscv_vfredosum_vs_f32m8_f32m1(__riscv_vfmul_vv_f32m8(_p, _p, vl), _sqsum, vl);
                    ptr += vl;
                    n -= vl;
                }
                sqsum += __riscv_vfmv_f_s_f32m1_f32(_sqsum);
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

                float* ptr = bottom_top_blob_g.channel(q);
                int n = size;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m8(n);
                    vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                    _p = __riscv_vfadd_vf_f32m8(__riscv_vfmul_vf_f32m8(_p, a, vl), b, vl);
                    __riscv_vse32_v_f32m8(ptr, _p, vl);
                    ptr += vl;
                    n -= vl;
                }
            }
        }
        return 0;
    }

    return 0;
#endif // __riscv_vector
}

} // namespace ncnn
