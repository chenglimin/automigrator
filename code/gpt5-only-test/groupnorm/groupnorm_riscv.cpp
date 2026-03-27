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

namespace ncnn {

GroupNorm_riscv::GroupNorm_riscv()
{
    // follow x86 implementation behavior
    support_packing = false;
}

int GroupNorm_riscv::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
#if __riscv_vector
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

            // mean
            float sum = 0.f;
            float* ptr = bottom_top_blob_g;
            int n = channels_per_group;
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e32m8(n);
                vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                vfloat32m1_t _acc = __riscv_vfmv_s_f_f32m1(0.f, vl);
                _acc = __riscv_vfredusum_vs_f32m8_f32m1(_p, _acc, vl);
                sum += __riscv_vfmv_f_s_f32m1_f32(_acc);
                ptr += vl;
                n -= vl;
            }
            float mean = sum / channels_per_group;

            // variance
            float sqsum = 0.f;
            ptr = bottom_top_blob_g;
            n = channels_per_group;
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e32m8(n);
                vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                _p = __riscv_vfsub_vf_f32m8(_p, mean, vl);
                vfloat32m8_t _p2 = __riscv_vfmul_vv_f32m8(_p, _p, vl);
                vfloat32m1_t _acc = __riscv_vfmv_s_f_f32m1(0.f, vl);
                _acc = __riscv_vfredusum_vs_f32m8_f32m1(_p2, _acc, vl);
                sqsum += __riscv_vfmv_f_s_f32m1_f32(_acc);
                ptr += vl;
                n -= vl;
            }

            float scale1 = 1.f / sqrtf(sqsum / channels_per_group + eps);
            float scale2 = -mean * scale1;

            ptr = bottom_top_blob_g;
            if (affine)
            {
                const float* gamma = gamma_data_g;
                const float* beta = beta_data_g;
                n = channels_per_group;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m8(n);
                    vfloat32m8_t _gamma = __riscv_vle32_v_f32m8(gamma, vl);
                    vfloat32m8_t _beta = __riscv_vle32_v_f32m8(beta, vl);
                    vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);

                    vfloat32m8_t _a = __riscv_vfmul_vf_f32m8(_gamma, scale1, vl);
                    vfloat32m8_t _b = __riscv_vfadd_vv_f32m8(__riscv_vfmul_vf_f32m8(_gamma, scale2, vl), _beta, vl);
                    vfloat32m8_t _out = __riscv_vfadd_vv_f32m8(__riscv_vfmul_vv_f32m8(_p, _a, vl), _b, vl);

                    __riscv_vse32_v_f32m8(ptr, _out, vl);
                    gamma += vl;
                    beta += vl;
                    ptr += vl;
                    n -= vl;
                }
            }
            else
            {
                n = channels_per_group;
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
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int g = 0; g < group; g++)
        {
            Mat bottom_top_blob_g = bottom_top_blob.row_range(g * channels_per_group, channels_per_group);
            const Mat gamma_data_g = gamma_data.range(g * channels_per_group, channels_per_group);
            const Mat beta_data_g = beta_data.range(g * channels_per_group, channels_per_group);

            // mean
            float sum = 0.f;
            for (int q = 0; q < channels_per_group; q++)
            {
                float* ptr = bottom_top_blob_g.row(q);
                int n = w;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m8(n);
                    vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                    vfloat32m1_t _acc = __riscv_vfmv_s_f_f32m1(0.f, vl);
                    _acc = __riscv_vfredusum_vs_f32m8_f32m1(_p, _acc, vl);
                    sum += __riscv_vfmv_f_s_f32m1_f32(_acc);
                    ptr += vl;
                    n -= vl;
                }
            }
            float mean = sum / (channels_per_group * w);

            // variance
            float sqsum = 0.f;
            for (int q = 0; q < channels_per_group; q++)
            {
                float* ptr = bottom_top_blob_g.row(q);
                int n = w;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m8(n);
                    vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                    _p = __riscv_vfsub_vf_f32m8(_p, mean, vl);
                    vfloat32m8_t _p2 = __riscv_vfmul_vv_f32m8(_p, _p, vl);
                    vfloat32m1_t _acc = __riscv_vfmv_s_f_f32m1(0.f, vl);
                    _acc = __riscv_vfredusum_vs_f32m8_f32m1(_p2, _acc, vl);
                    sqsum += __riscv_vfmv_f_s_f32m1_f32(_acc);
                    ptr += vl;
                    n -= vl;
                }
            }

            float scale1 = 1.f / sqrtf(sqsum / (channels_per_group * w) + eps);
            float scale2 = -mean * scale1;

            if (affine)
            {
                const float* gamma = gamma_data_g;
                const float* beta = beta_data_g;
                for (int q = 0; q < channels_per_group; q++)
                {
                    float a = *gamma * scale1;
                    float b = *gamma * scale2 + *beta;
                    float* ptr = bottom_top_blob_g.row(q);
                    int n = w;
                    while (n > 0)
                    {
                        size_t vl = __riscv_vsetvl_e32m8(n);
                        vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                        vfloat32m8_t _a = __riscv_vfmv_v_f_f32m8(a, vl);
                        vfloat32m8_t _b = __riscv_vfmv_v_f_f32m8(b, vl);
                        _p = __riscv_vfadd_vv_f32m8(__riscv_vfmul_vv_f32m8(_p, _a, vl), _b, vl);
                        __riscv_vse32_v_f32m8(ptr, _p, vl);
                        ptr += vl;
                        n -= vl;
                    }
                    gamma++;
                    beta++;
                }
            }
            else
            {
                for (int q = 0; q < channels_per_group; q++)
                {
                    float* ptr = bottom_top_blob_g.row(q);
                    int n = w;
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
                float* ptr = bottom_top_blob_g.channel(q);
                int n = size;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m8(n);
                    vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                    vfloat32m1_t _acc = __riscv_vfmv_s_f_f32m1(0.f, vl);
                    _acc = __riscv_vfredusum_vs_f32m8_f32m1(_p, _acc, vl);
                    sum += __riscv_vfmv_f_s_f32m1_f32(_acc);
                    ptr += vl;
                    n -= vl;
                }
            }
            float mean = sum / (channels_per_group * size);

            // variance
            float sqsum = 0.f;
            for (int q = 0; q < channels_per_group; q++)
            {
                float* ptr = bottom_top_blob_g.channel(q);
                int n = size;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m8(n);
                    vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                    _p = __riscv_vfsub_vf_f32m8(_p, mean, vl);
                    vfloat32m8_t _p2 = __riscv_vfmul_vv_f32m8(_p, _p, vl);
                    vfloat32m1_t _acc = __riscv_vfmv_s_f_f32m1(0.f, vl);
                    _acc = __riscv_vfredusum_vs_f32m8_f32m1(_p2, _acc, vl);
                    sqsum += __riscv_vfmv_f_s_f32m1_f32(_acc);
                    ptr += vl;
                    n -= vl;
                }
            }

            float scale1 = 1.f / sqrtf(sqsum / (channels_per_group * size) + eps);
            float scale2 = -mean * scale1;

            if (affine)
            {
                const float* gamma = gamma_data_g;
                const float* beta = beta_data_g;
                for (int q = 0; q < channels_per_group; q++)
                {
                    float a = *gamma * scale1;
                    float b = *gamma * scale2 + *beta;
                    float* ptr = bottom_top_blob_g.channel(q);
                    int n = size;
                    while (n > 0)
                    {
                        size_t vl = __riscv_vsetvl_e32m8(n);
                        vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                        vfloat32m8_t _a = __riscv_vfmv_v_f_f32m8(a, vl);
                        vfloat32m8_t _b = __riscv_vfmv_v_f_f32m8(b, vl);
                        _p = __riscv_vfadd_vv_f32m8(__riscv_vfmul_vv_f32m8(_p, _a, vl), _b, vl);
                        __riscv_vse32_v_f32m8(ptr, _p, vl);
                        ptr += vl;
                        n -= vl;
                    }
                    gamma++;
                    beta++;
                }
            }
            else
            {
                for (int q = 0; q < channels_per_group; q++)
                {
                    float* ptr = bottom_top_blob_g.channel(q);
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
        }
        return 0;
    }

    return 0;
#else  // __riscv_vector
    // fallback to reference implementation
    return GroupNorm::forward_inplace(bottom_top_blob, opt);
#endif // __riscv_vector
}

} // namespace ncnn
