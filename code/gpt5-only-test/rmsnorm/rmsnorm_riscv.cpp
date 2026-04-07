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

#include "rmsnorm_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

#include "riscv_usability.h"

namespace ncnn {

RMSNorm_riscv::RMSNorm_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector
}

#if __riscv_vector
static void rmsnorm_rvv(float* ptr, const float* gamma_ptr, float eps, int elemcount, int elempack)
{
    const int size = elemcount * elempack;

    // accumulate sum of squares
    float sqsum = 0.f;
    int n = size;
    while (n > 0)
    {
        size_t vl = __riscv_vsetvl_e32m8(n);
        vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
        vfloat32m8_t _pp = __riscv_vfmul_vv_f32m8(_p, _p, vl);
        vfloat32m1_t _acc = __riscv_vfmv_s_f_f32m1(0.f, vl);
        _acc = __riscv_vfredusum_vs_f32m8_f32m1(_pp, _acc, vl);
        sqsum += __riscv_vfmv_f_s_f32m1_f32(_acc);
        ptr += vl;
        n -= vl;
    }

    float rms = 1.f / sqrtf(sqsum / elemcount + eps);

    // apply normalization and optional gamma
    n = size;
    ptr -= size; // rewind to start
    if (gamma_ptr)
    {
        if (elempack == 1)
        {
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e32m8(n);
                vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                vfloat32m8_t _g = __riscv_vle32_v_f32m8(gamma_ptr, vl);
                _p = __riscv_vfmul_vf_f32m8(_p, rms, vl);
                _p = __riscv_vfmul_vv_f32m8(_p, _g, vl);
                __riscv_vse32_v_f32m8(ptr, _p, vl);
                ptr += vl;
                gamma_ptr += vl;
                n -= vl;
            }
        }
        else
        {
            // gamma per-lane for packed layout: broadcast scalar gamma per vector of elempack
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e32m8(n);
                vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                _p = __riscv_vfmul_vf_f32m8(_p, rms, vl);

                // build gamma vector by repeating gamma per block of elempack
                // process vl elements: for each block of elempack, load one gamma scalar and broadcast
                // implement by scalar loop over vl with pack-aware stride fallback for portability
                std::vector<float> tmp(vl);
                __riscv_vse32_v_f32m8(tmp.data(), _p, vl);
                for (size_t i = 0; i < vl; i += elempack)
                {
                    float g = gamma_ptr[0];
                    for (int k = 0; k < elempack && (i + k) < vl; k++)
                        tmp[i + k] = tmp[i + k] * g;
                    gamma_ptr += 1;
                }
                __riscv_vse32_v_f32m8(ptr, __riscv_vle32_v_f32m8(tmp.data(), vl), vl);
                ptr += vl;
                n -= vl;
            }
        }
    }
    else
    {
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(n);
            vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
            _p = __riscv_vfmul_vf_f32m8(_p, rms, vl);
            __riscv_vse32_v_f32m8(ptr, _p, vl);
            ptr += vl;
            n -= vl;
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
        rmsnorm_rvv(ptr, gamma_data, eps, w * elempack, 1);
#else
        // fallback to generic
        return RMSNorm::forward_inplace(bottom_top_blob, opt);
#endif
    }

    if (dims == 2)
    {
#if __riscv_vector
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            float* ptr = bottom_top_blob.row(i);
            rmsnorm_rvv(ptr, gamma_data, eps, w, elempack);
        }
#else
        return RMSNorm::forward_inplace(bottom_top_blob, opt);
#endif
    }

    if (dims == 3)
    {
#if __riscv_vector
        if (affine_size == w)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                for (int i = 0; i < h; i++)
                {
                    float* ptr = bottom_top_blob.channel(q).row(i);
                    rmsnorm_rvv(ptr, gamma_data, eps, w, elempack);
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);
                rmsnorm_rvv(ptr, gamma_data, eps, w * h, elempack);
            }
        }
#else
        return RMSNorm::forward_inplace(bottom_top_blob, opt);
#endif
    }

    return 0;
}

} // namespace ncnn
