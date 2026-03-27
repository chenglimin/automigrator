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
#endif // __riscv_vector

#include "riscv_usability.h"

namespace ncnn {

RMSNorm_riscv::RMSNorm_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector
}

static void rmsnorm(float* ptr, const float* gamma_ptr, float eps, int elemcount, int elempack)
{
    const int size = elemcount * elempack;

#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
    const size_t vl_packn = __riscv_vsetvl_e32m1(packn);
#endif // __riscv_vector

#if __riscv_vector
    vfloat32m1_t _rms_v = __riscv_vfmv_v_f_f32m1(0.f, vl_packn);
#endif
    float rms = 0.f;
    {
        const float* ptr0 = ptr;
        int i = 0;
#if __riscv_vector
        if (elempack == packn)
        {
            // accumulate sum of squares per lane
            for (int e = 0; e < elemcount; e++)
            {
                vfloat32m1_t _p = __riscv_vle32_v_f32m1(ptr0 + e * packn, vl_packn);
                _rms_v = __riscv_vfmacc_vv_f32m1(_rms_v, _p, _p, vl_packn);
            }
        }
        else
#endif // __riscv_vector
        {
            for (; i < size; i++)
            {
                rms += ptr0[0] * ptr0[0];
                ptr0++;
            }
        }
    }

#if __riscv_vector
    if (elempack == packn)
    {
        // rms_v = 1 / sqrt( _rms_v / elemcount + eps )
        vfloat32m1_t _elemcount = __riscv_vfmv_v_f_f32m1((float)elemcount, vl_packn);
        vfloat32m1_t _eps = __riscv_vfmv_v_f_f32m1(eps, vl_packn);
        _rms_v = __riscv_vfdiv_vv_f32m1(_rms_v, _elemcount, vl_packn);
        _rms_v = __riscv_vfadd_vv_f32m1(_rms_v, _eps, vl_packn);
        vfloat32m1_t _sqrt = __riscv_vfsqrt_v_f32m1(_rms_v, vl_packn);
        _rms_v = __riscv_vfrdiv_vf_f32m1(_sqrt, 1.f, vl_packn);
    }
#endif // __riscv_vector
    if (elempack == 1)
    {
        // rms has been accumulated as sum of squares in scalar path above
        rms = 1.f / sqrtf(rms / elemcount + eps);
#if __riscv_vector
        _rms_v = __riscv_vfmv_v_f_f32m1(rms, vl_packn);
#endif // __riscv_vector
    }

    if (gamma_ptr)
    {
        int i = 0;
#if __riscv_vector
        if (elempack == packn)
        {
            // apply per-pack gamma scalar
            for (int e = 0; e < elemcount; e++)
            {
                vfloat32m1_t _p = __riscv_vle32_v_f32m1(ptr + e * packn, vl_packn);
                vfloat32m1_t _gamma = __riscv_vfmv_v_f_f32m1(gamma_ptr[e], vl_packn);
                _p = __riscv_vfmul_vv_f32m1(_p, _rms_v, vl_packn);
                _p = __riscv_vfmul_vv_f32m1(_p, _gamma, vl_packn);
                __riscv_vse32_v_f32m1(ptr + e * packn, _p, vl_packn);
            }
            return;
        }
#endif // __riscv_vector
        for (; i < size; i++)
        {
            ptr[0] = (ptr[0] * rms) * gamma_ptr[0];
            ptr++;
            gamma_ptr++;
        }
    }
    else
    {
        int i = 0;
#if __riscv_vector
        if (elempack == packn)
        {
            for (int e = 0; e < elemcount; e++)
            {
                vfloat32m1_t _p = __riscv_vle32_v_f32m1(ptr + e * packn, vl_packn);
                _p = __riscv_vfmul_vv_f32m1(_p, _rms_v, vl_packn);
                __riscv_vse32_v_f32m1(ptr + e * packn, _p, vl_packn);
            }
            return;
        }
#endif // __riscv_vector
        for (; i < size; i++)
        {
            ptr[0] = ptr[0] * rms;
            ptr++;
        }
    }
}

int RMSNorm_riscv::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    const int dims = bottom_top_blob.dims;
    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int channels = bottom_top_blob.c;
    const int elempack = bottom_top_blob.elempack;

    if (dims == 1)
    {
        // assert affine_size == w
        float* ptr = bottom_top_blob;
        rmsnorm(ptr, gamma_data, eps, w * elempack, 1);
    }

    if (dims == 2)
    {
        // assert affine_size == w
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            float* ptr = bottom_top_blob.row(i);
            rmsnorm(ptr, gamma_data, eps, w, elempack);
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
                    rmsnorm(ptr, gamma_data, eps, w, elempack);
                }
            }
        }
        else // if (affine_size == w * h)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);
                rmsnorm(ptr, gamma_data, eps, w * h, elempack);
            }
        }
    }

    return 0;
}

} // namespace ncnn
