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

#include "eltwise_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#include "riscv_usability.h"
#endif // __riscv_vector

namespace ncnn {

Eltwise_riscv::Eltwise_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector
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
    if (top_blob.empty())
        return -100;

    if (op_type == Operation_PROD)
    {
        const Mat& bottom_blob1 = bottom_blobs[1];
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            const float* ptr1 = bottom_blob1.channel(q);
            float* outptr = top_blob.channel(q);

#if __riscv_vector
            int n = size;
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e32m8(n);
                vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                vfloat32m8_t _p1 = __riscv_vle32_v_f32m8(ptr1, vl);
                vfloat32m8_t _res = __riscv_vfmul_vv_f32m8(_p, _p1, vl);
                __riscv_vse32_v_f32m8(outptr, _res, vl);
                ptr += vl; ptr1 += vl; outptr += vl; n -= vl;
            }
#else
            for (int i = 0; i < size; i++)
            {
                outptr[i] = ptr[i] * ptr1[i];
            }
#endif
        }

        for (size_t b = 2; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob2 = bottom_blobs[b];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob2.channel(q);
                float* outptr = top_blob.channel(q);

#if __riscv_vector
                int n = size;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m8(n);
                    vfloat32m8_t _p = __riscv_vle32_v_f32m8(outptr, vl);
                    vfloat32m8_t _p1 = __riscv_vle32_v_f32m8(ptr, vl);
                    _p = __riscv_vfmul_vv_f32m8(_p, _p1, vl);
                    __riscv_vse32_v_f32m8(outptr, _p, vl);
                    ptr += vl; outptr += vl; n -= vl;
                }
#else
                for (int i = 0; i < size; i++)
                {
                    outptr[i] *= ptr[i];
                }
#endif
            }
        }
    }
    else if (op_type == Operation_SUM)
    {
        if (coeffs.w == 0)
        {
            const Mat& bottom_blob1 = bottom_blobs[1];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                const float* ptr1 = bottom_blob1.channel(q);
                float* outptr = top_blob.channel(q);

#if __riscv_vector
                int n = size;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m8(n);
                    vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                    vfloat32m8_t _p1 = __riscv_vle32_v_f32m8(ptr1, vl);
                    vfloat32m8_t _res = __riscv_vfadd_vv_f32m8(_p, _p1, vl);
                    __riscv_vse32_v_f32m8(outptr, _res, vl);
                    ptr += vl; ptr1 += vl; outptr += vl; n -= vl;
                }
#else
                for (int i = 0; i < size; i++)
                {
                    outptr[i] = ptr[i] + ptr1[i];
                }
#endif
            }

            for (size_t b = 2; b < bottom_blobs.size(); b++)
            {
                const Mat& bottom_blob2 = bottom_blobs[b];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = bottom_blob2.channel(q);
                    float* outptr = top_blob.channel(q);

#if __riscv_vector
                    int n = size;
                    while (n > 0)
                    {
                        size_t vl = __riscv_vsetvl_e32m8(n);
                        vfloat32m8_t _p = __riscv_vle32_v_f32m8(outptr, vl);
                        vfloat32m8_t _p1 = __riscv_vle32_v_f32m8(ptr, vl);
                        _p = __riscv_vfadd_vv_f32m8(_p, _p1, vl);
                        __riscv_vse32_v_f32m8(outptr, _p, vl);
                        ptr += vl; outptr += vl; n -= vl;
                    }
#else
                    for (int i = 0; i < size; i++)
                    {
                        outptr[i] += ptr[i];
                    }
#endif
                }
            }
        }
        else
        {
            const Mat& bottom_blob1 = bottom_blobs[1];
            float coeff0 = coeffs[0];
            float coeff1 = coeffs[1];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                const float* ptr1 = bottom_blob1.channel(q);
                float* outptr = top_blob.channel(q);

#if __riscv_vector
                int n = size;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m8(n);
                    vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                    vfloat32m8_t _p1 = __riscv_vle32_v_f32m8(ptr1, vl);
                    _p = __riscv_vfmul_vf_f32m8(_p, coeff0, vl);
                    vfloat32m8_t _tmp = __riscv_vfmul_vf_f32m8(_p1, coeff1, vl);
                    vfloat32m8_t _res = __riscv_vfadd_vv_f32m8(_p, _tmp, vl);
                    __riscv_vse32_v_f32m8(outptr, _res, vl);
                    ptr += vl; ptr1 += vl; outptr += vl; n -= vl;
                }
#else
                for (int i = 0; i < size; i++)
                {
                    outptr[i] = ptr[i] * coeff0 + ptr1[i] * coeff1;
                }
#endif
            }

            for (size_t b = 2; b < bottom_blobs.size(); b++)
            {
                const Mat& bottom_blob2 = bottom_blobs[b];
                float coeff = coeffs[b];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = bottom_blob2.channel(q);
                    float* outptr = top_blob.channel(q);

#if __riscv_vector
                    int n = size;
                    while (n > 0)
                    {
                        size_t vl = __riscv_vsetvl_e32m8(n);
                        vfloat32m8_t _p = __riscv_vle32_v_f32m8(outptr, vl);
                        vfloat32m8_t _p1 = __riscv_vle32_v_f32m8(ptr, vl);
                        vfloat32m8_t _tmp = __riscv_vfmul_vf_f32m8(_p1, coeff, vl);
                        _p = __riscv_vfadd_vv_f32m8(_p, _tmp, vl);
                        __riscv_vse32_v_f32m8(outptr, _p, vl);
                        ptr += vl; outptr += vl; n -= vl;
                    }
#else
                    for (int i = 0; i < size; i++)
                    {
                        outptr[i] += ptr[i] * coeff;
                    }
#endif
                }
            }
        }
    }
    else if (op_type == Operation_MAX)
    {
        const Mat& bottom_blob1 = bottom_blobs[1];
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            const float* ptr1 = bottom_blob1.channel(q);
            float* outptr = top_blob.channel(q);

#if __riscv_vector
            int n = size;
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e32m8(n);
                vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                vfloat32m8_t _p1 = __riscv_vle32_v_f32m8(ptr1, vl);
                vbool4_t _mask = __riscv_vmflt_vv_f32m8_b4(_p, _p1, vl);
                vfloat32m8_t _res = __riscv_vmerge_vvm_f32m8(_p1, _p, _mask, vl);
                __riscv_vse32_v_f32m8(outptr, _res, vl);
                ptr += vl; ptr1 += vl; outptr += vl; n -= vl;
            }
#else
            for (int i = 0; i < size; i++)
            {
                outptr[i] = std::max(ptr[i], ptr1[i]);
            }
#endif
        }

        for (size_t b = 2; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob2 = bottom_blobs[b];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob2.channel(q);
                float* outptr = top_blob.channel(q);

#if __riscv_vector
                int n = size;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m8(n);
                    vfloat32m8_t _p = __riscv_vle32_v_f32m8(outptr, vl);
                    vfloat32m8_t _p1 = __riscv_vle32_v_f32m8(ptr, vl);
                    vbool4_t _mask = __riscv_vmflt_vv_f32m8_b4(_p, _p1, vl);
                    vfloat32m8_t _res = __riscv_vmerge_vvm_f32m8(_p1, _p, _mask, vl);
                    __riscv_vse32_v_f32m8(outptr, _res, vl);
                    ptr += vl; outptr += vl; n -= vl;
                }
#else
                for (int i = 0; i < size; i++)
                {
                    outptr[i] = std::max(outptr[i], ptr[i]);
                }
#endif
            }
        }
    }

    return 0;
}

} // namespace ncnn
