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

#include "softmax_riscv.h"

#include <float.h>
#include <algorithm>

#if __riscv_vector
#include <riscv_vector.h>
#include "rvv_mathfun.h"
#endif // __riscv_vector

#include "riscv_usability.h"
#include "cpu.h"

namespace ncnn {

Softmax_riscv::Softmax_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector
}

#if __riscv_vector
static void softmax_contiguous(float* _ptr, int elemcount, int elempack)
{
    const int size = elemcount * elempack;

    // reduce max
    float max = -FLT_MAX;
    {
        const float* ptr = _ptr;
        int n = size;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(n);
            vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
            vfloat32m1_t _maxacc = __riscv_vfmv_s_f_f32m1(-FLT_MAX, __riscv_vsetvlmax_e32m1());
            _maxacc = __riscv_vfredmax_vs_f32m8_f32m1(_p, _maxacc, vl);
            max = std::max(max, __riscv_vfmv_f_s_f32m1_f32(_maxacc));
            ptr += vl;
            n -= vl;
        }
    }

    // reduce exp(x - max)
    float sum = 0.f;
    {
        float* ptr = _ptr;
        int n = size;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(n);
            vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
            _p = __riscv_vfsub_vf_f32m8(_p, max, vl);
            _p = exp_ps(_p, vl);
            __riscv_vse32_v_f32m8(ptr, _p, vl);
            vfloat32m1_t _sumacc = __riscv_vfmv_s_f_f32m1(0.f, __riscv_vsetvlmax_e32m1());
            _sumacc = __riscv_vfredusum_vs_f32m8_f32m1(_p, _sumacc, vl);
            sum += __riscv_vfmv_f_s_f32m1_f32(_sumacc);
            ptr += vl;
            n -= vl;
        }
    }

    const float invsum = 1.f / sum;

    // div sum
    {
        float* ptr = _ptr;
        int n = size;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(n);
            vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
            _p = __riscv_vfmul_vf_f32m8(_p, invsum, vl);
            __riscv_vse32_v_f32m8(ptr, _p, vl);
            ptr += vl;
            n -= vl;
        }
    }
}

static void softmax_pack1(float* _ptr, int elemcount, int stride, int size1, float* _maxptr, float* _sumptr)
{
    // reduce max
    for (int i = 0; i < elemcount; i++)
    {
        const float* ptr = _ptr + i * stride;
        float* maxptr = _maxptr;

        int n = size1;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(n);
            vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
            vfloat32m8_t _max = __riscv_vle32_v_f32m8(maxptr, vl);
            _max = __riscv_vfmax_vv_f32m8(_max, _p, vl);
            __riscv_vse32_v_f32m8(maxptr, _max, vl);
            ptr += vl;
            maxptr += vl;
            n -= vl;
        }
    }

    // reduce exp(x - max)
    for (int i = 0; i < elemcount; i++)
    {
        float* ptr = _ptr + i * stride;
        const float* maxptr = _maxptr;
        float* sumptr = _sumptr;

        int n = size1;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(n);
            vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
            vfloat32m8_t _max = __riscv_vle32_v_f32m8(maxptr, vl);
            _p = __riscv_vfsub_vv_f32m8(_p, _max, vl);
            _p = exp_ps(_p, vl);
            __riscv_vse32_v_f32m8(ptr, _p, vl);
            vfloat32m8_t _sum = __riscv_vle32_v_f32m8(sumptr, vl);
            _sum = __riscv_vfadd_vv_f32m8(_sum, _p, vl);
            __riscv_vse32_v_f32m8(sumptr, _sum, vl);
            ptr += vl;
            maxptr += vl;
            sumptr += vl;
            n -= vl;
        }
    }

    // reciprocal of sums
    {
        float* sumptr = _sumptr;
        int n = size1;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(n);
            vfloat32m8_t _sum = __riscv_vle32_v_f32m8(sumptr, vl);
            _sum = __riscv_vfrdiv_vf_f32m8(_sum, 1.f, vl);
            __riscv_vse32_v_f32m8(sumptr, _sum, vl);
            sumptr += vl;
            n -= vl;
        }
    }

    // div sum
    for (int i = 0; i < elemcount; i++)
    {
        float* ptr = _ptr + i * stride;
        const float* sumptr = _sumptr;

        int n = size1;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(n);
            vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
            vfloat32m8_t _sum = __riscv_vle32_v_f32m8(sumptr, vl);
            _p = __riscv_vfmul_vv_f32m8(_p, _sum, vl);
            __riscv_vse32_v_f32m8(ptr, _p, vl);
            ptr += vl;
            sumptr += vl;
            n -= vl;
        }
    }
}

static void softmax_packn(float* _ptr, int elemcount, int stride, int size1, float* _maxptr, float* _sumptr)
{
    // reduce max per column (scalar)
    const int packn = csrr_vlenb() / 4;
    const size_t vl_packn = __riscv_vsetvl_e32m1(packn);

    for (int i = 0; i < elemcount; i++)
    {
        const float* ptr = _ptr + i * stride;
        float* maxptr = _maxptr;
        for (int j = 0; j < size1; j++)
        {
            vfloat32m1_t _p = __riscv_vle32_v_f32m1(ptr, vl_packn);
            vfloat32m1_t _acc = __riscv_vfmv_s_f_f32m1(-FLT_MAX, __riscv_vsetvlmax_e32m1());
            _acc = __riscv_vfredmax_vs_f32m1_f32m1(_p, _acc, vl_packn);
            float max0 = std::max(*maxptr, __riscv_vfmv_f_s_f32m1_f32(_acc));
            *maxptr = max0;
            ptr += packn;
            maxptr++;
        }
    }

    // reduce exp(x - max)
    for (int i = 0; i < elemcount; i++)
    {
        float* ptr = _ptr + i * stride;
        const float* maxptr = _maxptr;
        float* sumptr = _sumptr;
        for (int j = 0; j < size1; j++)
        {
            vfloat32m1_t _p = __riscv_vle32_v_f32m1(ptr, vl_packn);
            _p = __riscv_vfsub_vf_f32m1(_p, *maxptr, vl_packn);
            _p = exp_ps(_p, vl_packn);
            __riscv_vse32_v_f32m1(ptr, _p, vl_packn);
            vfloat32m1_t _acc = __riscv_vfmv_s_f_f32m1(0.f, __riscv_vsetvlmax_e32m1());
            _acc = __riscv_vfredusum_vs_f32m1_f32m1(_p, _acc, vl_packn);
            *sumptr += __riscv_vfmv_f_s_f32m1_f32(_acc);
            ptr += packn;
            maxptr++;
            sumptr++;
        }
    }

    // reciprocal of sums
    for (int j = 0; j < size1; j++)
    {
        _sumptr[j] = 1.f / _sumptr[j];
    }

    // div sum
    for (int i = 0; i < elemcount; i++)
    {
        float* ptr = _ptr + i * stride;
        const float* sumptr = _sumptr;
        for (int j = 0; j < size1; j++)
        {
            vfloat32m1_t _p = __riscv_vle32_v_f32m1(ptr, vl_packn);
            _p = __riscv_vfmul_vf_f32m1(_p, *sumptr, vl_packn);
            __riscv_vse32_v_f32m1(ptr, _p, vl_packn);
            ptr += packn;
            sumptr++;
        }
    }
}

static void softmax_dispatch(float* _ptr, int elemcount, int elempack, int stride, int size1, float* _maxptr, float* _sumptr)
{
    // init max and sum buffers
    {
        float* maxptr = _maxptr;
        int n = size1;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(n);
            vfloat32m8_t _negmax = __riscv_vfmv_v_f_f32m8(-FLT_MAX, vl);
            __riscv_vse32_v_f32m8(maxptr, _negmax, vl);
            maxptr += vl;
            n -= vl;
        }
    }
    {
        float* sumptr = _sumptr;
        int n = size1;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(n);
            vfloat32m8_t _zero = __riscv_vfmv_v_f_f32m8(0.f, vl);
            __riscv_vse32_v_f32m8(sumptr, _zero, vl);
            sumptr += vl;
            n -= vl;
        }
    }

    const int packn = csrr_vlenb() / 4;
    if (elempack == 1)
    {
        softmax_pack1(_ptr, elemcount, stride, size1, _maxptr, _sumptr);
        return;
    }
    if (elempack == packn)
    {
        softmax_packn(_ptr, elemcount, stride, size1, _maxptr, _sumptr);
        return;
    }
    // fallback
    softmax_pack1(_ptr, elemcount, stride, size1, _maxptr, _sumptr);
}
#endif // __riscv_vector

int Softmax_riscv::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    const int dims = bottom_top_blob.dims;
    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int d = bottom_top_blob.d;
    const int channels = bottom_top_blob.c;
    const int elempack = bottom_top_blob.elempack;
    const int positive_axis = axis < 0 ? dims + axis : axis;

#if __riscv_vector
    if (dims == 1) // positive_axis == 0
    {
        float* ptr = bottom_top_blob;
        const int size = w * elempack;
        softmax_contiguous(ptr, size, 1);
        return 0;
    }

    if (dims == 2 && positive_axis == 0)
    {
        const int size = w;
        const int sizen = (size + (opt.num_threads - 1)) / opt.num_threads;
        const int stride = w * elempack;

        Mat maxsum(sizen, 2, opt.num_threads, 4u, opt.workspace_allocator);
        if (maxsum.empty())
            return -100;

        const int nn_size = size / sizen;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            const int i = ii * sizen;
            const int size1 = std::min(sizen, size - i);

            float* maxsumptr = maxsum.channel(get_omp_thread_num());
            float* maxptr = maxsumptr;
            float* sumptr = maxptr + sizen;

            float* ptr = (float*)bottom_top_blob + i * elempack;

            softmax_dispatch(ptr, h, elempack, stride, size1, maxptr, sumptr);
        }
        return 0;
    }

    if (dims == 2 && positive_axis == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            float* ptr = bottom_top_blob.row(i);
            softmax_contiguous(ptr, w, elempack);
        }
        return 0;
    }

    if ((dims == 3 || dims == 4) && positive_axis == 0)
    {
        const int size = w * h * d;
        const int sizen = (size + (opt.num_threads - 1)) / opt.num_threads;
        const int stride = bottom_top_blob.cstep * elempack;

        Mat maxsum(sizen, 2, opt.num_threads, 4u, opt.workspace_allocator);
        if (maxsum.empty())
            return -100;

        const int nn_size = size / sizen;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            const int i = ii * sizen;
            const int size1 = std::min(sizen, size - i);

            float* maxsumptr = maxsum.channel(get_omp_thread_num());
            float* maxptr = maxsumptr;
            float* sumptr = maxptr + sizen;

            float* ptr = (float*)bottom_top_blob + i * elempack;

            softmax_dispatch(ptr, channels, elempack, stride, size1, maxptr, sumptr);
        }
        return 0;
    }

    if ((dims == 3 && positive_axis == 1) || (dims == 4 && positive_axis == 2))
    {
        const int size = w * elempack;

        Mat maxsum(size, 2, opt.num_threads, 4u, opt.workspace_allocator);
        if (maxsum.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            for (int i = 0; i < d; i++)
            {
                float* ptr = bottom_top_blob.channel(q).depth(i);

                float* maxsumptr = maxsum.channel(get_omp_thread_num());
                float* maxptr = maxsumptr;
                float* sumptr = maxptr + size;

                softmax_dispatch(ptr, h, 1, size, size, maxptr, sumptr);
            }
        }
        return 0;
    }

    if (dims == 3 && positive_axis == 2)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);
            for (int i = 0; i < h; i++)
            {
                softmax_contiguous(ptr, w, elempack);
                ptr += w * elempack;
            }
        }
        return 0;
    }

    if (dims == 4 && positive_axis == 1)
    {
        const int size = w * h * elempack;

        Mat maxsum(size, 2, opt.num_threads, 4u, opt.workspace_allocator);
        if (maxsum.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            float* maxsumptr = maxsum.channel(get_omp_thread_num());
            float* maxptr = maxsumptr;
            float* sumptr = maxptr + size;

            softmax_dispatch(ptr, d, 1, size, size, maxptr, sumptr);
        }
        return 0;
    }

    if (dims == 4 && positive_axis == 3)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);
            for (int i = 0; i < d; i++)
            {
                for (int j = 0; j < h; j++)
                {
                    softmax_contiguous(ptr, w, elempack);
                    ptr += w * elempack;
                }
            }
        }
        return 0;
    }
#endif // __riscv_vector

    // fallback to reference implementation if rvv not available or unexpected axis
    return Softmax::forward_inplace(bottom_top_blob, opt);
}

} // namespace ncnn
