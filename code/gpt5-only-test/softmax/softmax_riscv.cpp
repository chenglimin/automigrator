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

static void softmax_scalar(float* _ptr, int elemcount, int elempack)
{
    const int size = elemcount * elempack;

    float maxv = -FLT_MAX;
    for (int i = 0; i < size; i++)
        maxv = std::max(maxv, _ptr[i]);

    float sum = 0.f;
    for (int i = 0; i < size; i++)
    {
        float v = expf(_ptr[i] - maxv);
        _ptr[i] = v;
        sum += v;
    }

    sum = 1.f / sum;
    for (int i = 0; i < size; i++)
        _ptr[i] *= sum;
}

#if __riscv_vector
static void softmax_vec(float* _ptr, int elemcount, int elempack)
{
    const int size = elemcount * elempack;

    // reduce max
    float maxv = -FLT_MAX;
    {
        int n = size;
        float* p = _ptr;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(n);
            vfloat32m8_t _x = __riscv_vle32_v_f32m8(p, vl);
            // pairwise horizontal max using scalar fallback per chunk
            std::vector<float> tmp(vl);
            __riscv_vse32_v_f32m8(tmp.data(), _x, vl);
            for (size_t i = 0; i < vl; i++) maxv = std::max(maxv, tmp[i]);
            p += vl;
            n -= vl;
        }
    }

    // compute exp(x-max) and sum
    float sumv = 0.f;
    {
        int n = size;
        float* p = _ptr;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(n);
            vfloat32m8_t _x = __riscv_vle32_v_f32m8(p, vl);
            _x = __riscv_vfsub_vf_f32m8(_x, maxv, vl);
            _x = exp_ps(_x, vl);
            __riscv_vse32_v_f32m8(p, _x, vl);
            // horizontal sum
            vfloat32m1_t _sum = __riscv_vfmv_s_f_f32m1(0.f, __riscv_vsetvlmax_e32m1());
            _sum = __riscv_vfredusum_vs_f32m8_f32m1(_x, _sum, vl);
            sumv += __riscv_vfmv_f_s_f32m1_f32(_sum);
            p += vl;
            n -= vl;
        }
    }

    // scale by 1/sum
    float invsum = 1.f / sumv;
    {
        int n = size;
        float* p = _ptr;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(n);
            vfloat32m8_t _x = __riscv_vle32_v_f32m8(p, vl);
            _x = __riscv_vfmul_vf_f32m8(_x, invsum, vl);
            __riscv_vse32_v_f32m8(p, _x, vl);
            p += vl;
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
        int j = 0;
        for (; j < size1; j++)
        {
            float v = ptr[j];
            *maxptr = std::max(*maxptr, v);
            maxptr++;
        }
    }

    // reduce exp(x - max)
    for (int i = 0; i < elemcount; i++)
    {
        float* ptr = _ptr + i * stride;
        const float* maxptr = _maxptr;
        float* sumptr = _sumptr;

        int j = 0;
        for (; j < size1; j++)
        {
            float v = expf(ptr[j] - maxptr[j]);
            ptr[j] = v;
            *sumptr += v;
            sumptr++;
        }
    }

    // div sum
    for (int i = 0; i < elemcount; i++)
    {
        float* ptr = _ptr + i * stride;
        const float* sumptr = _sumptr;

        int j = 0;
        for (; j < size1; j++)
        {
            ptr[j] /= sumptr[j];
        }
    }
}
#endif // __riscv_vector

static void softmax(float* _ptr, int elemcount, int elempack)
{
#if __riscv_vector
    if (elempack != 1)
        return softmax_vec(_ptr, elemcount, elempack);
#endif
    softmax_scalar(_ptr, elemcount, elempack);
}

#if __riscv_vector
static void softmax(float* _ptr, int elemcount, int elempack, int stride, int size1, float* _maxptr, float* _sumptr)
{
    // setup max and sum workspace
    for (int j = 0; j < size1; j++)
    {
        _maxptr[j] = -FLT_MAX;
        _sumptr[j] = 0.f;
    }

    if (elempack == 1)
    {
        softmax_pack1(_ptr, elemcount, stride, size1, _maxptr, _sumptr);
        return;
    }

    // generic path for packed: reduce max
    for (int i = 0; i < elemcount; i++)
    {
        const float* ptr = _ptr + i * stride;
        float* maxptr = _maxptr;
        int n = size1 * elempack;
        const float* p = ptr;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(n);
            vfloat32m8_t _x = __riscv_vle32_v_f32m8(p, vl);
            std::vector<float> tmp(vl);
            __riscv_vse32_v_f32m8(tmp.data(), _x, vl);
            for (size_t k = 0; k < vl; k++)
            {
                int col = (p - ptr + k) % elempack;
                int idx = (p - ptr + k) / elempack;
                if (idx < size1)
                    maxptr[idx] = std::max(maxptr[idx], tmp[k]);
            }
            p += vl;
            n -= vl;
        }
    }

    // reduce exp(x - max)
    for (int i = 0; i < elemcount; i++)
    {
        float* ptr = _ptr + i * stride;
        const float* maxptr = _maxptr;
        float* sumptr = _sumptr;

        int n = size1 * elempack;
        float* p = ptr;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(n);
            vfloat32m8_t _x = __riscv_vle32_v_f32m8(p, vl);
            // subtract per-column max
            std::vector<float> tmp(vl);
            __riscv_vse32_v_f32m8(tmp.data(), _x, vl);
            for (size_t k = 0; k < vl; k++)
            {
                int idx = ((p - ptr) + k) / elempack;
                float v = expf(tmp[k] - maxptr[idx]);
                tmp[k] = v;
                sumptr[idx] += v;
            }
            _x = __riscv_vle32_v_f32m8(tmp.data(), vl);
            __riscv_vse32_v_f32m8(p, _x, vl);
            p += vl;
            n -= vl;
        }
    }

    // div sum
    for (int i = 0; i < elemcount; i++)
    {
        float* ptr = _ptr + i * stride;
        const float* sumptr = _sumptr;

        int n = size1 * elempack;
        float* p = ptr;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(n);
            vfloat32m8_t _x = __riscv_vle32_v_f32m8(p, vl);
            std::vector<float> tmp(vl);
            __riscv_vse32_v_f32m8(tmp.data(), _x, vl);
            for (size_t k = 0; k < vl; k++)
            {
                int idx = ((p - ptr) + k) / elempack;
                tmp[k] /= sumptr[idx];
            }
            _x = __riscv_vle32_v_f32m8(tmp.data(), vl);
            __riscv_vse32_v_f32m8(p, _x, vl);
            p += vl;
            n -= vl;
        }
    }
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

    if (dims == 1) // positive_axis == 0
    {
        float* ptr = bottom_top_blob;
        const int size = w * elempack;
        softmax(ptr, size, 1);
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

            softmax(ptr, h, elempack, stride, size1, maxptr, sumptr);
        }
    }

    if (dims == 2 && positive_axis == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            float* ptr = bottom_top_blob.row(i);
            softmax(ptr, w, elempack);
        }
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

            softmax(ptr, channels, elempack, stride, size1, maxptr, sumptr);
        }
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

                softmax(ptr, h, 1, size, size, maxptr, sumptr);
            }
        }
    }

    if (dims == 3 && positive_axis == 2)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);
            for (int i = 0; i < h; i++)
            {
                softmax(ptr, w, elempack);
                ptr += w * elempack;
            }
        }
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

            softmax(ptr, d, 1, size, size, maxptr, sumptr);
        }
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
                    softmax(ptr, w, elempack);
                    ptr += w * elempack;
                }
            }
        }
    }

    return 0;
}

} // namespace ncnn
