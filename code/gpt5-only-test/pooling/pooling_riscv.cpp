// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "pooling_riscv.h"

#include <float.h>

#if __riscv_vector
#include <riscv_vector.h>
#include "riscv_usability.h"
#endif // __riscv_vector

namespace ncnn {

Pooling_riscv::Pooling_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector
}

int Pooling_riscv::create_pipeline(const Option& /*opt*/)
{
    if (adaptive_pooling)
    {
        support_packing = false;
    }
    return 0;
}

int Pooling_riscv::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (adaptive_pooling)
    {
        return Pooling::forward(bottom_blob, top_blob, opt);
    }

#if __riscv_vector
    int elempack = bottom_blob.elempack;
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;

    // global pooling fast path
    if (global_pooling)
    {
        top_blob.create(channels, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        int size = w * h;

        if (pooling_type == PoolMethod_MAX)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                int n = size * elempack;
                size_t vl = __riscv_vsetvl_e32m1(elempack);
                vfloat32m1_t _max = __riscv_vle32_v_f32m1(ptr, vl);
                ptr += elempack;
                n -= elempack;
                while (n > 0)
                {
                    vl = __riscv_vsetvl_e32m1(elempack);
                    vfloat32m1_t _val = __riscv_vle32_v_f32m1(ptr, vl);
                    _max = __riscv_vfmax_vv_f32m1(_max, _val, vl);
                    ptr += elempack;
                    n -= elempack;
                }
                float* outptr = top_blob;
                __riscv_vse32_v_f32m1(outptr + q * elempack, _max, __riscv_vsetvl_e32m1(elempack));
            }
        }
        else if (pooling_type == PoolMethod_AVE)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                int n = size * elempack;
                size_t vl = __riscv_vsetvl_e32m1(elempack);
                vfloat32m1_t _sum = __riscv_vfmv_v_f_f32m1(0.f, vl);
                while (n > 0)
                {
                    vl = __riscv_vsetvl_e32m1(elempack);
                    vfloat32m1_t _val = __riscv_vle32_v_f32m1(ptr, vl);
                    _sum = __riscv_vfadd_vv_f32m1(_sum, _val, vl);
                    ptr += elempack;
                    n -= elempack;
                }
                float inv_size = 1.f / size;
                vl = __riscv_vsetvl_e32m1(elempack);
                vfloat32m1_t _avg = __riscv_vfmul_vf_f32m1(_sum, inv_size, vl);
                float* outptr = top_blob;
                __riscv_vse32_v_f32m1(outptr + q * elempack, _avg, vl);
            }
        }

        return 0;
    }

    Mat bottom_blob_bordered;
    make_padding(bottom_blob, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    w = bottom_blob_bordered.w;
    h = bottom_blob_bordered.h;

    int outw = (w - kernel_w) / stride_w + 1;
    int outh = (h - kernel_h) / stride_h + 1;

    top_blob.create(outw, outh, channels, elemsize, elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    const int maxk = kernel_w * kernel_h;

    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w - kernel_w;
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2++;
            }
            p2 += gap;
        }
    }

    if (pooling_type == PoolMethod_MAX)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const Mat m = bottom_blob_bordered.channel(q);
            float* outptr = top_blob.channel(q);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    const float* sptr = m.row(i * stride_h) + j * stride_w * elempack;

                    size_t vl = __riscv_vsetvl_e32m1(elempack);
                    vfloat32m1_t _max = __riscv_vle32_v_f32m1(sptr, vl);

                    for (int k = 0; k < maxk; k++)
                    {
                        vfloat32m1_t _val = __riscv_vle32_v_f32m1(sptr + space_ofs[k] * elempack, vl);
                        _max = __riscv_vfmax_vv_f32m1(_max, _val, vl);
                    }

                    __riscv_vse32_v_f32m1(outptr, _max, vl);
                    outptr += elempack;
                }
            }
        }
    }
    else if (pooling_type == PoolMethod_AVE)
    {
        if (avgpool_count_include_pad == 0)
        {
            int wtailpad = 0;
            int htailpad = 0;

            if (pad_mode == 0)
            {
                wtailpad = bottom_blob_bordered.w - bottom_blob.w - pad_left - pad_right;
                htailpad = bottom_blob_bordered.h - bottom_blob.h - pad_top - pad_bottom;
            }

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat m = bottom_blob_bordered.channel(q);
                float* outptr = top_blob.channel(q);

                for (int i = 0; i < outh; i++)
                {
                    int sy0 = i * stride_h;

                    for (int j = 0; j < outw; j++)
                    {
                        int sx0 = j * stride_w;

                        size_t vl = __riscv_vsetvl_e32m1(elempack);
                        vfloat32m1_t _sum = __riscv_vfmv_v_f_f32m1(0.f, vl);
                        int area = 0;

                        for (int ki = 0; ki < kernel_h; ki++)
                        {
                            int sy = sy0 + ki;

                            if (sy < pad_top)
                                continue;

                            if (sy >= h - pad_bottom - htailpad)
                                break;

                            for (int kj = 0; kj < kernel_w; kj++)
                            {
                                int sx = sx0 + kj;

                                if (sx < pad_left)
                                    continue;

                                if (sx >= w - pad_right - wtailpad)
                                    break;

                                vfloat32m1_t _val = __riscv_vle32_v_f32m1(m.row(sy) + sx * elempack, vl);
                                _sum = __riscv_vfadd_vv_f32m1(_sum, _val, vl);
                                area += 1;
                            }
                        }

                        vfloat32m1_t _avg = __riscv_vfmul_vf_f32m1(_sum, 1.f / area, vl);
                        __riscv_vse32_v_f32m1(outptr, _avg, vl);
                        outptr += elempack;
                    }
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat m = bottom_blob_bordered.channel(q);
                float* outptr = top_blob.channel(q);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        const float* sptr = m.row(i * stride_h) + j * stride_w * elempack;

                        size_t vl = __riscv_vsetvl_e32m1(elempack);
                        vfloat32m1_t _sum = __riscv_vfmv_v_f_f32m1(0.f, vl);

                        for (int k = 0; k < maxk; k++)
                        {
                            vfloat32m1_t _val = __riscv_vle32_v_f32m1(sptr + space_ofs[k] * elempack, vl);
                            _sum = __riscv_vfadd_vv_f32m1(_sum, _val, vl);
                        }

                        vfloat32m1_t _avg = __riscv_vfmul_vf_f32m1(_sum, 1.f / maxk, vl);
                        __riscv_vse32_v_f32m1(outptr, _avg, vl);
                        outptr += elempack;
                    }
                }
            }
        }
    }

    return 0;
#else
    // no rvv, fallback
    return Pooling::forward(bottom_blob, top_blob, opt);
#endif // __riscv_vector
}

} // namespace ncnn
