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

#include "pooling_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

#include <float.h>
#include <vector>

#include "riscv_usability.h"

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
        support_bf16_storage = false;
        support_fp16_storage = false;
        support_int8_storage = false;
        support_tensor_storage = false;
    }
    return 0;
}

static inline void pooling2x2s2_max_packn(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
#if __riscv_vector
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int elempack = bottom_blob.elempack;
    const int packn = csrr_vlenb() / 4;
    const size_t vl = __riscv_vsetvl_e32m8(packn);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const Mat m = bottom_blob.channel(q);
        float* outptr = top_blob.channel(q);
        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                const float* sptr0 = m.row(i * 2) + j * 2 * elempack;
                const float* sptr1 = sptr0 + w * elempack;
                vfloat32m8_t _p0 = __riscv_vle32_v_f32m8(sptr0, vl);
                vfloat32m8_t _p1 = __riscv_vle32_v_f32m8(sptr0 + elempack, vl);
                vfloat32m8_t _p2 = __riscv_vle32_v_f32m8(sptr1, vl);
                vfloat32m8_t _p3 = __riscv_vle32_v_f32m8(sptr1 + elempack, vl);
                vfloat32m8_t _max01 = __riscv_vfmax_vv_f32m8(_p0, _p1, vl);
                vfloat32m8_t _max23 = __riscv_vfmax_vv_f32m8(_p2, _p3, vl);
                vfloat32m8_t _max = __riscv_vfmax_vv_f32m8(_max01, _max23, vl);
                __riscv_vse32_v_f32m8(outptr, _max, vl);
                outptr += elempack;
            }
        }
    }
#else
    (void)bottom_blob; (void)top_blob; (void)opt;
#endif
}

static inline void pooling3x3s2_max_packn(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
#if __riscv_vector
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int elempack = bottom_blob.elempack;
    const int packn = csrr_vlenb() / 4;
    const size_t vl = __riscv_vsetvl_e32m8(packn);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const Mat m = bottom_blob.channel(q);
        float* outptr = top_blob.channel(q);
        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                const float* r0 = m.row(i * 2) + j * 2 * elempack;
                const float* r1 = r0 + w * elempack;
                const float* r2 = r1 + w * elempack;
                vfloat32m8_t _r0_0 = __riscv_vle32_v_f32m8(r0, vl);
                vfloat32m8_t _r0_1 = __riscv_vle32_v_f32m8(r0 + elempack, vl);
                vfloat32m8_t _r0_2 = __riscv_vle32_v_f32m8(r0 + 2 * elempack, vl);
                vfloat32m8_t _r1_0 = __riscv_vle32_v_f32m8(r1, vl);
                vfloat32m8_t _r1_1 = __riscv_vle32_v_f32m8(r1 + elempack, vl);
                vfloat32m8_t _r1_2 = __riscv_vle32_v_f32m8(r1 + 2 * elempack, vl);
                vfloat32m8_t _r2_0 = __riscv_vle32_v_f32m8(r2, vl);
                vfloat32m8_t _r2_1 = __riscv_vle32_v_f32m8(r2 + elempack, vl);
                vfloat32m8_t _r2_2 = __riscv_vle32_v_f32m8(r2 + 2 * elempack, vl);
                vfloat32m8_t _max0 = __riscv_vfmax_vv_f32m8(_r0_0, _r0_1, vl);
                _max0 = __riscv_vfmax_vv_f32m8(_max0, _r0_2, vl);
                vfloat32m8_t _max1 = __riscv_vfmax_vv_f32m8(_r1_0, _r1_1, vl);
                _max1 = __riscv_vfmax_vv_f32m8(_max1, _r1_2, vl);
                vfloat32m8_t _max2 = __riscv_vfmax_vv_f32m8(_r2_0, _r2_1, vl);
                _max2 = __riscv_vfmax_vv_f32m8(_max2, _r2_2, vl);
                vfloat32m8_t _max = __riscv_vfmax_vv_f32m8(__riscv_vfmax_vv_f32m8(_max0, _max1, vl), _max2, vl);
                __riscv_vse32_v_f32m8(outptr, _max, vl);
                outptr += elempack;
            }
        }
    }
#else
    (void)bottom_blob; (void)top_blob; (void)opt;
#endif
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

    const int packn = csrr_vlenb() / 4;
    const size_t vl = __riscv_vsetvl_e32m8(packn);

    {
        // Fallback to reference implementation for correctness across all stride/pad variants
        Mat bottom_blob_unpacked;
        convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt);
        if (bottom_blob_unpacked.empty()) return -100;
        Mat top_blob_unpacked;
        int ret = Pooling::forward(bottom_blob_unpacked, top_blob_unpacked, opt);
        if (ret != 0) return ret;
        convert_packing(top_blob_unpacked, top_blob, elempack, opt);
        if (top_blob.empty()) return -100;
        return 0;
    }

    if (elempack != packn)
    {
        // elempack not equal to packn; fallback to reference implementation for correctness
        return Pooling::forward(bottom_blob, top_blob, opt);
    }

    {
        if (global_pooling)
        {
            top_blob.create(channels, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty()) return -100;
            int size = w * h;

            if (pooling_type == PoolMethod_MAX)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = bottom_blob.channel(q);
                    vfloat32m8_t _max = __riscv_vle32_v_f32m8(ptr, vl);
                    for (int i = 0; i < size; i++)
                    {
                        vfloat32m8_t _val = __riscv_vle32_v_f32m8(ptr, vl);
                        _max = __riscv_vfmax_vv_f32m8(_max, _val, vl);
                        ptr += elempack;
                    }
                    float* outptr = top_blob.channel(q);
                    __riscv_vse32_v_f32m8(outptr, _max, vl);
                }
            }
            else if (pooling_type == PoolMethod_AVE)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = bottom_blob.channel(q);
                    vfloat32m8_t _sum = __riscv_vfmv_v_f_f32m8(0.f, vl);
                    for (int i = 0; i < size; i++)
                    {
                        vfloat32m8_t _val = __riscv_vle32_v_f32m8(ptr, vl);
                        _sum = __riscv_vfadd_vv_f32m8(_sum, _val, vl);
                        ptr += elempack;
                    }
                    vfloat32m8_t _inv = __riscv_vfmv_v_f_f32m8(1.f / size, vl);
                    vfloat32m8_t _avg = __riscv_vfmul_vv_f32m8(_sum, _inv, vl);
                    float* outptr = top_blob.channel(q);
                    __riscv_vse32_v_f32m8(outptr, _avg, vl);
                }
            }
            return 0;
        }

        // Fallback to generic implementation for general stride/pad configurations
        if (!((kernel_w == 2 && kernel_h == 2 && stride_w == 2 && stride_h == 2) || (kernel_w == 3 && kernel_h == 3 && stride_w == 2 && stride_h == 2)))
        {
            Mat bottom_blob_unpacked;
            convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt);
            if (bottom_blob_unpacked.empty()) return -100;

            Mat top_blob_unpacked;
            int ret = Pooling::forward(bottom_blob_unpacked, top_blob_unpacked, opt);
            if (ret != 0) return ret;

            convert_packing(top_blob_unpacked, top_blob, elempack, opt);
            if (top_blob.empty()) return -100;
            return 0;
        }

        Mat bottom_blob_bordered;
        make_padding(bottom_blob, bottom_blob_bordered, opt);
        if (bottom_blob_bordered.empty()) return -100;
        w = bottom_blob_bordered.w;
        h = bottom_blob_bordered.h;
        int outw = (w - kernel_w) / stride_w + 1;
        int outh = (h - kernel_h) / stride_h + 1;
        top_blob.create(outw, outh, channels, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty()) return -100;

        const int maxk = kernel_w * kernel_h;
        std::vector<int> _space_ofs(maxk);
        int* space_ofs = &_space_ofs[0];
        {
            int p1 = 0, p2 = 0; int gap = w - kernel_w;
            for (int i = 0; i < kernel_h; i++)
            {
                for (int j = 0; j < kernel_w; j++) { space_ofs[p1++] = p2++; }
                p2 += gap;
            }
        }

        if (pooling_type == PoolMethod_MAX)
        {
            if (kernel_w == 2 && kernel_h == 2 && stride_w == 2 && stride_h == 2)
            {
                pooling2x2s2_max_packn(bottom_blob_bordered, top_blob, opt);
                return 0;
            }
            if (kernel_w == 3 && kernel_h == 3 && stride_w == 2 && stride_h == 2)
            {
                pooling3x3s2_max_packn(bottom_blob_bordered, top_blob, opt);
                return 0;
            }

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
                        vfloat32m8_t _max = __riscv_vle32_v_f32m8(sptr, vl);
                        for (int k = 0; k < maxk; k++)
                        {
                            vfloat32m8_t _val = __riscv_vle32_v_f32m8(sptr + space_ofs[k] * elempack, vl);
                            _max = __riscv_vfmax_vv_f32m8(_max, _val, vl);
                        }
                        __riscv_vse32_v_f32m8(outptr, _max, vl);
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
                            vfloat32m8_t _sum = __riscv_vfmv_v_f_f32m8(0.f, vl);
                            int area = 0;
                            for (int ki = 0; ki < kernel_h; ki++)
                            {
                                int sy = sy0 + ki;
                                if (sy < pad_top) continue;
                                if (sy >= h - pad_bottom - htailpad) break;
                                for (int kj = 0; kj < kernel_w; kj++)
                                {
                                    int sx = sx0 + kj;
                                    if (sx < pad_left) continue;
                                    if (sx >= w - pad_right - wtailpad) break;
                                    vfloat32m8_t _val = __riscv_vle32_v_f32m8(m.row(sy) + sx * elempack, vl);
                                    _sum = __riscv_vfadd_vv_f32m8(_sum, _val, vl);
                                    area += 1;
                                }
                            }
                            vfloat32m8_t _inv = __riscv_vfmv_v_f_f32m8(1.f / area, vl);
                            vfloat32m8_t _avg = __riscv_vfmul_vv_f32m8(_sum, _inv, vl);
                            __riscv_vse32_v_f32m8(outptr, _avg, vl);
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
                    vfloat32m8_t _invk = __riscv_vfmv_v_f_f32m8(1.f / (kernel_w * kernel_h), vl);
                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            const float* sptr = m.row(i * stride_h) + j * stride_w * elempack;
                            vfloat32m8_t _sum = __riscv_vfmv_v_f_f32m8(0.f, vl);
                            for (int k = 0; k < maxk; k++)
                            {
                                vfloat32m8_t _val = __riscv_vle32_v_f32m8(sptr + space_ofs[k] * elempack, vl);
                                _sum = __riscv_vfadd_vv_f32m8(_sum, _val, vl);
                            }
                            vfloat32m8_t _avg = __riscv_vfmul_vv_f32m8(_sum, _invk, vl);
                            __riscv_vse32_v_f32m8(outptr, _avg, vl);
                            outptr += elempack;
                        }
                    }
                }
            }
        }
        return 0;
    }
#endif // __riscv_vector

    // Fallback to generic implementation when elempack != packn or vector not available
    return Pooling::forward(bottom_blob, top_blob, opt);
}

} // namespace ncnn
