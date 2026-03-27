// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2025 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "padding_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

#include "riscv_usability.h"

namespace ncnn {

Padding_riscv::Padding_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector
}

// RVV packn helpers mapped from x86 pack4/pack8/pack16
static inline void padding_constant_packn_rvv(const Mat& src, Mat& dst, int top, int bottom, int left, int right, const float v)
{
#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
    const size_t vl = __riscv_vsetvl_e32m1(packn);
    const float* ptr = src;
    float* outptr = dst;
    int top_size = top * dst.w;
    int bottom_size = bottom * dst.w;

    vfloat32m1_t _v = __riscv_vfmv_v_f_f32m1(v, vl);

    for (int y = 0; y < top_size; y++)
    {
        __riscv_vse32_v_f32m1(outptr, _v, vl);
        outptr += packn;
    }

    for (int y = 0; y < src.h; y++)
    {
        for (int x = 0; x < left; x++)
        {
            __riscv_vse32_v_f32m1(outptr, _v, vl);
            outptr += packn;
        }
        for (int x = 0; x < src.w; x++)
        {
            vfloat32m1_t _p = __riscv_vle32_v_f32m1(ptr, vl);
            __riscv_vse32_v_f32m1(outptr, _p, vl);
            ptr += packn;
            outptr += packn;
        }
        for (int x = 0; x < right; x++)
        {
            __riscv_vse32_v_f32m1(outptr, _v, vl);
            outptr += packn;
        }
    }

    for (int y = 0; y < bottom_size; y++)
    {
        __riscv_vse32_v_f32m1(outptr, _v, vl);
        outptr += packn;
    }
#else
    const float* ptr = src;
    float* outptr = dst;
    int top_size = top * dst.w;
    int bottom_size = bottom * dst.w;
    for (int y = 0; y < top_size; y++)
    {
        for (int k = 0; k < src.elempack; k++) *outptr++ = v;
    }
    for (int y = 0; y < src.h; y++)
    {
        for (int x = 0; x < left; x++)
        {
            for (int k = 0; k < src.elempack; k++) *outptr++ = v;
        }
        for (int x = 0; x < src.w; x++)
        {
            for (int k = 0; k < src.elempack; k++) *outptr++ = *ptr++;
        }
        for (int x = 0; x < right; x++)
        {
            for (int k = 0; k < src.elempack; k++) *outptr++ = v;
        }
    }
    for (int y = 0; y < bottom_size; y++)
    {
        for (int k = 0; k < src.elempack; k++) *outptr++ = v;
    }
#endif
}

// replicate/reflection helpers use scalar path due to index pattern differences
static inline void padding_replicate_packn_rvv(const Mat& src, Mat& dst, int top, int bottom, int left, int right)
{
    const float* ptr = src;
    float* outptr = dst;
    for (int y = 0; y < top; y++)
    {
        const float* ptr0 = ptr;
        for (int x = 0; x < left; x++)
        {
            for (int k = 0; k < src.elempack; k++) outptr[k] = ptr0[k];
            outptr += src.elempack;
        }
        for (int x = 0; x < src.w; x++)
        {
            for (int k = 0; k < src.elempack; k++) outptr[k] = ptr0[k];
            ptr0 += src.elempack;
            outptr += src.elempack;
        }
        for (int x = 0; x < right; x++)
        {
            for (int k = 0; k < src.elempack; k++) outptr[k] = ptr0[k - src.elempack];
            outptr += src.elempack;
        }
    }
    for (int y = 0; y < src.h; y++)
    {
        const float* ptr0 = ptr;
        for (int x = 0; x < left; x++)
        {
            for (int k = 0; k < src.elempack; k++) outptr[k] = ptr0[k];
            outptr += src.elempack;
        }
        for (int x = 0; x < src.w; x++)
        {
            for (int k = 0; k < src.elempack; k++) outptr[k] = ptr[k];
            ptr += src.elempack;
            outptr += src.elempack;
        }
        for (int x = 0; x < right; x++)
        {
            for (int k = 0; k < src.elempack; k++) outptr[k] = ptr[k - src.elempack];
            outptr += src.elempack;
        }
    }
    ptr -= src.w * src.elempack;
    for (int y = 0; y < bottom; y++)
    {
        const float* ptr0 = ptr;
        for (int x = 0; x < left; x++)
        {
            for (int k = 0; k < src.elempack; k++) outptr[k] = ptr0[k];
            outptr += src.elempack;
        }
        for (int x = 0; x < src.w; x++)
        {
            for (int k = 0; k < src.elempack; k++) outptr[k] = ptr0[k];
            ptr0 += src.elempack;
            outptr += src.elempack;
        }
        for (int x = 0; x < right; x++)
        {
            for (int k = 0; k < src.elempack; k++) outptr[k] = ptr0[k - src.elempack];
            outptr += src.elempack;
        }
    }
}

static inline void padding_reflect_packn_rvv(const Mat& src, Mat& dst, int top, int bottom, int left, int right)
{
    const float* ptr = src;
    float* outptr = dst;
    ptr += top * src.w * src.elempack;
    for (int y = 0; y < top; y++)
    {
        const float* ptr0 = ptr;
        for (int x = 0; x < left; x++)
        {
            const float* psrc = ptr0 + (left - x) * src.elempack;
            for (int k = 0; k < src.elempack; k++) outptr[k] = psrc[k];
            outptr += src.elempack;
        }
        for (int x = 0; x < src.w; x++)
        {
            for (int k = 0; k < src.elempack; k++) outptr[k] = ptr0[k];
            ptr0 += src.elempack;
            outptr += src.elempack;
        }
        for (int x = 0; x < right; x++)
        {
            const float* psrc = ptr0 - 2 * src.elempack - x * src.elempack;
            for (int k = 0; k < src.elempack; k++) outptr[k] = psrc[k];
            outptr += src.elempack;
        }
        ptr -= src.w * src.elempack;
    }
    for (int y = 0; y < src.h; y++)
    {
        for (int x = 0; x < left; x++)
        {
            const float* psrc = ptr + (left - x) * src.elempack;
            for (int k = 0; k < src.elempack; k++) outptr[k] = psrc[k];
            outptr += src.elempack;
        }
        for (int x = 0; x < src.w; x++)
        {
            for (int k = 0; k < src.elempack; k++) outptr[k] = ptr[k];
            ptr += src.elempack;
            outptr += src.elempack;
        }
        for (int x = 0; x < right; x++)
        {
            const float* psrc = ptr - 2 * src.elempack - x * src.elempack;
            for (int k = 0; k < src.elempack; k++) outptr[k] = psrc[k];
            outptr += src.elempack;
        }
    }
    ptr -= 2 * src.w * src.elempack;
    for (int y = 0; y < bottom; y++)
    {
        const float* ptr0 = ptr;
        for (int x = 0; x < left; x++)
        {
            const float* psrc = ptr0 + (left - x) * src.elempack;
            for (int k = 0; k < src.elempack; k++) outptr[k] = psrc[k];
            outptr += src.elempack;
        }
        for (int x = 0; x < src.w; x++)
        {
            for (int k = 0; k < src.elempack; k++) outptr[k] = ptr0[k];
            ptr0 += src.elempack;
            outptr += src.elempack;
        }
        for (int x = 0; x < right; x++)
        {
            const float* psrc = ptr0 - 2 * src.elempack - x * src.elempack;
            for (int k = 0; k < src.elempack; k++) outptr[k] = psrc[k];
            outptr += src.elempack;
        }
        ptr -= src.w * src.elempack;
    }
}

int Padding_riscv::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (top == 0 && bottom == 0 && left == 0 && right == 0 && front == 0 && behind == 0)
    {
        top_blob = bottom_blob;
        return 0;
    }

    int elembits = bottom_blob.elembits();

    if (elembits == 8)
        return forward_int8(bottom_blob, top_blob, opt);

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

#if __riscv_vector
    if (elempack > 1)
    {
        if (dims == 1)
        {
            int outw = w * elempack + left + right;
            int out_elempack = outw % elempack == 0 ? elempack : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (left % elempack == 0 && out_elempack == elempack && type == 0)
            {
                top_blob.create(outw / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;
                padding_constant_packn_rvv(bottom_blob, top_blob, 0, 0, left / elempack, right / elempack, value);
                return 0;
            }
        }
        if (dims == 2)
        {
            int outw = w + left + right;
            int outh = h * elempack + top + bottom;
            int out_elempack = outh % elempack == 0 ? elempack : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (top % elempack == 0 && out_elempack == elempack && type == 0)
            {
                top_blob.create(outw, outh / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;
                padding_constant_packn_rvv(bottom_blob, top_blob, top / elempack, bottom / elempack, left, right, value);
                return 0;
            }
        }
        if (dims == 3)
        {
            int outw = w + left + right;
            int outh = h + top + bottom;
            int outc = channels * elempack + front + behind;
            int out_elempack = outc % elempack == 0 ? elempack : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (per_channel_pad_data_size == 0 && front % elempack == 0 && out_elempack == elempack && !(outc != channels * elempack && type != 0))
            {
                top_blob.create(outw, outh, outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                int front_ = front / elempack;
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < outc / out_elempack; q++)
                {
                    Mat borderm = top_blob.channel(q);
                    if ((q - front_) < 0 || (q - front_) >= channels)
                    {
                        // per-channel pad with packn: fallback to scalar fill for correctness
                        float pad_value = per_channel_pad_data_size ? ((const float*)per_channel_pad_data)[q * elempack] : value;
                        borderm.fill(pad_value);
                    }
                    else
                    {
                        const Mat m = bottom_blob.channel(q - front_);
                        float pad_value = per_channel_pad_data_size ? ((const float*)per_channel_pad_data)[q * elempack] : value;
                        if (type == 0)
                            padding_constant_packn_rvv(m, borderm, top, bottom, left, right, pad_value);
                        if (type == 1)
                            padding_replicate_packn_rvv(m, borderm, top, bottom, left, right);
                        if (type == 2)
                            padding_reflect_packn_rvv(m, borderm, top, bottom, left, right);
                    }
                }

                return 0;
            }
        }
        if (dims == 4 && false)
        {
            // fallback to generic path for 4D tensors to ensure correctness across variable VLEN
        }
    }
#endif // __riscv_vector

    Mat bottom_blob_unpacked = bottom_blob;
    if (elempack != 1)
    {
        Option opt_pack1 = opt;
        opt_pack1.blob_allocator = opt.workspace_allocator;
        convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt_pack1);
        if (bottom_blob_unpacked.empty())
            return -100;
    }

    return Padding::forward(bottom_blob_unpacked, top_blob, opt);
}

int Padding_riscv::forward_int8(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // fallback to generic int8 implementation
    Mat bottom_blob_unpacked = bottom_blob;
    if (bottom_blob.elempack != 1)
    {
        Option opt_pack1 = opt;
        opt_pack1.blob_allocator = opt.workspace_allocator;
        convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt_pack1);
        if (bottom_blob_unpacked.empty())
            return -100;
    }
    return Padding::forward(bottom_blob_unpacked, top_blob, opt);
}

} // namespace ncnn
