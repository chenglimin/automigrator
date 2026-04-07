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

#include "padding_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

#include "riscv_usability.h"

namespace ncnn {

#if __riscv_vector
// pack4 fp32 helpers
static inline void padding_constant_pack4_rvv(const Mat& src, Mat& dst, int top, int bottom, int left, int right, vfloat32m1_t pad_value)
{
    int w = src.w;
    int h = src.h;
    int outw = dst.w;
    float* outptr = dst;
    const float* ptr = src;

    // top rows
    for (int y = 0; y < top; y++)
    {
        int n = outw;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m1(n);
            __riscv_vse32_v_f32m1(outptr, pad_value, vl);
            outptr += vl;
            n -= vl;
        }
    }

    // center rows
    for (int y = 0; y < h; y++)
    {
        // left
        int nleft = left;
        while (nleft > 0)
        {
            size_t vl = __riscv_vsetvl_e32m1(nleft);
            __riscv_vse32_v_f32m1(outptr, pad_value, vl);
            outptr += vl;
            nleft -= vl;
        }
        // copy src
        memcpy(outptr, ptr, w * sizeof(float));
        outptr += w;
        ptr += w;
        // right
        int nright = right;
        while (nright > 0)
        {
            size_t vl = __riscv_vsetvl_e32m1(nright);
            __riscv_vse32_v_f32m1(outptr, pad_value, vl);
            outptr += vl;
            nright -= vl;
        }
    }

    // bottom rows
    for (int y = 0; y < bottom; y++)
    {
        int n = outw;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m1(n);
            __riscv_vse32_v_f32m1(outptr, pad_value, vl);
            outptr += vl;
            n -= vl;
        }
    }
}

static inline void padding_replicate_pack4_rvv(const Mat& src, Mat& dst, int top, int bottom, int left, int right)
{
    int w = src.w;
    int h = src.h;
    int outw = dst.w;
    float* outptr = dst;
    const float* ptr = src;

    // top rows
    for (int y = 0; y < top; y++)
    {
        // left replicate
        for (int x = 0; x < left; x++)
            outptr[x] = ptr[0];
        memcpy(outptr + left, ptr, w * sizeof(float));
        for (int x = left + w; x < outw; x++)
            outptr[x] = ptr[w - 1];
        outptr += outw;
    }
    // center rows
    for (int y = 0; y < h; y++)
    {
        // left replicate
        for (int x = 0; x < left; x++)
            outptr[x] = ptr[0];
        memcpy(outptr + left, ptr, w * sizeof(float));
        for (int x = left + w; x < outw; x++)
            outptr[x] = ptr[w - 1];
        outptr += outw;
        ptr += w;
    }
    // bottom rows
    ptr -= w;
    for (int y = 0; y < bottom; y++)
    {
        for (int x = 0; x < left; x++)
            outptr[x] = ptr[0];
        memcpy(outptr + left, ptr, w * sizeof(float));
        for (int x = left + w; x < outw; x++)
            outptr[x] = ptr[w - 1];
        outptr += outw;
    }
}

static inline void padding_reflect_pack4_rvv(const Mat& src, Mat& dst, int top, int bottom, int left, int right)
{
    int w = src.w;
    int h = src.h;
    int outw = dst.w;
    float* outptr = dst;
    const float* ptr = src;

    // top rows
    ptr += top * w;
    for (int y = 0; y < top; y++)
    {
        for (int x = 0; x < left; x++)
            outptr[x] = ptr[left - x];
        memcpy(outptr + left, ptr, w * sizeof(float));
        for (int x = left + w; x < outw; x++)
            outptr[x] = ptr[w - (x - left - w) - 2];
        outptr += outw;
        ptr -= w;
    }
    // center rows
    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < left; x++)
            outptr[x] = ptr[left - x];
        memcpy(outptr + left, ptr, w * sizeof(float));
        for (int x = left + w; x < outw; x++)
            outptr[x] = ptr[w - (x - left - w) - 2];
        outptr += outw;
        ptr += w;
    }
    // bottom rows
    ptr -= 2 * w;
    for (int y = 0; y < bottom; y++)
    {
        for (int x = 0; x < left; x++)
            outptr[x] = ptr[left - x];
        memcpy(outptr + left, ptr, w * sizeof(float));
        for (int x = left + w; x < outw; x++)
            outptr[x] = ptr[w - (x - left - w) - 2];
        outptr += outw;
        ptr -= w;
    }
}
#endif // __riscv_vector

Padding_riscv::Padding_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector
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
    if (elempack == 4)
    {
        if (dims == 1)
        {
            int outw = w * elempack + left + right;
            int out_elempack = outw % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (left % 4 == 0 && out_elempack == 4 && type == 0)
            {
                top_blob.create(outw / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                vfloat32m1_t pad_value;
                {
                    size_t vl = __riscv_vsetvl_e32m1(4);
                    pad_value = __riscv_vfmv_v_f_f32m1(value, vl);
                }
                padding_constant_pack4_rvv(bottom_blob, top_blob, 0, 0, left / 4, right / 4, pad_value);
                return 0;
            }
        }
        if (dims == 2)
        {
            int outw = w + left + right;
            int outh = h * elempack + top + bottom;
            int out_elempack = outh % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;
            if (top % 4 == 0 && out_elempack == 4)
            {
                top_blob.create(outw, outh / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;
                if (type == 0)
                {
                    vfloat32m1_t pad_value;
                    size_t vl = __riscv_vsetvl_e32m1(4);
                    pad_value = __riscv_vfmv_v_f_f32m1(value, vl);
                    padding_constant_pack4_rvv(bottom_blob, top_blob, top / 4, bottom / 4, left, right, pad_value);
                    return 0;
                }
            }
        }
        if (dims == 3)
        {
            int outw = w + left + right;
            int outh = h + top + bottom;
            int outc = channels * elempack + front + behind;
            int out_elempack = outc % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;
            if (front % 4 == 0 && out_elempack == 4 && !(outc != channels * elempack && type != 0))
            {
                top_blob.create(outw, outh, outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                int front_ = front / elempack;
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < outc / out_elempack; q++)
                {
                    Mat borderm = top_blob.channel(q);
                    vfloat32m1_t pad_value;
                    size_t vl = __riscv_vsetvl_e32m1(4);
                    if (per_channel_pad_data_size)
                    {
                        pad_value = __riscv_vle32_v_f32m1((const float*)per_channel_pad_data + q * 4, vl);
                    }
                    else
                    {
                        pad_value = __riscv_vfmv_v_f_f32m1(value, vl);
                    }
                    if ((q - front_) < 0 || (q - front_) >= channels)
                    {
                        borderm.fill(pad_value);
                    }
                    else
                    {
                        const Mat m = bottom_blob.channel(q - front_);
                        if (type == 0)
                            padding_constant_pack4_rvv(m, borderm, top, bottom, left, right, pad_value);
                        if (type == 1)
                            padding_replicate_pack4_rvv(m, borderm, top, bottom, left, right);
                        if (type == 2)
                            padding_reflect_pack4_rvv(m, borderm, top, bottom, left, right);
                    }
                }
                return 0;
            }
        }
        if (dims == 4)
        {
            int outw = w + left + right;
            int outh = h + top + bottom;
            int outd = d + front + behind;
            if (type == 0)
            {
                top_blob.create(outw, outh, outd, channels, elemsize, elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    vfloat32m1_t pad_value;
                    size_t vl = __riscv_vsetvl_e32m1(4);
                    if (per_channel_pad_data_size)
                        pad_value = __riscv_vle32_v_f32m1((const float*)per_channel_pad_data + q * 4, vl);
                    else
                        pad_value = __riscv_vfmv_v_f_f32m1(value, vl);
                    for (int z = 0; z < outd; z++)
                    {
                        Mat borderm = top_blob.channel(q).depth(z);
                        if ((z - front) < 0 || (z - front) >= d)
                        {
                            borderm.fill(pad_value);
                        }
                        else
                        {
                            const Mat m = bottom_blob.channel(q).depth(z - front);
                            padding_constant_pack4_rvv(m, borderm, top, bottom, left, right, pad_value);
                        }
                    }
                }
                return 0;
            }
        }
    }
#endif // __riscv_vector

    Mat bottom_blob_unpacked = bottom_blob;
    if (elempack != 1)
    {
        Option opt_pack1 = opt;
        opt_pack1.blob_allocator = opt.workspace_allocator;
        convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt_pack1);
    }

    Mat top_blob_unpacked;
    int ret = Padding::forward(bottom_blob_unpacked, top_blob_unpacked, opt);
    if (ret != 0)
        return ret;

    int out_elempack = 1;
#if __riscv_vector
    if (opt.use_packing_layout)
    {
        out_elempack = top_blob_unpacked.c % 4 == 0 ? 4 : 1;
    }
#endif

    convert_packing(top_blob_unpacked, top_blob, out_elempack, opt);

    return 0;
}

int Padding_riscv::forward_int8(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

#if __riscv_vector
    if (elempack == 8)
    {
        if (dims == 1)
        {
            int outw = w * elempack + left + right;
            int out_elempack = outw % 8 == 0 ? 8 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;
            if (left % 8 == 0 && out_elempack == 8 && type == 0)
            {
                top_blob.create(outw / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;
                // build int8 vector value
                signed char v8 = (signed char)value;
                vint8m1_t pad_value = __riscv_vmv_v_x_i8m1(v8, __riscv_vsetvl_e8m1(8));
                // fill
                int outw8 = outw / 8;
                signed char* outptr = (signed char*)top_blob;
                const signed char* ptr = (const signed char*)bottom_blob;
                for (int y = 0; y < 1; y++)
                {
                    for (int x = 0; x < left / 8; x++)
                    {
                        __riscv_vse8_v_i8m1(outptr, pad_value, __riscv_vsetvl_e8m1(8));
                        outptr += 8;
                    }
                    memcpy(outptr, ptr, w * 8 * sizeof(signed char));
                    outptr += w * 8;
                    ptr += w * 8;
                    for (int x = 0; x < right / 8; x++)
                    {
                        __riscv_vse8_v_i8m1(outptr, pad_value, __riscv_vsetvl_e8m1(8));
                        outptr += 8;
                    }
                }
                return 0;
            }
        }
        // for dims 2/3/4, fallback to generic
    }
#endif // __riscv_vector

    Mat bottom_blob_unpacked = bottom_blob;
    if (elempack != 1)
    {
        Option opt_pack1 = opt;
        opt_pack1.blob_allocator = opt.workspace_allocator;
        convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt_pack1);
    }

    Mat top_blob_unpacked;
    int ret = Padding::forward(bottom_blob_unpacked, top_blob_unpacked, opt);
    if (ret != 0)
        return ret;

    int out_elempack = 1;
#if __riscv_vector
    if (opt.use_packing_layout)
    {
        out_elempack = top_blob_unpacked.c % 8 == 0 ? 8 : 1;
    }
#endif

    convert_packing(top_blob_unpacked, top_blob, out_elempack, opt);
    return 0;
}

} // namespace ncnn
