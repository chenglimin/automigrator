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
// on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and limitations under the License.

#include "padding_riscv.h"

#include <stdint.h>

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

#include "riscv_usability.h"

namespace ncnn {

#if __riscv_vector
#include "padding_packn.h"
#include "padding_packn_fp16.h"
#include "padding_pack8_int8.h"
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
    const int packn = csrr_vlenb() / 4;
    if (elempack == packn)
    {
        if (dims == 1)
        {
            int outw = w * elempack + left + right;

            int out_elempack = outw % packn == 0 ? packn : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (left % packn == 0 && out_elempack == packn && type == 0)
            {
                top_blob.create(outw / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                float vbuf[packn];
                for (int i = 0; i < packn; i++) vbuf[i] = value;
                padding_constant_packn_rvv(bottom_blob, top_blob, 0, 0, left / packn, right / packn, vbuf);

                return 0;
            }
        }

        if (dims == 2)
        {
            int outw = w + left + right;
            int outh = h * elempack + top + bottom;

            int out_elempack = outh % packn == 0 ? packn : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (top % packn == 0 && out_elempack == packn && type == 0)
            {
                top_blob.create(outw, outh / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                float vbuf[packn];
                for (int i = 0; i < packn; i++) vbuf[i] = value;
                padding_constant_packn_rvv(bottom_blob, top_blob, top / packn, bottom / packn, left, right, vbuf);

                return 0;
            }
        }

        if (dims == 3)
        {
            int outw = w + left + right;
            int outh = h + top + bottom;
            int outc = channels * elempack + front + behind;

            int out_elempack = outc % packn == 0 ? packn : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (front % packn == 0 && out_elempack == packn && !(outc != channels * elempack && type != 0))
            {
                top_blob.create(outw, outh, outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                int front_ = front / elempack;
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < outc / out_elempack; q++)
                {
                    Mat borderm = top_blob.channel(q);

                    // per channel pad
                    float vbuf[packn];
                    if (per_channel_pad_data_size)
                    {
                        const float* p = (const float*)per_channel_pad_data + q * packn;
                        for (int i = 0; i < packn; i++) vbuf[i] = p[i];
                    }
                    else
                    {
                        for (int i = 0; i < packn; i++) vbuf[i] = value;
                    }

                    //Channel padding
                    if ((q - front_) < 0 || (q - front_) >= channels)
                    {
            #if __riscv_vector
                        size_t vlpad = __riscv_vsetvl_e32m1(packn);
                        vfloat32m1_t _pad = __riscv_vle32_v_f32m1(vbuf, vlpad);
                        borderm.fill(_pad);
            #else
                        borderm.fill(value);
            #endif
                    }
                    else
                    {
                        const Mat m = bottom_blob.channel(q - front_);
                        if (type == 0)
                            padding_constant_packn_rvv(m, borderm, top, bottom, left, right, vbuf);
                        if (type == 1)
                            padding_replicate_packn_rvv(m, borderm, top, bottom, left, right);
                        if (type == 2)
                            padding_reflect_packn_rvv(m, borderm, top, bottom, left, right);
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
                    float vbuf[packn];
                    if (per_channel_pad_data_size)
                    {
                        const float* p = (const float*)per_channel_pad_data + q * packn;
                        for (int i = 0; i < packn; i++) vbuf[i] = p[i];
                    }
                    else
                    {
                        for (int i = 0; i < packn; i++) vbuf[i] = value;
                    }

                    for (int z = 0; z < outd; z++)
                    {
                        Mat borderm = top_blob.channel(q).depth(z);

                        // depth padding
                        if ((z - front) < 0 || (z - front) >= d)
                        {
            #if __riscv_vector
                            size_t vlpad = __riscv_vsetvl_e32m1(packn);
                            vfloat32m1_t _pad = __riscv_vle32_v_f32m1(vbuf, vlpad);
                            borderm.fill(_pad);
            #else
                            borderm.fill(value);
            #endif
                        }
                        else
                        {
                            const Mat m = bottom_blob.channel(q).depth(z - front);
                            padding_constant_packn_rvv(m, borderm, top, bottom, left, right, vbuf);
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
        if (bottom_blob_unpacked.empty())
            return -100;
    }

    return Padding::forward(bottom_blob_unpacked, top_blob, opt);
}

int Padding_riscv::forward_int8(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // int8 path: treat each pack of 8 int8 as int64 (same as x86/loongarch style)
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

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

                int64_t v8 = (int64_t)value;
                int64_t pad_value = v8 | (v8 << 8) | (v8 << 16) | (v8 << 24) | (v8 << 32) | (v8 << 40) | (v8 << 48) | (v8 << 56);

                // fallback scalar fill for int8 pack8
                const int64_t* ptr = bottom_blob;
                int64_t* outptr = top_blob;

                // top rows
                for (int y = 0; y < 0; y++) {}
                // center
                for (int y = 0; y < bottom_blob.h; y++)
                {
                    for (int x = 0; x < left / 8; x++) *outptr++ = pad_value;
                    for (int x = 0; x < bottom_blob.w; x++) *outptr++ = *ptr++;
                    for (int x = 0; x < right / 8; x++) *outptr++ = pad_value;
                }
                // bottom rows
                for (int y = 0; y < 0; y++) {}

                return 0;
            }
        }
        if (dims == 2)
        {
            int outw = w + left + right;
            int outh = h * elempack + top + bottom;

            int out_elempack = outh % 8 == 0 ? 8 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (top % 8 == 0 && out_elempack == 8 && type == 0)
            {
                top_blob.create(outw, outh / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                int64_t v8 = (int64_t)value;
                int64_t pad_value = v8 | (v8 << 8) | (v8 << 16) | (v8 << 24) | (v8 << 32) | (v8 << 40) | (v8 << 48) | (v8 << 56);

                const int64_t* ptr = bottom_blob;
                int64_t* outptr = top_blob;

                // top rows
                for (int y = 0; y < top / 8; y++)
                {
                    for (int x = 0; x < outw; x++) *outptr++ = pad_value;
                }
                // center
                for (int y = 0; y < h; y++)
                {
                    for (int x = 0; x < left; x++) *outptr++ = pad_value;
                    for (int x = 0; x < w; x++) *outptr++ = *ptr++;
                    for (int x = 0; x < right; x++) *outptr++ = pad_value;
                }
                // bottom rows
                for (int y = 0; y < bottom / 8; y++)
                {
                    for (int x = 0; x < outw; x++) *outptr++ = pad_value;
                }

                return 0;
            }
        }
        if (dims == 3)
        {
            int outw = w + left + right;
            int outh = h + top + bottom;
            int outc = channels * elempack + front + behind;

            int out_elempack = outc % 8 == 0 ? 8 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (front % 8 == 0 && out_elempack == 8 && !(outc != channels * elempack && type != 0))
            {
                top_blob.create(outw, outh, outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                int front_ = front / elempack;
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < outc / out_elempack; q++)
                {
                    Mat borderm = top_blob.channel(q);

                    int64_t v8 = (int64_t)value;
                    int64_t pad_value = v8 | (v8 << 8) | (v8 << 16) | (v8 << 24) | (v8 << 32) | (v8 << 40) | (v8 << 48) | (v8 << 56);

                    if ((q - front_) < 0 || (q - front_) >= channels)
                    {
                        borderm.fill<int64_t>(pad_value);
                    }
                    else
                    {
                        const Mat m = bottom_blob.channel(q - front_);
                        // type 0 constant only for int8
                        if (type == 0)
                        {
                            for (int y = 0; y < outh; y++)
                            {
                                int64_t* outptr = (int64_t*)borderm.row<int64_t>(y);
                                const int64_t* ptr = (const int64_t*)m.row<int64_t>(y - top);
                                for (int x = 0; x < left; x++) *outptr++ = pad_value;
                                for (int x = 0; x < w; x++) *outptr++ = *ptr++;
                                for (int x = 0; x < right; x++) *outptr++ = pad_value;
                            }
                        }
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
                    int64_t v8 = (int64_t)value;
                    int64_t pad_value = v8 | (v8 << 8) | (v8 << 16) | (v8 << 24) | (v8 << 32) | (v8 << 40) | (v8 << 48) | (v8 << 56);

                    for (int z = 0; z < outd; z++)
                    {
                        Mat borderm = top_blob.channel(q).depth(z);

                        if ((z - front) < 0 || (z - front) >= d)
                        {
                            borderm.fill<int64_t>(pad_value);
                        }
                        else
                        {
                            const Mat m = bottom_blob.channel(q).depth(z - front);
                            for (int y = 0; y < outh; y++)
                            {
                                int64_t* outptr = (int64_t*)borderm.row<int64_t>(y);
                                const int64_t* ptr = (const int64_t*)m.row<int64_t>(y - top);
                                for (int x = 0; x < left; x++) *outptr++ = pad_value;
                                for (int x = 0; x < w; x++) *outptr++ = *ptr++;
                                for (int x = 0; x < right; x++) *outptr++ = pad_value;
                            }
                        }
                    }
                }

                return 0;
            }
        }
    }

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
        const int packn = csrr_vlenb() / 4;
        out_elempack = top_blob_unpacked.c % packn == 0 ? packn : 1;
    }
#endif

    convert_packing(top_blob_unpacked, top_blob, out_elempack, opt);

    return 0;
}

} // namespace ncnn
