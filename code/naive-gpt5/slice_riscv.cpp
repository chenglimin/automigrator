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

#include "slice_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

namespace ncnn {

Slice_riscv::Slice_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector
}

int Slice_riscv::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;
    const int* slices_ptr = slices;
    const int* indices_ptr = indices;
    int positive_axis = axis < 0 ? dims + axis : axis;

    if (dims == 1) // positive_axis == 0
    {
        // slice vector
        int w = bottom_blob.w * elempack;
        int q = 0;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            int slice;
            if (indices_ptr)
            {
                if (i == top_blobs.size() - 1)
                {
                    slice = w - q;
                }
                else
                {
                    int indice = indices_ptr[i];
                    int positive_indice = indice < 0 ? w + indice : indice;
                    slice = positive_indice - q;
                }
            }
            else
            {
                slice = slices_ptr[i];
                if (slice == -233)
                {
                    slice = static_cast<int>((w - q) / (top_blobs.size() - i));
                }
            }

            int out_elempack = 1;
#if __riscv_vector
            if (opt.use_packing_layout)
            {
                out_elempack = slice % 8 == 0 ? 8 : slice % 4 == 0 ? 4 : 1;
            }
#endif // __riscv_vector
            size_t out_elemsize = elemsize / elempack * out_elempack;

            Mat& top_blob = top_blobs[i];
            top_blob.create(slice / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            const float* ptr = (const float*)bottom_blob + q;
            float* outptr = top_blob;
            memcpy(outptr, ptr, top_blob.w * top_blob.elemsize);

            q += slice;
        }
    }

    if (dims == 2 && positive_axis == 0)
    {
        // slice image height
        int w = bottom_blob.w;
        int h = bottom_blob.h * elempack;

        int q = 0;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            int slice;
            if (indices_ptr)
            {
                if (i == top_blobs.size() - 1)
                {
                    slice = h - q;
                }
                else
                {
                    int indice = indices_ptr[i];
                    int positive_indice = indice < 0 ? h + indice : indice;
                    slice = positive_indice - q;
                }
            }
            else
            {
                slice = slices_ptr[i];
                if (slice == -233)
                {
                    slice = static_cast<int>((h - q) / (top_blobs.size() - i));
                }
            }

            int out_elempack = 1;
#if __riscv_vector
            if (opt.use_packing_layout)
            {
                out_elempack = slice % 8 == 0 ? 8 : slice % 4 == 0 ? 4 : 1;
            }
#endif // __riscv_vector
            size_t out_elemsize = elemsize / elempack * out_elempack;

            Mat& top_blob = top_blobs[i];
            top_blob.create(w, slice / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            q += slice;
        }

        size_t out_elemsize2 = top_blobs[0].elemsize;
        int out_elempack2 = top_blobs[0].elempack;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            out_elemsize2 = std::min(out_elemsize2, top_blobs[i].elemsize);
            out_elempack2 = std::min(out_elempack2, top_blobs[i].elempack);
        }

        Mat bottom_blob_unpacked = bottom_blob;
        if (elempack > out_elempack2)
        {
            convert_packing(bottom_blob, bottom_blob_unpacked, out_elempack2, opt);
            if (bottom_blob_unpacked.empty())
                return -100;
        }

        const float* ptr = bottom_blob_unpacked;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            Mat& top_blob = top_blobs[i];

#if __riscv_vector
            if (out_elempack2 == 4 && top_blob.elempack == 8)
            {
                for (int j = 0; j < top_blob.h; j++)
                {
                    const float* r0 = ptr;
                    const float* r1 = ptr + w * 4;

                    float* outptr0 = top_blob.row(j);

                    int n = w;
                    while (n > 0)
                    {
                        size_t vl = __riscv_vsetvl_e32m1(n);

                        vfloat32m1x4_t _p0 = __riscv_vlseg4e32_v_f32m1x4(r0, vl);
                        vfloat32m1x4_t _p1 = __riscv_vlseg4e32_v_f32m1x4(r1, vl);

                        vfloat32m1_t _p00 = __riscv_vget_v_f32m1x4_f32m1(_p0, 0);
                        vfloat32m1_t _p01 = __riscv_vget_v_f32m1x4_f32m1(_p0, 1);
                        vfloat32m1_t _p02 = __riscv_vget_v_f32m1x4_f32m1(_p0, 2);
                        vfloat32m1_t _p03 = __riscv_vget_v_f32m1x4_f32m1(_p0, 3);
                        vfloat32m1_t _p10 = __riscv_vget_v_f32m1x4_f32m1(_p1, 0);
                        vfloat32m1_t _p11 = __riscv_vget_v_f32m1x4_f32m1(_p1, 1);
                        vfloat32m1_t _p12 = __riscv_vget_v_f32m1x4_f32m1(_p1, 2);
                        vfloat32m1_t _p13 = __riscv_vget_v_f32m1x4_f32m1(_p1, 3);

                        __riscv_vsseg8e32_v_f32m1x8(outptr0, __riscv_vcreate_v_f32m1x8(_p00, _p01, _p02, _p03, _p10, _p11, _p12, _p13), vl);

                        r0 += vl * 4;
                        r1 += vl * 4;
                        outptr0 += vl * 8;
                        n -= vl;
                    }

                    ptr += w * 8;
                }
            }
            else if (out_elempack2 == 1 && top_blob.elempack == 8)
            {
                for (int j = 0; j < top_blob.h; j++)
                {
                    const float* r0 = ptr;
                    const float* r1 = ptr + w;
                    const float* r2 = ptr + w * 2;
                    const float* r3 = ptr + w * 3;
                    const float* r4 = ptr + w * 4;
                    const float* r5 = ptr + w * 5;
                    const float* r6 = ptr + w * 6;
                    const float* r7 = ptr + w * 7;

                    float* outptr0 = top_blob.row(j);

                    int n = w;
                    while (n > 0)
                    {
                        size_t vl = __riscv_vsetvl_e32m1(n);

                        vfloat32m1_t _p0 = __riscv_vle32_v_f32m1(r0, vl);
                        vfloat32m1_t _p1 = __riscv_vle32_v_f32m1(r1, vl);
                        vfloat32m1_t _p2 = __riscv_vle32_v_f32m1(r2, vl);
                        vfloat32m1_t _p3 = __riscv_vle32_v_f32m1(r3, vl);
                        vfloat32m1_t _p4 = __riscv_vle32_v_f32m1(r4, vl);
                        vfloat32m1_t _p5 = __riscv_vle32_v_f32m1(r5, vl);
                        vfloat32m1_t _p6 = __riscv_vle32_v_f32m1(r6, vl);
                        vfloat32m1_t _p7 = __riscv_vle32_v_f32m1(r7, vl);
                        __riscv_vsseg8e32_v_f32m1x8(outptr0, __riscv_vcreate_v_f32m1x8(_p0, _p1, _p2, _p3, _p4, _p5, _p6, _p7), vl);

                        r0 += vl;
                        r1 += vl;
                        r2 += vl;
                        r3 += vl;
                        r4 += vl;
                        r5 += vl;
                        r6 += vl;
                        r7 += vl;
                        outptr0 += vl * 8;
                        n -= vl;
                    }

                    ptr += w * 8;
                }
            }
            else if (out_elempack2 == 1 && top_blob.elempack == 4)
            {
                for (int j = 0; j < top_blob.h; j++)
                {
                    const float* r0 = ptr;
                    const float* r1 = ptr + w;
                    const float* r2 = ptr + w * 2;
                    const float* r3 = ptr + w * 3;

                    float* outptr0 = top_blob.row(j);

                    int n = w;
                    while (n > 0)
                    {
                        size_t vl = __riscv_vsetvl_e32m2(n);

                        vfloat32m2_t _p0 = __riscv_vle32_v_f32m2(r0, vl);
                        vfloat32m2_t _p1 = __riscv_vle32_v_f32m2(r1, vl);
                        vfloat32m2_t _p2 = __riscv_vle32_v_f32m2(r2, vl);
                        vfloat32m2_t _p3 = __riscv_vle32_v_f32m2(r3, vl);
                        __riscv_vsseg4e32_v_f32m2x4(outptr0, __riscv_vcreate_v_f32m2x4(_p0, _p1, _p2, _p3), vl);

                        r0 += vl;
                        r1 += vl;
                        r2 += vl;
                        r3 += vl;
                        outptr0 += vl * 4;
                        n -= vl;
                    }

                    ptr += w * 4;
                }
            }
#endif // __riscv_vector
            if (out_elempack2 == top_blob.elempack)
            {
                int size = w * top_blob.h;
                float* outptr = top_blob;
                memcpy(outptr, ptr, size * top_blob.elemsize);
                ptr += size * top_blob.elempack;
            }
        }
    }

    if (dims == 2 && positive_axis == 1)
    {
        // slice image width
        int w = bottom_blob.w;
        int h = bottom_blob.h;

        int q = 0;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            int slice;
            if (indices_ptr)
            {
                if (i == top_blobs.size() - 1)
                {
                    slice = w - q;
                }
                else
                {
                    int indice = indices_ptr[i];
                    int positive_indice = indice < 0 ? w + indice : indice;
                    slice = positive_indice - q;
                }
            }
            else
            {
                slice = slices_ptr[i];
                if (slice == -233)
                {
                    slice = static_cast<int>((w - q) / (top_blobs.size() - i));
                }
            }

            Mat& top_blob = top_blobs[i];
            top_blob.create(slice, h, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            q += slice;
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int j = 0; j < h; j++)
        {
            const float* ptr = bottom_blob.row<const float>(j);
            for (size_t i = 0; i < top_blobs.size(); i++)
            {
                Mat& top_blob = top_blobs[i];

                float* outptr = top_blob.row(j);
                memcpy(outptr, ptr, top_blob.w * elemsize);

                ptr += top_blob.w * elempack;
            }
        }
    }

    if ((dims == 3 || dims == 4) && positive_axis == 0)
    {
        // slice dim channel
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int d = bottom_blob.d;
        int channels = bottom_blob.c * elempack;

        int q = 0;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            int slice;
            if (indices_ptr)
            {
                if (i == top_blobs.size() - 1)
                {
                    slice = channels - q;
                }
                else
                {
                    int indice = indices_ptr[i];
                    int positive_indice = indice < 0 ? channels + indice : indice;
                    slice = positive_indice - q;
                }
            }
            else
            {
                slice = slices_ptr[i];
                if (slice == -233)
                {
                    slice = static_cast<int>((channels - q) / (top_blobs.size() - i));
                }
            }

            int out_elempack = 1;
#if __riscv_vector
            if (opt.use_packing_layout)
            {
                out_elempack = slice % 8 == 0 ? 8 : slice % 4 == 0 ? 4 : 1;
            }
#endif // __riscv_vector
            size_t out_elemsize = elemsize / elempack * out_elempack;

            Mat& top_blob = top_blobs[i];
            top_blob.create(w, h, d, slice / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            top_blob.dims = dims;

            q += slice;
        }

        size_t out_elemsize2 = top_blobs[0].elemsize;
        int out_elempack2 = top_blobs[0].elempack;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            out_elemsize2 = std::min(out_elemsize2, top_blobs[i].elemsize);
            out_elempack2 = std::min(out_elempack2, top_blobs[i].elempack);
        }

        Mat bottom_blob_unpacked = bottom_blob;
        if (elempack > out_elempack2)
        {
            convert_packing(bottom_blob, bottom_blob_unpacked, out_elempack2, opt);
            if (bottom_blob_unpacked.empty())
                return -100;
        }

        int p = 0;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            Mat& top_blob = top_blobs[i];

#if __riscv_vector
            if (out_elempack2 == 4 && top_blob.elempack == 8)
            {
                int size = top_blob.w * top_blob.h * top_blob.d;

                for (int qq = 0; qq < top_blob.c; qq++)
                {
                    const float* r0 = bottom_blob_unpacked.channel(p);
                    const float* r1 = bottom_blob_unpacked.channel(p + 1);

                    float* outptr0 = top_blob.channel(qq);

                    int n = size;
                    while (n > 0)
                    {
                        size_t vl = __riscv_vsetvl_e32m1(n);

                        vfloat32m1x4_t _p0 = __riscv_vlseg4e32_v_f32m1x4(r0, vl);
                        vfloat32m1x4_t _p1 = __riscv_vlseg4e32_v_f32m1x4(r1, vl);

                        vfloat32m1_t _p00 = __riscv_vget_v_f32m1x4_f32m1(_p0, 0);
                        vfloat32m1_t _p01 = __riscv_vget_v_f32m1x4_f32m1(_p0, 1);
                        vfloat32m1_t _p02 = __riscv_vget_v_f32m1x4_f32m1(_p0, 2);
                        vfloat32m1_t _p03 = __riscv_vget_v_f32m1x4_f32m1(_p0, 3);
                        vfloat32m1_t _p10 = __riscv_vget_v_f32m1x4_f32m1(_p1, 0);
                        vfloat32m1_t _p11 = __riscv_vget_v_f32m1x4_f32m1(_p1, 1);
                        vfloat32m1_t _p12 = __riscv_vget_v_f32m1x4_f32m1(_p1, 2);
                        vfloat32m1_t _p13 = __riscv_vget_v_f32m1x4_f32m1(_p1, 3);

                        __riscv_vsseg8e32_v_f32m1x8(outptr0, __riscv_vcreate_v_f32m1x8(_p00, _p01, _p02, _p03, _p10, _p11, _p12, _p13), vl);

                        r0 += vl * 4;
                        r1 += vl * 4;
                        outptr0 += vl * 8;
                        n -= vl;
                    }

                    p += 2;
                }
            }
            else if (out_elempack2 == 1 && top_blob.elempack == 8)
            {
                int size = top_blob.w * top_blob.h * top_blob.d;

                for (int qq = 0; qq < top_blob.c; qq++)
                {
                    const float* r0 = bottom_blob_unpacked.channel(p);
                    const float* r1 = bottom_blob_unpacked.channel(p + 1);
                    const float* r2 = bottom_blob_unpacked.channel(p + 2);
                    const float* r3 = bottom_blob_unpacked.channel(p + 3);
                    const float* r4 = bottom_blob_unpacked.channel(p + 4);
                    const float* r5 = bottom_blob_unpacked.channel(p + 5);
                    const float* r6 = bottom_blob_unpacked.channel(p + 6);
                    const float* r7 = bottom_blob_unpacked.channel(p + 7);

                    float* outptr0 = top_blob.channel(qq);

                    int n = size;
                    while (n > 0)
                    {
                        size_t vl = __riscv_vsetvl_e32m1(n);

                        vfloat32m1_t _p0 = __riscv_vle32_v_f32m1(r0, vl);
                        vfloat32m1_t _p1 = __riscv_vle32_v_f32m1(r1, vl);
                        vfloat32m1_t _p2 = __riscv_vle32_v_f32m1(r2, vl);
                        vfloat32m1_t _p3 = __riscv_vle32_v_f32m1(r3, vl);
                        vfloat32m1_t _p4 = __riscv_vle32_v_f32m1(r4, vl);
                        vfloat32m1_t _p5 = __riscv_vle32_v_f32m1(r5, vl);
                        vfloat32m1_t _p6 = __riscv_vle32_v_f32m1(r6, vl);
                        vfloat32m1_t _p7 = __riscv_vle32_v_f32m1(r7, vl);
                        __riscv_vsseg8e32_v_f32m1x8(outptr0, __riscv_vcreate_v_f32m1x8(_p0, _p1, _p2, _p3, _p4, _p5, _p6, _p7), vl);

                        r0 += vl;
                        r1 += vl;
                        r2 += vl;
                        r3 += vl;
                        r4 += vl;
                        r5 += vl;
                        r6 += vl;
                        r7 += vl;
                        outptr0 += vl * 8;
                        n -= vl;
                    }

                    p += 8;
                }
            }
            else if (out_elempack2 == 1 && top_blob.elempack == 4)
            {
                int size = top_blob.w * top_blob.h * top_blob.d;

                for (int qq = 0; qq < top_blob.c; qq++)
                {
                    const float* r0 = bottom_blob_unpacked.channel(p);
                    const float* r1 = bottom_blob_unpacked.channel(p + 1);
                    const float* r2 = bottom_blob_unpacked.channel(p + 2);
                    const float* r3 = bottom_blob_unpacked.channel(p + 3);

                    float* outptr0 = top_blob.channel(qq);

                    int n = size;
                    while (n > 0)
                    {
                        size_t vl = __riscv_vsetvl_e32m2(n);

                        vfloat32m2_t _pp0 = __riscv_vle32_v_f32m2(r0, vl);
                        vfloat32m2_t _pp1 = __riscv_vle32_v_f32m2(r1, vl);
                        vfloat32m2_t _pp2 = __riscv_vle32_v_f32m2(r2, vl);
                        vfloat32m2_t _pp3 = __riscv_vle32_v_f32m2(r3, vl);
                        __riscv_vsseg4e32_v_f32m2x4(outptr0, __riscv_vcreate_v_f32m2x4(_pp0, _pp1, _pp2, _pp3), vl);

                        r0 += vl;
                        r1 += vl;
                        r2 += vl;
                        r3 += vl;
                        outptr0 += vl * 4;
                        n -= vl;
                    }

                    p += 4;
                }
            }
#endif // __riscv_vector
            if (out_elempack2 == top_blob.elempack)
            {
                int size = top_blob.total();

                const float* ptr2 = bottom_blob_unpacked.channel(p);
                float* outptr = top_blob;
                memcpy(outptr, ptr2, size * top_blob.elemsize);

                p += top_blob.c;
            }
        }
    }

    if ((dims == 3 && positive_axis == 1) || (dims == 4 && positive_axis == 2))
    {
        // slice dim height
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int d = bottom_blob.d;
        int channels = bottom_blob.c;

        int q = 0;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            int slice;
            if (indices_ptr)
            {
                if (i == top_blobs.size() - 1)
                {
                    slice = h - q;
                }
                else
                {
                    int indice = indices_ptr[i];
                    int positive_indice = indice < 0 ? h + indice : indice;
                    slice = positive_indice - q;
                }
            }
            else
            {
                slice = slices_ptr[i];
                if (slice == -233)
                {
                    slice = static_cast<int>((h - q) / (top_blobs.size() - i));
                }
            }

            Mat& top_blob = top_blobs[i];
            top_blob.create(w, slice, d, channels, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            top_blob.dims = dims;

            q += slice;
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int pch = 0; pch < channels; pch++)
        {
            const float* ptr = bottom_blob.channel(pch);

            for (int j = 0; j < d; j++)
            {
                for (size_t i = 0; i < top_blobs.size(); i++)
                {
                    Mat& top_blob = top_blobs[i];

                    int size = top_blob.w * top_blob.h;

                    float* outptr = top_blob.channel(pch).depth(j);
                    memcpy(outptr, ptr, size * elemsize);

                    ptr += size * elempack;
                }
            }
        }
    }

    if ((dims == 3 && positive_axis == 2) || (dims == 4 && positive_axis == 3))
    {
        // slice dim width
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int d = bottom_blob.d;
        int channels = bottom_blob.c;

        int q = 0;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            int slice;
            if (indices_ptr)
            {
                if (i == top_blobs.size() - 1)
                {
                    slice = w - q;
                }
                else
                {
                    int indice = indices_ptr[i];
                    int positive_indice = indice < 0 ? w + indice : indice;
                    slice = positive_indice - q;
                }
            }
            else
            {
                slice = slices_ptr[i];
                if (slice == -233)
                {
                    slice = static_cast<int>((w - q) / (top_blobs.size() - i));
                }
            }

            Mat& top_blob = top_blobs[i];
            top_blob.create(slice, h, d, channels, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            top_blob.dims = dims;

            q += slice;
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int pch = 0; pch < channels; pch++)
        {
            const float* ptr = bottom_blob.channel(pch);

            for (int j = 0; j < d; j++)
            {
                for (int k = 0; k < h; k++)
                {
                    for (size_t i = 0; i < top_blobs.size(); i++)
                    {
                        Mat& top_blob = top_blobs[i];

                        float* outptr = top_blob.channel(pch).depth(j).row(k);
                        memcpy(outptr, ptr, top_blob.w * elemsize);

                        ptr += top_blob.w * elempack;
                    }
                }
            }
        }
    }

    if (dims == 4 && positive_axis == 1)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int d = bottom_blob.d;
        int channels = bottom_blob.c;

        int q = 0;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            int slice;
            if (indices_ptr)
            {
                if (i == top_blobs.size() - 1)
                {
                    slice = d - q;
                }
                else
                {
                    int indice = indices_ptr[i];
                    int positive_indice = indice < 0 ? d + indice : indice;
                    slice = positive_indice - q;
                }
            }
            else
            {
                slice = slices_ptr[i];
                if (slice == -233)
                {
                    slice = static_cast<int>((d - q) / (top_blobs.size() - i));
                }
            }

            Mat& top_blob = top_blobs[i];
            top_blob.create(w, h, slice, channels, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            q += slice;
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int pch = 0; pch < channels; pch++)
        {
            const float* ptr = bottom_blob.channel(pch);

            for (size_t i = 0; i < top_blobs.size(); i++)
            {
                Mat& top_blob = top_blobs[i];

                int size = top_blob.w * top_blob.h * top_blob.d;

                float* outptr = top_blob.channel(pch);
                memcpy(outptr, ptr, size * elemsize);

                ptr += size * elempack;
            }
        }
    }

    return 0;
}

} // namespace ncnn
