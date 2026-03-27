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

#include "reshape_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

#include "riscv_usability.h"

namespace ncnn {

Reshape_riscv::Reshape_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector
}

int Reshape_riscv::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    Mat& top_blob = top_blobs[0];

    // resolve out shape
    int outw = w;
    int outh = h;
    int outd = d;
    int outc = c;

    if (!shape_expr.empty())
    {
        int er = eval_shape_expr(bottom_blobs, outw, outh, outd, outc);
        if (er != 0)
            return -1;
    }

    if (ndim == 1)
    {
        // flatten
        flatten(bottom_blob, top_blob, opt);
        if (top_blob.empty())
            return -100;

        return 0;
    }

    const int dims = bottom_blob.dims;
    const int elempack = bottom_blob.elempack;
    const size_t elemsize = bottom_blob.elemsize;

    const int total = bottom_blob.w * bottom_blob.h * bottom_blob.d * bottom_blob.c * elempack;

    if (ndim == 2)
    {
        if (outw == 0)
            outw = dims == 1 ? bottom_blob.w * elempack : bottom_blob.w;
        if (outh == 0)
            outh = dims == 2 ? bottom_blob.h * elempack : bottom_blob.h;

        if (outw == -1)
            outw = total / outh;
        if (outh == -1)
            outh = total / outw;

        int out_elempack = 1;
#if __riscv_vector
        if (opt.use_packing_layout)
        {
            // prefer pack8 then pack4 on riscv-v
            out_elempack = outh % 8 == 0 ? 8 : outh % 4 == 0 ? 4 : 1;
        }
#endif // __riscv_vector
        size_t out_elemsize = elemsize / elempack * out_elempack;

        if (dims == 2 && bottom_blob.h * elempack == outh && elempack == out_elempack)
        {
            top_blob = bottom_blob;
            return 0;
        }

        if (out_elempack == 1)
        {
            // flatten
            flatten(bottom_blob, top_blob, opt);
            if (top_blob.empty())
                return -100;

            top_blob.dims = 2;
            top_blob.w = outw;
            top_blob.h = outh;
            top_blob.cstep = (size_t)outw * outh;
            top_blob.elemsize = out_elemsize;
            top_blob.elempack = out_elempack;

            return 0;
        }

        // flatten
        Mat bottom_blob_flattened = bottom_blob;
        {
            Option opt_flatten = opt;
            opt_flatten.blob_allocator = opt.workspace_allocator;

            flatten(bottom_blob, bottom_blob_flattened, opt_flatten);
            if (bottom_blob_flattened.empty())
                return -100;
        }

        top_blob.create(outw, outh / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

#if __riscv_vector
        if (out_elempack == 8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < top_blob.h; i++)
            {
                const float* ptr0 = (const float*)bottom_blob_flattened + outw * i * 8;
                const float* ptr1 = (const float*)bottom_blob_flattened + outw * (i * 8 + 1);
                const float* ptr2 = (const float*)bottom_blob_flattened + outw * (i * 8 + 2);
                const float* ptr3 = (const float*)bottom_blob_flattened + outw * (i * 8 + 3);
                const float* ptr4 = (const float*)bottom_blob_flattened + outw * (i * 8 + 4);
                const float* ptr5 = (const float*)bottom_blob_flattened + outw * (i * 8 + 5);
                const float* ptr6 = (const float*)bottom_blob_flattened + outw * (i * 8 + 6);
                const float* ptr7 = (const float*)bottom_blob_flattened + outw * (i * 8 + 7);
                float* outptr = top_blob.row(i);

                int n = outw;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m1(n);
                    vfloat32m1_t _p0 = __riscv_vle32_v_f32m1(ptr0, vl);
                    vfloat32m1_t _p1 = __riscv_vle32_v_f32m1(ptr1, vl);
                    vfloat32m1_t _p2 = __riscv_vle32_v_f32m1(ptr2, vl);
                    vfloat32m1_t _p3 = __riscv_vle32_v_f32m1(ptr3, vl);
                    vfloat32m1_t _p4 = __riscv_vle32_v_f32m1(ptr4, vl);
                    vfloat32m1_t _p5 = __riscv_vle32_v_f32m1(ptr5, vl);
                    vfloat32m1_t _p6 = __riscv_vle32_v_f32m1(ptr6, vl);
                    vfloat32m1_t _p7 = __riscv_vle32_v_f32m1(ptr7, vl);
                    __riscv_vsseg8e32_v_f32m1x8(outptr, __riscv_vcreate_v_f32m1x8(_p0, _p1, _p2, _p3, _p4, _p5, _p6, _p7), vl);

                    ptr0 += vl;
                    ptr1 += vl;
                    ptr2 += vl;
                    ptr3 += vl;
                    ptr4 += vl;
                    ptr5 += vl;
                    ptr6 += vl;
                    ptr7 += vl;
                    outptr += vl * 8;
                    n -= vl;
                }
            }
        }
        if (out_elempack == 4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < top_blob.h; i++)
            {
                const float* ptr0 = (const float*)bottom_blob_flattened + outw * i * 4;
                const float* ptr1 = (const float*)bottom_blob_flattened + outw * (i * 4 + 1);
                const float* ptr2 = (const float*)bottom_blob_flattened + outw * (i * 4 + 2);
                const float* ptr3 = (const float*)bottom_blob_flattened + outw * (i * 4 + 3);
                float* outptr = top_blob.row(i);

                int n = outw;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m2(n);
                    vfloat32m2_t _p0 = __riscv_vle32_v_f32m2(ptr0, vl);
                    vfloat32m2_t _p1 = __riscv_vle32_v_f32m2(ptr1, vl);
                    vfloat32m2_t _p2 = __riscv_vle32_v_f32m2(ptr2, vl);
                    vfloat32m2_t _p3 = __riscv_vle32_v_f32m2(ptr3, vl);
                    __riscv_vsseg4e32_v_f32m2x4(outptr, __riscv_vcreate_v_f32m2x4(_p0, _p1, _p2, _p3), vl);

                    ptr0 += vl;
                    ptr1 += vl;
                    ptr2 += vl;
                    ptr3 += vl;
                    outptr += vl * 4;
                    n -= vl;
                }
            }
        }
#endif // __riscv_vector
    }

    if (ndim == 3 || ndim == 4)
    {
        if (ndim == 3)
        {
            if (outw == 0)
                outw = dims == 1 ? bottom_blob.w * elempack : bottom_blob.w;
            if (outh == 0)
                outh = dims == 2 ? bottom_blob.h * elempack : bottom_blob.h;
            if (outc == 0)
                outc = dims == 3 ? bottom_blob.c * elempack : bottom_blob.c;

            if (outw == -1)
                outw = total / outc / outh;
            if (outh == -1)
                outh = total / outc / outw;
            if (outc == -1)
                outc = total / outh / outw;

            outd = 1;
        }
        else // if (ndim == 4)
        {
            if (outw == 0)
                outw = dims == 1 ? bottom_blob.w * elempack : bottom_blob.w;
            if (outh == 0)
                outh = dims == 2 ? bottom_blob.h * elempack : bottom_blob.h;
            if (outd == 0)
                outd = bottom_blob.d;
            if (outc == 0)
                outc = (dims == 3 || dims == 4) ? bottom_blob.c * elempack : bottom_blob.c;

            if (outw == -1)
                outw = total / outc / outd / outh;
            if (outh == -1)
                outh = total / outc / outd / outw;
            if (outd == -1)
                outd = total / outc / outh / outw;
            if (outc == -1)
                outc = total / outd / outh / outw;
        }

        int out_elempack = 1;
#if __riscv_vector
        if (opt.use_packing_layout)
        {
            // prefer pack8 then pack4 on riscv-v
            out_elempack = outc % 8 == 0 ? 8 : outc % 4 == 0 ? 4 : 1;
        }
#endif // __riscv_vector
        size_t out_elemsize = elemsize / elempack * out_elempack;

        if ((dims == 3 || dims == 4) && bottom_blob.c * elempack == outc && elempack == out_elempack)
        {
            top_blob = bottom_blob;
            top_blob.dims = ndim;
            top_blob.w = outw;
            top_blob.h = outh;
            top_blob.d = outd;
            return 0;
        }

        // flatten
        Mat bottom_blob_flattened = bottom_blob;
        {
            Option opt_flatten = opt;
            opt_flatten.blob_allocator = opt.workspace_allocator;

            flatten(bottom_blob, bottom_blob_flattened, opt_flatten);
            if (bottom_blob_flattened.empty())
                return -100;
        }

        if (ndim == 3)
        {
            top_blob.create(outw, outh, outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        }
        else // if (ndim == 4)
        {
            top_blob.create(outw, outh, outd, outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        }
        if (top_blob.empty())
            return -100;

        int size = top_blob.w * top_blob.h * top_blob.d;

#if __riscv_vector
        if (out_elempack == 8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < top_blob.c; q++)
            {
                const float* ptr0 = (const float*)bottom_blob_flattened + size * q * 8;
                const float* ptr1 = (const float*)bottom_blob_flattened + size * (q * 8 + 1);
                const float* ptr2 = (const float*)bottom_blob_flattened + size * (q * 8 + 2);
                const float* ptr3 = (const float*)bottom_blob_flattened + size * (q * 8 + 3);
                const float* ptr4 = (const float*)bottom_blob_flattened + size * (q * 8 + 4);
                const float* ptr5 = (const float*)bottom_blob_flattened + size * (q * 8 + 5);
                const float* ptr6 = (const float*)bottom_blob_flattened + size * (q * 8 + 6);
                const float* ptr7 = (const float*)bottom_blob_flattened + size * (q * 8 + 7);
                float* outptr = top_blob.channel(q);

                int n = size;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m1(n);
                    vfloat32m1_t _p0 = __riscv_vle32_v_f32m1(ptr0, vl);
                    vfloat32m1_t _p1 = __riscv_vle32_v_f32m1(ptr1, vl);
                    vfloat32m1_t _p2 = __riscv_vle32_v_f32m1(ptr2, vl);
                    vfloat32m1_t _p3 = __riscv_vle32_v_f32m1(ptr3, vl);
                    vfloat32m1_t _p4 = __riscv_vle32_v_f32m1(ptr4, vl);
                    vfloat32m1_t _p5 = __riscv_vle32_v_f32m1(ptr5, vl);
                    vfloat32m1_t _p6 = __riscv_vle32_v_f32m1(ptr6, vl);
                    vfloat32m1_t _p7 = __riscv_vle32_v_f32m1(ptr7, vl);
                    __riscv_vsseg8e32_v_f32m1x8(outptr, __riscv_vcreate_v_f32m1x8(_p0, _p1, _p2, _p3, _p4, _p5, _p6, _p7), vl);

                    ptr0 += vl;
                    ptr1 += vl;
                    ptr2 += vl;
                    ptr3 += vl;
                    ptr4 += vl;
                    ptr5 += vl;
                    ptr6 += vl;
                    ptr7 += vl;
                    outptr += vl * 8;
                    n -= vl;
                }
            }
        }
        if (out_elempack == 4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < top_blob.c; q++)
            {
                const float* ptr0 = (const float*)bottom_blob_flattened + size * q * 4;
                const float* ptr1 = (const float*)bottom_blob_flattened + size * (q * 4 + 1);
                const float* ptr2 = (const float*)bottom_blob_flattened + size * (q * 4 + 2);
                const float* ptr3 = (const float*)bottom_blob_flattened + size * (q * 4 + 3);
                float* outptr = top_blob.channel(q);

                int n = size;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m2(n);
                    vfloat32m2_t _p0 = __riscv_vle32_v_f32m2(ptr0, vl);
                    vfloat32m2_t _p1 = __riscv_vle32_v_f32m2(ptr1, vl);
                    vfloat32m2_t _p2 = __riscv_vle32_v_f32m2(ptr2, vl);
                    vfloat32m2_t _p3 = __riscv_vle32_v_f32m2(ptr3, vl);
                    __riscv_vsseg4e32_v_f32m2x4(outptr, __riscv_vcreate_v_f32m2x4(_p0, _p1, _p2, _p3), vl);

                    ptr0 += vl;
                    ptr1 += vl;
                    ptr2 += vl;
                    ptr3 += vl;
                    outptr += vl * 4;
                    n -= vl;
                }
            }
        }
        if (out_elempack == 1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < top_blob.c; q++)
            {
                const float* ptr = (const float*)bottom_blob_flattened + size * q;
                float* outptr = top_blob.channel(q);

                int n = size;
#if __riscv_vector
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m8(n);
                    vfloat32m8_t _v = __riscv_vle32_v_f32m8(ptr, vl);
                    __riscv_vse32_v_f32m8(outptr, _v, vl);
                    ptr += vl;
                    outptr += vl;
                    n -= vl;
                }
#else
                for (; n > 0; n--)
                {
                    *outptr++ = *ptr++;
                }
#endif // __riscv_vector
            }
        }
#endif // __riscv_vector
    }

    return 0;
}

} // namespace ncnn
