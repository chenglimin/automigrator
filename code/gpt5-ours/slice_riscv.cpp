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

#include "slice_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

#include "riscv_usability.h"

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

    // fast path: if input is planar, use generic implementation to ensure semantics
    if (elempack == 1)
    {
        return Slice::forward(bottom_blobs, top_blobs, opt);
    }

    // robust fallback for packed input: unpack -> base slice -> repack per packn
    if (elempack > 1)
    {
        Mat bottom_blob_unpacked;
        convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt);
        if (bottom_blob_unpacked.empty())
            return -100;

        std::vector<Mat> bb(1);
        bb[0] = bottom_blob_unpacked;
        std::vector<Mat> tmp_tops(top_blobs.size());
        int retb = Slice::forward(bb, tmp_tops, opt);
        if (retb != 0)
            return retb;

        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            int desired_elempack = 1;
#if __riscv_vector
            if (opt.use_packing_layout)
            {
                int packn = csrr_vlenb() / 4;
                if (packn > 8) packn = 8;
                int axis_len = 0;
                if (dims == 1 && positive_axis == 0)
                    axis_len = tmp_tops[i].w;
                else if (dims == 2 && positive_axis == 0)
                    axis_len = tmp_tops[i].h;
                else if ((dims == 3 || dims == 4) && positive_axis == 0)
                    axis_len = tmp_tops[i].c;
                else
                {
                    desired_elempack = elempack;
                }
                if (axis_len > 0)
                {
                    if (axis_len % packn == 0)
                        desired_elempack = packn;
                    else if (axis_len % 8 == 0)
                        desired_elempack = 8;
                    else if (axis_len % 4 == 0)
                        desired_elempack = 4;
                    else
                        desired_elempack = 1;
                }
            }
#endif // __riscv_vector
            Mat repacked;
            convert_packing(tmp_tops[i], repacked, desired_elempack, opt);
            if (repacked.empty())
                return -100;
            top_blobs[i] = repacked;
        }
        return 0;
    }

    if (dims == 1) // positive_axis == 0
    // elempack == 1: run base slice and repack outputs for axis 0
    {
        std::vector<Mat> bb2(1);
        bb2[0] = bottom_blob;
        // reuse top_blobs container
        int ret2 = Slice::forward(bb2, top_blobs, opt);
        if (ret2 != 0)
            return ret2;

        if (positive_axis == 0)
        {
            for (size_t i = 0; i < top_blobs.size(); i++)
            {
                int axis_len = 0;
                if (dims == 1)
                    axis_len = top_blobs[i].w;
                else if (dims == 2)
                    axis_len = top_blobs[i].h;
                else
                    axis_len = top_blobs[i].c;

                int desired_elempack = 1;
    #if __riscv_vector
                if (opt.use_packing_layout)
                {
                    int packn = csrr_vlenb() / 4;
                    if (packn > 8) packn = 8;
                    if (axis_len % packn == 0)
                        desired_elempack = packn;
                    else if (axis_len % 8 == 0)
                        desired_elempack = 8;
                    else if (axis_len % 4 == 0)
                        desired_elempack = 4;
                    else
                        desired_elempack = 1;
                }
    #endif // __riscv_vector
                if (top_blobs[i].elempack != desired_elempack)
                {
                    Mat repacked;
                    convert_packing(top_blobs[i], repacked, desired_elempack, opt);
                    if (repacked.empty())
                        return -100;
                    top_blobs[i] = repacked;
                }
            }
        }
        return 0;
    }



    if (dims == 2 && positive_axis == 0)
    {
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
                const int packn = csrr_vlenb() / 4;
                out_elempack = slice % packn == 0 ? packn : 1;
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

        const unsigned char* ptr = bottom_blob_unpacked;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            Mat& top_blob = top_blobs[i];
            int size = w * top_blob.h;
            if (out_elempack2 == top_blob.elempack)
            {
                unsigned char* outptr = top_blob;
                memcpy(outptr, ptr, size * top_blob.elemsize);
                ptr += size * out_elempack2 * (top_blob.elemsize / top_blob.elempack);
            }
            else
            {
                Mat tmp;
                tmp.create(w, top_blob.h, elemsize / elempack * out_elempack2, out_elempack2, opt.blob_allocator);
                if (tmp.empty())
                    return -100;
                unsigned char* tmpptr = tmp;
                memcpy(tmpptr, ptr, size * tmp.elemsize);
                ptr += size * out_elempack2 * (tmp.elemsize / tmp.elempack);
                Mat repacked;
                convert_packing(tmp, repacked, top_blob.elempack, opt);
                if (repacked.empty())
                    return -100;
                memcpy((unsigned char*)top_blob, (unsigned char*)repacked, size * top_blob.elemsize);
            }
        }

        return 0;
    }

    if (dims == 2 && positive_axis == 1)
    {
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
            const unsigned char* ptr = bottom_blob.row<const unsigned char>(j);
            for (size_t i = 0; i < top_blobs.size(); i++)
            {
                Mat& top_blob = top_blobs[i];

                unsigned char* outptr = top_blob.row<unsigned char>(j);
                memcpy(outptr, ptr, top_blob.w * elemsize);

                ptr += top_blob.w * elempack * (elemsize / elempack);
            }
        }

        return 0;
    }

    if ((dims == 3 || dims == 4) && positive_axis == 0)
    {
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
                const int packn = csrr_vlenb() / 4;
                out_elempack = slice % packn == 0 ? packn : 1;
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
            int size = top_blob.total();

            if (out_elempack2 == top_blob.elempack)
            {
                const unsigned char* ptr = (const unsigned char*)bottom_blob_unpacked.channel(p);
                unsigned char* outptr = top_blob;
                memcpy(outptr, ptr, size * top_blob.elemsize);
            }
            else
            {
                Mat tmp;
                tmp.create(w, h, d, top_blob.c, elemsize / elempack * out_elempack2, out_elempack2, opt.blob_allocator);
                if (tmp.empty())
                    return -100;
                int sizech = w * h * d;
                for (int q = 0; q < tmp.c; q++)
                {
                    const unsigned char* src = (const unsigned char*)bottom_blob_unpacked.channel(p + q);
                    unsigned char* dst = (unsigned char*)tmp.channel(q);
                    memcpy(dst, src, sizech * tmp.elemsize);
                }
                Mat repacked;
                convert_packing(tmp, repacked, top_blob.elempack, opt);
                if (repacked.empty())
                    return -100;
                memcpy((unsigned char*)top_blob, (unsigned char*)repacked, size * top_blob.elemsize);
            }

            p += top_blob.c;
        }

        return 0;
    }

    if ((dims == 3 && positive_axis == 1) || (dims == 4 && positive_axis == 2))
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
        for (int p = 0; p < channels; p++)
        {
            const unsigned char* ptr = (const unsigned char*)bottom_blob.channel(p);

            for (int j = 0; j < d; j++)
            {
                for (size_t i = 0; i < top_blobs.size(); i++)
                {
                    Mat& top_blob = top_blobs[i];

                    int size = top_blob.w * top_blob.h;

                    unsigned char* outptr = (unsigned char*)top_blob.channel(p).depth(j);
                    memcpy(outptr, ptr, size * elemsize);

                    ptr += size * elempack * (elemsize / elempack);
                }
            }
        }

        return 0;
    }

    if ((dims == 3 && positive_axis == 2) || (dims == 4 && positive_axis == 3))
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
        for (int p = 0; p < channels; p++)
        {
            const unsigned char* ptr = (const unsigned char*)bottom_blob.channel(p);

            for (int j = 0; j < d; j++)
            {
                for (int k = 0; k < h; k++)
                {
                    for (size_t i = 0; i < top_blobs.size(); i++)
                    {
                        Mat& top_blob = top_blobs[i];

                        unsigned char* outptr = (unsigned char*)top_blob.channel(p).depth(j).row(k);
                        memcpy(outptr, ptr, top_blob.w * elemsize);

                        ptr += top_blob.w * elempack * (elemsize / elempack);
                    }
                }
            }
        }

        return 0;
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
        for (int p = 0; p < channels; p++)
        {
            const unsigned char* ptr = (const unsigned char*)bottom_blob.channel(p);

            for (size_t i = 0; i < top_blobs.size(); i++)
            {
                Mat& top_blob = top_blobs[i];

                int size = top_blob.w * top_blob.h * top_blob.d;

                unsigned char* outptr = (unsigned char*)top_blob.channel(p);
                memcpy(outptr, ptr, size * elemsize);

                ptr += size * elempack * (elemsize / elempack);
            }
        }

        return 0;
    }

    return 0;
}

} // namespace ncnn
