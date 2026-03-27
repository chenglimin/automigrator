// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "cpu.h"

#if __riscv_vector
#include <riscv_vector.h>
#include "riscv_usability.h"
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
        int w_total = bottom_blob.w * elempack;

        // Fallback: unpack -> generic slice -> repack
        if (elempack > 1)
        {
            Mat bottom_blob_p1;
            convert_packing(bottom_blob, bottom_blob_p1, 1, opt);
            if (bottom_blob_p1.empty())
                return -100;

            std::vector<Mat> top_blobs_p1(top_blobs.size());
            int ret = Slice::forward({bottom_blob_p1}, top_blobs_p1, opt);
            if (ret != 0)
                return ret;

            const int packn = csrr_vlenb() / 4;
            for (size_t i = 0; i < top_blobs.size(); i++)
            {
                Mat& tb1 = top_blobs_p1[i];
                int desired = (opt.use_packing_layout && tb1.w % packn == 0) ? packn : 1;
                if (desired == 1)
                {
                    top_blobs[i] = tb1;
                }
                else
                {
                    convert_packing(tb1, top_blobs[i], desired, opt);
                    if (top_blobs[i].empty())
                        return -100;
                }
            }
            return 0;
        }

        int q = 0;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            int slice;
            if (indices_ptr)
            {
                if (i == top_blobs.size() - 1)
                {
                    slice = w_total - q;
                }
                else
                {
                    int indice = indices_ptr[i];
                    int positive_indice = indice < 0 ? w_total + indice : indice;
                    slice = positive_indice - q;
                }
            }
            else
            {
                slice = slices_ptr[i];
                if (slice == -233)
                {
                    slice = static_cast<int>((w_total - q) / (top_blobs.size() - i));
                }
            }

            Mat& top_blob = top_blobs[i];
            top_blob.create(slice, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            const unsigned char* ptr = (const unsigned char*)bottom_blob + (size_t)q * elemsize;
            memcpy((unsigned char*)top_blob.data, ptr, (size_t)slice * elemsize);

            q += slice;
        }
        return 0;
    }

    if (dims == 2 && positive_axis == 0)
    {
        int w = bottom_blob.w;
        int h_total = bottom_blob.h * elempack;

        // Fallback: unpack to elempack=1, slice with generic, then repack per output
        if (elempack > 1)
        {
            Mat bottom_blob_p1;
            convert_packing(bottom_blob, bottom_blob_p1, 1, opt);
            if (bottom_blob_p1.empty())
                return -100;

            std::vector<Mat> top_blobs_p1(top_blobs.size());
            int ret = Slice::forward({bottom_blob_p1}, top_blobs_p1, opt);
            if (ret != 0)
                return ret;

            const int packn = csrr_vlenb() / 4;
            for (size_t i = 0; i < top_blobs.size(); i++)
            {
                Mat& top_blob_p1 = top_blobs_p1[i];
                int desired_elempack = (opt.use_packing_layout && top_blob_p1.h % packn == 0) ? packn : 1;
                if (desired_elempack == 1)
                {
                    top_blobs[i] = top_blob_p1;
                }
                else
                {
                    convert_packing(top_blob_p1, top_blobs[i], desired_elempack, opt);
                    if (top_blobs[i].empty())
                        return -100;
                }
            }
            return 0;
        }

        int q = 0;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            int slice;
            if (indices_ptr)
            {
                if (i == top_blobs.size() - 1)
                {
                    slice = h_total - q;
                }
                else
                {
                    int indice = indices_ptr[i];
                    int positive_indice = indice < 0 ? h_total + indice : indice;
                    slice = positive_indice - q;
                }
            }
            else
            {
                slice = slices_ptr[i];
                if (slice == -233)
                {
                    slice = static_cast<int>((h_total - q) / (top_blobs.size() - i));
                }
            }

            Mat& top_blob = top_blobs[i];
            top_blob.create(w, slice, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            q += slice;
        }

        const unsigned char* ptr = bottom_blob.row<const unsigned char>(0);
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            Mat& top_blob = top_blobs[i];
            int size = w * top_blob.h;
            memcpy((unsigned char*)top_blob.data, ptr, (size_t)size * elemsize);
            ptr += (size_t)size * elemsize;
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
                memcpy(outptr, ptr, (size_t)top_blob.w * elemsize);
                ptr += (size_t)top_blob.w * elempack * (elemsize / elempack);
            }
        }
        return 0;
    }

    if ((dims == 3 || dims == 4) && positive_axis == 0)
    {
        // Fallback robust path per memcopy_map: unpack -> generic slice -> repack
        if (elempack > 1)
        {
            Mat bottom_blob_p1;
            convert_packing(bottom_blob, bottom_blob_p1, 1, opt);
            if (bottom_blob_p1.empty())
                return -100;

            std::vector<Mat> top_blobs_p1(top_blobs.size());
            int ret = Slice::forward({bottom_blob_p1}, top_blobs_p1, opt);
            if (ret != 0)
                return ret;

            const int packn = csrr_vlenb() / 4;
            for (size_t i = 0; i < top_blobs.size(); i++)
            {
                Mat& tb1 = top_blobs_p1[i];
                int desired = (opt.use_packing_layout && tb1.c % packn == 0) ? packn : 1;
                if (desired == 1)
                {
                    top_blobs[i] = tb1;
                }
                else
                {
                    convert_packing(tb1, top_blobs[i], desired, opt);
                    if (top_blobs[i].empty())
                        return -100;
                }
            }
            return 0;
        }

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

            Mat& top_blob = top_blobs[i];
            top_blob.create(w, h, d, slice, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            top_blob.dims = dims;

            q += slice;
        }

        int p = 0;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            Mat& top_blob = top_blobs[i];
            int size = top_blob.total();
            const unsigned char* ptr = bottom_blob.channel(p);
            memcpy((unsigned char*)top_blob.data, ptr, (size_t)size * elemsize);
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
            for (int j = 0; j < d; j++)
            {
                const unsigned char* ptr = bottom_blob.channel(p).depth(j);
                for (size_t i = 0; i < top_blobs.size(); i++)
                {
                    Mat& top_blob = top_blobs[i];
                    int size = top_blob.w * top_blob.h;
                    unsigned char* outptr = top_blob.channel(p).depth(j);
                    memcpy(outptr, ptr, (size_t)size * elemsize);
                    ptr += (size_t)size * elempack * (elemsize / elempack);
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
            const Mat m = bottom_blob.channel(p);
            unsigned char* outptr = 0;
            for (int j = 0; j < d; j++)
            {
                for (int k = 0; k < h; k++)
                {
                    const unsigned char* ptr = m.depth(j).row<const unsigned char>(k);
                    for (size_t i = 0; i < top_blobs.size(); i++)
                    {
                        Mat& top_blob = top_blobs[i];
                        outptr = top_blob.channel(p).depth(j).row<unsigned char>(k);
                        memcpy(outptr, ptr, (size_t)top_blob.w * elemsize);
                        ptr += (size_t)top_blob.w * elempack * (elemsize / elempack);
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
            const unsigned char* ptr = bottom_blob.channel(p);
            for (size_t i = 0; i < top_blobs.size(); i++)
            {
                Mat& top_blob = top_blobs[i];
                int size = top_blob.w * top_blob.h * top_blob.d;
                unsigned char* outptr = top_blob.channel(p);
                memcpy(outptr, ptr, (size_t)size * elemsize);
                ptr += (size_t)size * elempack * (elemsize / elempack);
            }
        }
        return 0;
    }

    return 0;
}

} // namespace ncnn
