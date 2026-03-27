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

#include "concat_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

#include "packing.h"
#include "cpu.h"
#include "riscv_usability.h"

namespace ncnn {

Concat_riscv::Concat_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector
}

static inline int resolve_out_elempack(int elemcount, int elembits, const Option& opt)
{
    int out_elempack = 1;
#if __riscv_vector
    if (opt.use_packing_layout)
    {
        if (elembits == 32)
        {
            const int packn = ncnn::cpu_riscv_vlenb() / 4;
            if (elemcount % packn == 0)
                out_elempack = packn;
            else if (elemcount % 4 == 0)
                out_elempack = 4;
            else
                out_elempack = 1;
        }
        else if (elembits == 16)
        {
            const int packn = ncnn::cpu_riscv_vlenb() / 2;
            if (elemcount % packn == 0)
                out_elempack = packn;
            else if (elemcount % 4 == 0)
                out_elempack = 4;
            else
                out_elempack = 1;
        }
        else if (elembits == 8)
        {
            const int packn = ncnn::cpu_riscv_vlenb() / 1;
            if (elemcount % packn == 0)
                out_elempack = packn;
            else if (elemcount % 8 == 0)
                out_elempack = 8;
            else
                out_elempack = 1;
        }
    }
#endif // __riscv_vector
    return out_elempack;
}

int Concat_riscv::forward(const std::vector<Mat>& _bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    // Fallback strategy for packing: if any input is packed (elempack > 1),
    // unpack all to elempack=1, do vanilla concat, then repack to optimal out_elempack.
    auto needs_unpack = [] (const std::vector<Mat>& bs)->bool {
        int ep = bs[0].elempack;
        if (ep != 1) return true;
        for (size_t i = 1; i < bs.size(); i++)
        {
            if (bs[i].elempack != ep) return true;
        }
        return false;
    };

    std::vector<Mat> bottom_blobs = _bottom_blobs;

    int dims = bottom_blobs[0].dims;
    int positive_axis = axis < 0 ? dims + axis : axis;

    // compute out_elempack decision base on final element-count along concatenation axis
    int elembits = bottom_blobs[0].elembits();

    // If any input is packed, do unpack->concat->repack
    if (needs_unpack(bottom_blobs))
    {
        std::vector<Mat> bottoms_planar(bottom_blobs.size());
        // Unpack each to elempack=1
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            if (bottom_blobs[b].elempack == 1)
            {
                bottoms_planar[b] = bottom_blobs[b];
            }
            else
            {
                convert_packing(bottom_blobs[b], bottoms_planar[b], 1, opt);
            }
        }

        // determine output shape and bytesize then perform vanilla concat identical to Concat::forward
        size_t elemsize = bottoms_planar[0].elemsize;

        if (dims == 1)
        {
            int top_w = 0;
            for (size_t b = 0; b < bottoms_planar.size(); b++) top_w += bottoms_planar[b].w;

            // choose optimal repack
            int out_elempack = resolve_out_elempack(top_w, elembits, opt);

            // create temporary planar top
            Mat top_blob_planar;
            top_blob_planar.create(top_w, elemsize, opt.blob_allocator);
            if (top_blob_planar.empty()) return -100;

            unsigned char* outptr = (unsigned char*)top_blob_planar.data;
            for (size_t b = 0; b < bottoms_planar.size(); b++)
            {
                const Mat& bb = bottoms_planar[b];
                size_t bytes = (size_t)bb.w * elemsize;
                memcpy(outptr, bb.data, bytes);
                outptr += bytes;
            }

            // repack to target elempack
            Mat& top_blob = top_blobs[0];
            if (out_elempack == 1)
            {
                top_blob = top_blob_planar;
            }
            else
            {
                convert_packing(top_blob_planar, top_blob, out_elempack, opt);
            }
            return 0;
        }

        if (dims == 2 && positive_axis == 0)
        {
            int w = bottoms_planar[0].w;
            int top_h = 0;
            for (size_t b = 0; b < bottoms_planar.size(); b++) top_h += bottoms_planar[b].h;

            int out_elempack = resolve_out_elempack(top_h, elembits, opt);

            Mat top_blob_planar;
            top_blob_planar.create(w, top_h, elemsize, opt.blob_allocator);
            if (top_blob_planar.empty()) return -100;

            unsigned char* outptr = (unsigned char*)top_blob_planar.data;
            for (size_t b = 0; b < bottoms_planar.size(); b++)
            {
                const Mat& bb = bottoms_planar[b];
                size_t bytes = (size_t)w * bb.h * elemsize;
                memcpy(outptr, bb.data, bytes);
                outptr += bytes;
            }

            Mat& top_blob = top_blobs[0];
            if (out_elempack == 1)
                top_blob = top_blob_planar;
            else
                convert_packing(top_blob_planar, top_blob, out_elempack, opt);
            return 0;
        }

        if (dims == 2 && positive_axis == 1)
        {
            int h = bottoms_planar[0].h;
            int top_w = 0;
            for (size_t b = 0; b < bottoms_planar.size(); b++) top_w += bottoms_planar[b].w;

            Mat& top_blob = top_blobs[0];
            top_blob.create(top_w, h, elemsize, opt.blob_allocator);
            if (top_blob.empty()) return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                unsigned char* outptr = (unsigned char*)top_blob.row<unsigned char>(i);
                for (size_t b = 0; b < bottoms_planar.size(); b++)
                {
                    const Mat& bb = bottoms_planar[b];
                    const unsigned char* ptr = (const unsigned char*)bb.row<const unsigned char>(i);
                    memcpy(outptr, ptr, (size_t)bb.w * elemsize);
                    outptr += (size_t)bb.w * elemsize;
                }
            }
            return 0;
        }

        if ((dims == 3 || dims == 4) && positive_axis == 0)
        {
            int w = bottoms_planar[0].w;
            int h = bottoms_planar[0].h;
            int d = bottoms_planar[0].d;
            int top_channels = 0;
            for (size_t b = 0; b < bottoms_planar.size(); b++) top_channels += bottoms_planar[b].c;

            int out_elempack = resolve_out_elempack(top_channels, elembits, opt);

            Mat top_blob_planar;
            top_blob_planar.create(w, h, d, top_channels, elemsize, opt.blob_allocator);
            if (top_blob_planar.empty()) return -100;
            top_blob_planar.dims = dims;

            int q = 0;
            for (size_t b = 0; b < bottoms_planar.size(); b++)
            {
                const Mat& bb = bottoms_planar[b];
                int channels = bb.c;
                size_t size = (size_t)bb.cstep * channels;
                const unsigned char* ptr = (const unsigned char*)bb.data;
                unsigned char* outptr = (unsigned char*)top_blob_planar.channel(q).data;
                memcpy(outptr, ptr, size * elemsize);
                q += channels;
            }

            Mat& top_blob = top_blobs[0];
            if (out_elempack == 1)
            {
                top_blob = top_blob_planar;
            }
            else
            {
                convert_packing(top_blob_planar, top_blob, out_elempack, opt);
            }
            return 0;
        }

        if ((dims == 3 && positive_axis == 1) || (dims == 4 && positive_axis == 2))
        {
            int w = bottoms_planar[0].w;
            int d = bottoms_planar[0].d;
            int channels = bottoms_planar[0].c;
            int top_h = 0;
            for (size_t b = 0; b < bottoms_planar.size(); b++) top_h += bottoms_planar[b].h;

            Mat& top_blob = top_blobs[0];
            top_blob.create(w, top_h, d, channels, elemsize, opt.blob_allocator);
            if (top_blob.empty()) return -100;
            top_blob.dims = dims;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                unsigned char* outptr = (unsigned char*)top_blob.channel(q).data;
                for (int i = 0; i < d; i++)
                {
                    for (size_t b = 0; b < bottoms_planar.size(); b++)
                    {
                        const Mat& bb = bottoms_planar[b];
                        int size = bb.w * bb.h;
                        const unsigned char* ptr = (const unsigned char*)bb.channel(q).depth(i).data;
                        memcpy(outptr, ptr, (size_t)size * elemsize);
                        outptr += (size_t)size * elemsize;
                    }
                }
            }
            return 0;
        }

        if ((dims == 3 && positive_axis == 2) || (dims == 4 && positive_axis == 3))
        {
            int h = bottoms_planar[0].h;
            int d = bottoms_planar[0].d;
            int channels = bottoms_planar[0].c;
            int top_w = 0;
            for (size_t b = 0; b < bottoms_planar.size(); b++) top_w += bottoms_planar[b].w;

            Mat& top_blob = top_blobs[0];
            top_blob.create(top_w, h, d, channels, elemsize, opt.blob_allocator);
            if (top_blob.empty()) return -100;
            top_blob.dims = dims;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                unsigned char* outptr = (unsigned char*)top_blob.channel(q).data;
                for (int i = 0; i < d; i++)
                {
                    for (int j = 0; j < h; j++)
                    {
                        for (size_t b = 0; b < bottoms_planar.size(); b++)
                        {
                            const Mat& bb = bottoms_planar[b];
                            const unsigned char* ptr = (const unsigned char*)bb.channel(q).depth(i).row<const unsigned char>(j);
                            memcpy(outptr, ptr, (size_t)bb.w * elemsize);
                            outptr += (size_t)bb.w * elemsize;
                        }
                    }
                }
            }
            return 0;
        }

        if (dims == 4 && positive_axis == 1)
        {
            int w = bottoms_planar[0].w;
            int h = bottoms_planar[0].h;
            int channels = bottoms_planar[0].c;
            int top_d = 0;
            for (size_t b = 0; b < bottoms_planar.size(); b++) top_d += bottoms_planar[b].d;

            Mat& top_blob = top_blobs[0];
            top_blob.create(w, h, top_d, channels, elemsize, opt.blob_allocator);
            if (top_blob.empty()) return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                unsigned char* outptr = (unsigned char*)top_blob.channel(q).data;
                for (size_t b = 0; b < bottoms_planar.size(); b++)
                {
                    const Mat& bb = bottoms_planar[b];
                    int size = bb.w * bb.h * bb.d;
                    const unsigned char* ptr = (const unsigned char*)bb.channel(q).data;
                    memcpy(outptr, ptr, (size_t)size * elemsize);
                    outptr += (size_t)size * elemsize;
                }
            }
            return 0;
        }

        return 0;
    }

    // Fast path when all inputs are planar already (elempack == 1)
    if (dims == 1)
    {
        size_t elemsize = bottom_blobs[0].elemsize;
        int top_w = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++) top_w += bottom_blobs[b].w;

        Mat& top_blob = top_blobs[0];
        top_blob.create(top_w, elemsize, opt.blob_allocator);
        if (top_blob.empty()) return -100;

        unsigned char* outptr = (unsigned char*)top_blob.data;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bb = bottom_blobs[b];
            size_t bytes = (size_t)bb.w * elemsize;
            memcpy(outptr, bb.data, bytes);
            outptr += bytes;
        }
        return 0;
    }

    if (dims == 2 && positive_axis == 0)
    {
        int w = bottom_blobs[0].w;
        size_t elemsize = bottom_blobs[0].elemsize;
        int top_h = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++) top_h += bottom_blobs[b].h;

        Mat& top_blob = top_blobs[0];
        top_blob.create(w, top_h, elemsize, opt.blob_allocator);
        if (top_blob.empty()) return -100;

        unsigned char* outptr = (unsigned char*)top_blob.data;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bb = bottom_blobs[b];
            size_t bytes = (size_t)w * bb.h * elemsize;
            memcpy(outptr, bb.data, bytes);
            outptr += bytes;
        }
        return 0;
    }

    if (dims == 2 && positive_axis == 1)
    {
        int h = bottom_blobs[0].h;
        size_t elemsize = bottom_blobs[0].elemsize;

        int top_w = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++) top_w += bottom_blobs[b].w;

        Mat& top_blob = top_blobs[0];
        top_blob.create(top_w, h, elemsize, opt.blob_allocator);
        if (top_blob.empty()) return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            unsigned char* outptr = (unsigned char*)top_blob.row<unsigned char>(i);
            for (size_t b = 0; b < bottom_blobs.size(); b++)
            {
                const Mat& bb = bottom_blobs[b];
                const unsigned char* ptr = (const unsigned char*)bb.row<const unsigned char>(i);
                memcpy(outptr, ptr, (size_t)bb.w * elemsize);
                outptr += (size_t)bb.w * elemsize;
            }
        }
        return 0;
    }

    if ((dims == 3 || dims == 4) && positive_axis == 0)
    {
        int w = bottom_blobs[0].w;
        int h = bottom_blobs[0].h;
        int d = bottom_blobs[0].d;

        size_t elemsize = bottom_blobs[0].elemsize;
        int top_channels = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++) top_channels += bottom_blobs[b].c;

        Mat& top_blob = top_blobs[0];
        top_blob.create(w, h, d, top_channels, elemsize, opt.blob_allocator);
        if (top_blob.empty()) return -100;
        top_blob.dims = dims;

        int q = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bb = bottom_blobs[b];
            int channels = bb.c;
            size_t size = (size_t)bb.cstep * channels;
            const unsigned char* ptr = (const unsigned char*)bb.data;
            unsigned char* outptr = (unsigned char*)top_blob.channel(q).data;
            memcpy(outptr, ptr, size * elemsize);
            q += channels;
        }
        return 0;
    }

    if ((dims == 3 && positive_axis == 1) || (dims == 4 && positive_axis == 2))
    {
        int w = bottom_blobs[0].w;
        int d = bottom_blobs[0].d;
        int channels = bottom_blobs[0].c;

        int top_h = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++) top_h += bottom_blobs[b].h;

        Mat& top_blob = top_blobs[0];
        top_blob.create(w, top_h, d, channels, bottom_blobs[0].elemsize, opt.blob_allocator);
        if (top_blob.empty()) return -100;
        top_blob.dims = dims;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            unsigned char* outptr = (unsigned char*)top_blob.channel(q).data;
            for (int i = 0; i < d; i++)
            {
                for (size_t b = 0; b < bottom_blobs.size(); b++)
                {
                    const Mat& bb = bottom_blobs[b];
                    int size = bb.w * bb.h;
                    const unsigned char* ptr = (const unsigned char*)bb.channel(q).depth(i).data;
                    memcpy(outptr, ptr, (size_t)size * bottom_blobs[0].elemsize);
                    outptr += (size_t)size * bottom_blobs[0].elemsize;
                }
            }
        }
        return 0;
    }

    if ((dims == 3 && positive_axis == 2) || (dims == 4 && positive_axis == 3))
    {
        int h = bottom_blobs[0].h;
        int d = bottom_blobs[0].d;
        int channels = bottom_blobs[0].c;

        int top_w = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++) top_w += bottom_blobs[b].w;

        Mat& top_blob = top_blobs[0];
        top_blob.create(top_w, h, d, channels, bottom_blobs[0].elemsize, opt.blob_allocator);
        if (top_blob.empty()) return -100;
        top_blob.dims = dims;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            unsigned char* outptr = (unsigned char*)top_blob.channel(q).data;
            for (int i = 0; i < d; i++)
            {
                for (int j = 0; j < h; j++)
                {
                    for (size_t b = 0; b < bottom_blobs.size(); b++)
                    {
                        const Mat& bb = bottom_blobs[b];
                        const unsigned char* ptr = (const unsigned char*)bb.channel(q).depth(i).row<const unsigned char>(j);
                        memcpy(outptr, ptr, (size_t)bb.w * bottom_blobs[0].elemsize);
                        outptr += (size_t)bb.w * bottom_blobs[0].elemsize;
                    }
                }
            }
        }
        return 0;
    }

    if (dims == 4 && positive_axis == 1)
    {
        int w = bottom_blobs[0].w;
        int h = bottom_blobs[0].h;
        int channels = bottom_blobs[0].c;

        int top_d = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++) top_d += bottom_blobs[b].d;

        Mat& top_blob = top_blobs[0];
        top_blob.create(w, h, top_d, channels, bottom_blobs[0].elemsize, opt.blob_allocator);
        if (top_blob.empty()) return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            unsigned char* outptr = (unsigned char*)top_blob.channel(q).data;
            for (size_t b = 0; b < bottom_blobs.size(); b++)
            {
                const Mat& bb = bottom_blobs[b];
                int size = bb.w * bb.h * bb.d;
                const unsigned char* ptr = (const unsigned char*)bb.channel(q).data;
                memcpy(outptr, ptr, (size_t)size * bottom_blobs[0].elemsize);
                outptr += (size_t)size * bottom_blobs[0].elemsize;
            }
        }
        return 0;
    }

    return 0;
}

} // namespace ncnn
