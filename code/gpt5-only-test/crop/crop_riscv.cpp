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

#include "crop_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

namespace ncnn {

Crop_riscv::Crop_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector
}

#if __riscv_vector
static inline void crop_packN_rvv(const Mat& src, Mat& dst, int top, int left)
{
    // copy contiguous blocks by rvv vector load/store
    // src and dst have same elempack
    const int elempack = src.elempack;
    const int w = dst.w;
    const int h = dst.h;

    for (int y = 0; y < h; y++)
    {
        const float* sptr = src.row(top + y) + left * elempack;
        float* dptr = dst.row(y);

        int n = w * elempack;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(n);
            vfloat32m8_t _p = __riscv_vle32_v_f32m8(sptr, vl);
            __riscv_vse32_v_f32m8(dptr, _p, vl);
            sptr += vl;
            dptr += vl;
            n -= vl;
        }
    }
}
#endif // __riscv_vector

int Crop_riscv::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

#if __riscv_vector
    int _woffset, _hoffset, _doffset, _coffset;
    int _outw, _outh, _outd, _outc;
    if (!starts_expr.empty() && !ends_expr.empty())
    {
        std::vector<Mat> bottom_blob_shapes(1);
        bottom_blob_shapes[0] = bottom_blob.shape();
        eval_crop_expr(bottom_blob_shapes, _woffset, _hoffset, _doffset, _coffset, _outw, _outh, _outd, _outc);
    }
    else
    {
        resolve_crop_roi(bottom_blob.shape(), _woffset, _hoffset, _doffset, _coffset, _outw, _outh, _outd, _outc);
    }

    if (elempack == 8)
    {
        if (dims == 1)
        {
            int out_elempack = _outw % 8 == 0 ? 8 : _outw % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (_outw / out_elempack == w && out_elempack == 8)
            {
                top_blob = bottom_blob;
                return 0;
            }

            if (_woffset % 8 == 0 && out_elempack == 8)
            {
                top_blob.create(_outw / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

#if __riscv_vector
                crop_packN_rvv(bottom_blob, top_blob, 0, _woffset / elempack);
#else
                const float* sptr = bottom_blob.row(0) + (_woffset / elempack) * elempack;
                float* dptr = top_blob.row(0);
                for (int i = 0; i < top_blob.w * elempack; i++) dptr[i] = sptr[i];
#endif
                return 0;
            }
        }

        if (dims == 2)
        {
            int out_elempack = _outh % 8 == 0 ? 8 : _outh % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (_outw == w && _outh / out_elempack == h && out_elempack == 8)
            {
                top_blob = bottom_blob;
                return 0;
            }

            if (_hoffset % 8 == 0 && out_elempack == 8)
            {
                top_blob.create(_outw, _outh / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

#if __riscv_vector
                crop_packN_rvv(bottom_blob, top_blob, _hoffset / elempack, _woffset);
#else
                for (int y = 0; y < top_blob.h; y++)
                {
                    const float* sptr = bottom_blob.row(y + _hoffset / elempack) + _woffset * elempack;
                    float* dptr = top_blob.row(y);
                    memcpy(dptr, sptr, top_blob.w * elempack * sizeof(float));
                }
#endif
                return 0;
            }
        }

        if (dims == 3)
        {
            int out_elempack = _outc % 8 == 0 ? 8 : _outc % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (_outw == w && _outh == h && _outc / out_elempack == channels && out_elempack == 8)
            {
                top_blob = bottom_blob;
                return 0;
            }

            if (_coffset % 8 == 0 && out_elempack == 8)
            {
                const Mat bottom_blob_sliced = bottom_blob.channel_range(_coffset / out_elempack, _outc / out_elempack);

                if (_outw == w && _outh == h)
                {
                    top_blob = bottom_blob_sliced.clone(opt.blob_allocator);
                    if (top_blob.empty())
                        return -100;
                }

                top_blob.create(_outw, _outh, _outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < top_blob.c; q++)
                {
                    const Mat m = bottom_blob_sliced.channel(q);
                    Mat borderm = top_blob.channel(q);
#if __riscv_vector
                    crop_packN_rvv(m, borderm, _hoffset, _woffset);
#else
                    for (int y = 0; y < borderm.h; y++)
                    {
                        const float* sptr = m.row(y + _hoffset) + _woffset * elempack;
                        float* dptr = borderm.row(y);
                        memcpy(dptr, sptr, borderm.w * elempack * sizeof(float));
                    }
#endif
                }

                return 0;
            }
        }

        if (dims == 4)
        {
            int out_elempack = _outc % 8 == 0 ? 8 : _outc % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (_outw == w && _outh == h && _outd == d && _outc / out_elempack == channels && out_elempack == 8)
            {
                top_blob = bottom_blob;
                return 0;
            }

            if (_coffset % 8 == 0 && out_elempack == 8)
            {
                const Mat bottom_blob_sliced = bottom_blob.channel_range(_coffset / out_elempack, _outc / out_elempack);

                if (_outw == w && _outh == h && _outd == d)
                {
                    top_blob = bottom_blob_sliced.clone(opt.blob_allocator);
                    if (top_blob.empty())
                        return -100;
                }

                top_blob.create(_outw, _outh, _outd, _outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < top_blob.c; q++)
                {
                    for (int z = 0; z < _outd; z++)
                    {
                        const Mat m = bottom_blob_sliced.channel(q).depth(z + _doffset);
                        Mat borderm = top_blob.channel(q).depth(z);
#if __riscv_vector
                        crop_packN_rvv(m, borderm, _hoffset, _woffset);
#else
                        for (int y = 0; y < borderm.h; y++)
                        {
                            const float* sptr = m.row(y + _hoffset) + _woffset * elempack;
                            float* dptr = borderm.row(y);
                            memcpy(dptr, sptr, borderm.w * elempack * sizeof(float));
                        }
#endif
                    }
                }

                return 0;
            }
        }
    }

    if (elempack == 4)
    {
        if (dims == 1)
        {
            int out_elempack = _outw % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (_outw / out_elempack == w && out_elempack == 4)
            {
                top_blob = bottom_blob;
                return 0;
            }

            if (_woffset % 4 == 0 && out_elempack == 4)
            {
                top_blob.create(_outw / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

#if __riscv_vector
                crop_packN_rvv(bottom_blob, top_blob, 0, _woffset / elempack);
#else
                const float* sptr = bottom_blob.row(0) + (_woffset / elempack) * elempack;
                float* dptr = top_blob.row(0);
                for (int i = 0; i < top_blob.w * elempack; i++) dptr[i] = sptr[i];
#endif
                return 0;
            }
        }

        if (dims == 2)
        {
            int out_elempack = _outh % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (_outw == w && _outh / out_elempack == h && out_elempack == 4)
            {
                top_blob = bottom_blob;
                return 0;
            }

            if (_hoffset % 4 == 0 && out_elempack == 4)
            {
                top_blob.create(_outw, _outh / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

#if __riscv_vector
                crop_packN_rvv(bottom_blob, top_blob, _hoffset / elempack, _woffset);
#else
                for (int y = 0; y < top_blob.h; y++)
                {
                    const float* sptr = bottom_blob.row(y + _hoffset / elempack) + _woffset * elempack;
                    float* dptr = top_blob.row(y);
                    memcpy(dptr, sptr, top_blob.w * elempack * sizeof(float));
                }
#endif
                return 0;
            }
        }

        if (dims == 3)
        {
            int out_elempack = _outc % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (_outw == w && _outh == h && _outc / out_elempack == channels && out_elempack == 4)
            {
                top_blob = bottom_blob;
                return 0;
            }

            if (_coffset % 4 == 0 && out_elempack == 4)
            {
                const Mat bottom_blob_sliced = bottom_blob.channel_range(_coffset / out_elempack, _outc / out_elempack);

                if (_outw == w && _outh == h)
                {
                    top_blob = bottom_blob_sliced.clone(opt.blob_allocator);
                    if (top_blob.empty())
                        return -100;
                }

                top_blob.create(_outw, _outh, _outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < top_blob.c; q++)
                {
                    const Mat m = bottom_blob_sliced.channel(q);
                    Mat borderm = top_blob.channel(q);
#if __riscv_vector
                    crop_packN_rvv(m, borderm, _hoffset, _woffset);
#else
                    for (int y = 0; y < borderm.h; y++)
                    {
                        const float* sptr = m.row(y + _hoffset) + _woffset * elempack;
                        float* dptr = borderm.row(y);
                        memcpy(dptr, sptr, borderm.w * elempack * sizeof(float));
                    }
#endif
                }

                return 0;
            }
        }

        if (dims == 4)
        {
            int out_elempack = _outc % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (_outw == w && _outh == h && _outd == d && _outc / out_elempack == channels && out_elempack == 4)
            {
                top_blob = bottom_blob;
                return 0;
            }

            if (_coffset % 4 == 0 && out_elempack == 4)
            {
                const Mat bottom_blob_sliced = bottom_blob.channel_range(_coffset / out_elempack, _outc / out_elempack);

                if (_outw == w && _outh == h && _outd == d)
                {
                    top_blob = bottom_blob_sliced.clone(opt.blob_allocator);
                    if (top_blob.empty())
                        return -100;
                }

                top_blob.create(_outw, _outh, _outd, _outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < top_blob.c; q++)
                {
                    for (int z = 0; z < _outd; z++)
                    {
                        const Mat m = bottom_blob_sliced.channel(q).depth(z + _doffset);
                        Mat borderm = top_blob.channel(q).depth(z);
#if __riscv_vector
                        crop_packN_rvv(m, borderm, _hoffset, _woffset);
#else
                        for (int y = 0; y < borderm.h; y++)
                        {
                            const float* sptr = m.row(y + _hoffset) + _woffset * elempack;
                            float* dptr = borderm.row(y);
                            memcpy(dptr, sptr, borderm.w * elempack * sizeof(float));
                        }
#endif
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

    return Crop::forward(bottom_blob_unpacked, top_blob, opt);
}

int Crop_riscv::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& reference_blob = bottom_blobs[1];

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    Mat& top_blob = top_blobs[0];

#if __riscv_vector
    int _woffset, _hoffset, _doffset, _coffset;
    int _outw, _outh, _outd, _outc;
    if (!starts_expr.empty() && !ends_expr.empty())
    {
        std::vector<Mat> bottom_blob_shapes(bottom_blobs.size());
        for (size_t i = 0; i < bottom_blobs.size(); i++)
        {
            bottom_blob_shapes[i] = bottom_blobs[i].shape();
        }
        eval_crop_expr(bottom_blob_shapes, _woffset, _hoffset, _doffset, _coffset, _outw, _outh, _outd, _outc);
    }
    else if (woffset == -233)
    {
        resolve_crop_roi(bottom_blob.shape(), (const int*)reference_blob, _woffset, _hoffset, _doffset, _coffset, _outw, _outh, _outd, _outc);
    }
    else
    {
        resolve_crop_roi(bottom_blob.shape(), reference_blob.shape(), _woffset, _hoffset, _doffset, _coffset, _outw, _outh, _outd, _outc);
    }

    if (elempack == 8)
    {
        if (dims == 1)
        {
            int out_elempack = _outw % 8 == 0 ? 8 : _outw % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (_outw / out_elempack == w && out_elempack == 8)
            {
                top_blob = bottom_blob;
                return 0;
            }

            if (_woffset % 8 == 0 && out_elempack == 8)
            {
                top_blob.create(_outw / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

#if __riscv_vector
                crop_packN_rvv(bottom_blob, top_blob, 0, _woffset / elempack);
#else
                const float* sptr = bottom_blob.row(0) + (_woffset / elempack) * elempack;
                float* dptr = top_blob.row(0);
                for (int i = 0; i < top_blob.w * elempack; i++) dptr[i] = sptr[i];
#endif
                return 0;
            }
        }

        if (dims == 2)
        {
            int out_elempack = _outh % 8 == 0 ? 8 : _outh % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (_outw == w && _outh / out_elempack == h && out_elempack == 8)
            {
                top_blob = bottom_blob;
                return 0;
            }

            if (_hoffset % 8 == 0 && out_elempack == 8)
            {
                top_blob.create(_outw, _outh / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

#if __riscv_vector
                crop_packN_rvv(bottom_blob, top_blob, _hoffset / elempack, _woffset);
#else
                for (int y = 0; y < top_blob.h; y++)
                {
                    const float* sptr = bottom_blob.row(y + _hoffset / elempack) + _woffset * elempack;
                    float* dptr = top_blob.row(y);
                    memcpy(dptr, sptr, top_blob.w * elempack * sizeof(float));
                }
#endif
                return 0;
            }
        }

        if (dims == 3)
        {
            int out_elempack = _outc % 8 == 0 ? 8 : _outc % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (_outw == w && _outh == h && _outc / out_elempack == channels && out_elempack == 8)
            {
                top_blob = bottom_blob;
                return 0;
            }

            if (_coffset % 8 == 0 && out_elempack == 8)
            {
                const Mat bottom_blob_sliced = bottom_blob.channel_range(_coffset / out_elempack, _outc / out_elempack);

                if (_outw == w && _outh == h)
                {
                    top_blob = bottom_blob_sliced.clone(opt.blob_allocator);
                    if (top_blob.empty())
                        return -100;
                }

                top_blob.create(_outw, _outh, _outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < top_blob.c; q++)
                {
                    const Mat m = bottom_blob_sliced.channel(q);
                    Mat borderm = top_blob.channel(q);
#if __riscv_vector
                    crop_packN_rvv(m, borderm, _hoffset, _woffset);
#else
                    for (int y = 0; y < borderm.h; y++)
                    {
                        const float* sptr = m.row(y + _hoffset) + _woffset * elempack;
                        float* dptr = borderm.row(y);
                        memcpy(dptr, sptr, borderm.w * elempack * sizeof(float));
                    }
#endif
                }

                return 0;
            }
        }

        if (dims == 4)
        {
            int out_elempack = _outc % 8 == 0 ? 8 : _outc % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (_outw == w && _outh == h && _outd == d && _outc / out_elempack == channels && out_elempack == 8)
            {
                top_blob = bottom_blob;
                return 0;
            }

            if (_coffset % 8 == 0 && out_elempack == 8)
            {
                const Mat bottom_blob_sliced = bottom_blob.channel_range(_coffset / out_elempack, _outc / out_elempack);

                if (_outw == w && _outh == h && _outd == d)
                {
                    top_blob = bottom_blob_sliced.clone(opt.blob_allocator);
                    if (top_blob.empty())
                        return -100;
                }

                top_blob.create(_outw, _outh, _outd, _outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < top_blob.c; q++)
                {
                    for (int z = 0; z < _outd; z++)
                    {
                        const Mat m = bottom_blob_sliced.channel(q).depth(z + _doffset);
                        Mat borderm = top_blob.channel(q).depth(z);
#if __riscv_vector
                        crop_packN_rvv(m, borderm, _hoffset, _woffset);
#else
                        for (int y = 0; y < borderm.h; y++)
                        {
                            const float* sptr = m.row(y + _hoffset) + _woffset * elempack;
                            float* dptr = borderm.row(y);
                            memcpy(dptr, sptr, borderm.w * elempack * sizeof(float));
                        }
#endif
                    }
                }

                return 0;
            }
        }
    }
#endif // __riscv_vector

    std::vector<Mat> bottom_blobs_unpacked(1);
    Mat bottom_blob_unpacked = bottom_blob;
    if (elempack != 1)
    {
        Option opt_pack1 = opt;
        opt_pack1.blob_allocator = opt.workspace_allocator;

        convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt_pack1);
        if (bottom_blob_unpacked.empty())
            return -100;
    }
    bottom_blobs_unpacked[0] = bottom_blob_unpacked;

    return Crop::forward(bottom_blobs_unpacked[0], top_blob, opt);
}

int Crop_riscv::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& reference_blob = bottom_blobs[1];

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    Mat& top_blob = top_blobs[0];

#if __riscv_vector
    int _woffset, _hoffset, _doffset, _coffset;
    int _outw, _outh, _outd, _outc;
    if (!starts_expr.empty() && !ends_expr.empty())
    {
        std::vector<Mat> bottom_blob_shapes(bottom_blobs.size());
        for (size_t i = 0; i < bottom_blobs.size(); i++)
        {
            bottom_blob_shapes[i] = bottom_blobs[i].shape();
        }
        eval_crop_expr(bottom_blob_shapes, _woffset, _hoffset, _doffset, _coffset, _outw, _outh, _outd, _outc);
    }
    else if (woffset == -233)
    {
        resolve_crop_roi(bottom_blob.shape(), (const int*)reference_blob, _woffset, _hoffset, _doffset, _coffset, _outw, _outh, _outd, _outc);
    }
    else
    {
        resolve_crop_roi(bottom_blob.shape(), reference_blob.shape(), _woffset, _hoffset, _doffset, _coffset, _outw, _outh, _outd, _outc);
    }

    if (elempack == 8)
    {
        if (dims == 1)
        {
            int out_elempack = _outw % 8 == 0 ? 8 : _outw % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (_outw / out_elempack == w && out_elempack == 8)
            {
                top_blob = bottom_blob;
                return 0;
            }

            if (_woffset % 8 == 0 && out_elempack == 8)
            {
                top_blob.create(_outw / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

#if __riscv_vector
                crop_packN_rvv(bottom_blob, top_blob, 0, _woffset / elempack);
#else
                const float* sptr = bottom_blob.row(0) + (_woffset / elempack) * elempack;
                float* dptr = top_blob.row(0);
                for (int i = 0; i < top_blob.w * elempack; i++) dptr[i] = sptr[i];
#endif
                return 0;
            }
        }

        if (dims == 2)
        {
            int out_elempack = _outh % 8 == 0 ? 8 : _outh % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (_outw == w && _outh / out_elempack == h && out_elempack == 8)
            {
                top_blob = bottom_blob;
                return 0;
            }

            if (_hoffset % 8 == 0 && out_elempack == 8)
            {
                top_blob.create(_outw, _outh / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

#if __riscv_vector
                crop_packN_rvv(bottom_blob, top_blob, _hoffset / elempack, _woffset);
#else
                for (int y = 0; y < top_blob.h; y++)
                {
                    const float* sptr = bottom_blob.row(y + _hoffset / elempack) + _woffset * elempack;
                    float* dptr = top_blob.row(y);
                    memcpy(dptr, sptr, top_blob.w * elempack * sizeof(float));
                }
#endif
                return 0;
            }
        }

        if (dims == 3)
        {
            int out_elempack = _outc % 8 == 0 ? 8 : _outc % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (_outw == w && _outh == h && _outc / out_elempack == channels && out_elempack == 8)
            {
                top_blob = bottom_blob;
                return 0;
            }

            if (_coffset % 8 == 0 && out_elempack == 8)
            {
                const Mat bottom_blob_sliced = bottom_blob.channel_range(_coffset / out_elempack, _outc / out_elempack);

                if (_outw == w && _outh == h)
                {
                    top_blob = bottom_blob_sliced.clone(opt.blob_allocator);
                    if (top_blob.empty())
                        return -100;
                }

                top_blob.create(_outw, _outh, _outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < top_blob.c; q++)
                {
                    const Mat m = bottom_blob_sliced.channel(q);
                    Mat borderm = top_blob.channel(q);
#if __riscv_vector
                    crop_packN_rvv(m, borderm, _hoffset, _woffset);
#else
                    for (int y = 0; y < borderm.h; y++)
                    {
                        const float* sptr = m.row(y + _hoffset) + _woffset * elempack;
                        float* dptr = borderm.row(y);
                        memcpy(dptr, sptr, borderm.w * elempack * sizeof(float));
                    }
#endif
                }

                return 0;
            }
        }

        if (dims == 4)
        {
            int out_elempack = _outc % 8 == 0 ? 8 : _outc % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (_outw == w && _outh == h && _outd == d && _outc / out_elempack == channels && out_elempack == 8)
            {
                top_blob = bottom_blob;
                return 0;
            }

            if (_coffset % 8 == 0 && out_elempack == 8)
            {
                const Mat bottom_blob_sliced = bottom_blob.channel_range(_coffset / out_elempack, _outc / out_elempack);

                if (_outw == w && _outh == h && _outd == d)
                {
                    top_blob = bottom_blob_sliced.clone(opt.blob_allocator);
                    if (top_blob.empty())
                        return -100;
                }

                top_blob.create(_outw, _outh, _outd, _outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < top_blob.c; q++)
                {
                    for (int z = 0; z < _outd; z++)
                    {
                        const Mat m = bottom_blob_sliced.channel(q).depth(z + _doffset);
                        Mat borderm = top_blob.channel(q).depth(z);
#if __riscv_vector
                        crop_packN_rvv(m, borderm, _hoffset, _woffset);
#else
                        for (int y = 0; y < borderm.h; y++)
                        {
                            const float* sptr = m.row(y + _hoffset) + _woffset * elempack;
                            float* dptr = borderm.row(y);
                            memcpy(dptr, sptr, borderm.w * elempack * sizeof(float));
                        }
#endif
                    }
                }

                return 0;
            }
        }
    }
#endif // __riscv_vector

    std::vector<Mat> bottom_blobs_unpacked(bottom_blobs.size());
    for (size_t i = 0; i < bottom_blobs.size(); i++)
    {
        Mat bottom_blob_unpacked = bottom_blobs[i];
        if (bottom_blob_unpacked.elempack != 1)
        {
            Option opt_pack1 = opt;
            opt_pack1.blob_allocator = opt.workspace_allocator;

            convert_packing(bottom_blobs[i], bottom_blob_unpacked, 1, opt_pack1);
            if (bottom_blob_unpacked.empty())
                return -100;
        }
        bottom_blobs_unpacked[i] = bottom_blob_unpacked;
    }

    return Crop::forward(bottom_blobs_unpacked, top_blobs, opt);
}

} // namespace ncnn
