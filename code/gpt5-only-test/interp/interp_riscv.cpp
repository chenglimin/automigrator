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

#include "interp_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

#include "riscv_usability.h"

namespace ncnn {

#include "interp_bicubic.h"
#include "interp_bilinear.h"

Interp_riscv::Interp_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector
}

int Interp_riscv::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& reference_blob = bottom_blobs[1];
    Mat& top_blob = top_blobs[0];

    int h = bottom_blob.h;
    int w = bottom_blob.w;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    int outw = reference_blob.w;
    int outh = reference_blob.h;

    if (!size_expr.empty())
    {
        std::vector<Mat> bottom_blob_shapes(bottom_blobs.size());
        for (size_t i = 0; i < bottom_blobs.size(); i++)
        {
            bottom_blob_shapes[i] = bottom_blobs[i].shape();
        }
        eval_size_expr(bottom_blob_shapes, outw, outh);
    }

    if (dims == 1)
    {
        top_blob.create(outw, outh, w, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

#if __riscv_vector
        if (elempack > 1)
        {
            const int packn = cpu_riscv_vlenb() / 4;
            (void)packn;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < w; q++)
            {
                Mat top_blob_c = top_blob.channel(q);
                const float* pv = (const float*)bottom_blob + q * elempack;
                size_t vl = __riscv_vsetvl_e32m1(elempack);
                vfloat32m1_t _v = __riscv_vle32_v_f32m1(pv, vl);
                top_blob_c.fill(_v);
            }
            return 0;
        }
#endif // __riscv_vector

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < w; q++)
        {
            Mat top_blob_c = top_blob.channel(q);
            const float v = bottom_blob[q];
            top_blob_c.fill(v);
        }

        return 0;
    }

    if (dims == 2)
    {
        if (outw == w)
        {
            top_blob = bottom_blob;
            return 0;
        }

        top_blob.create(outw, h, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

#if __riscv_vector
        if (elempack > 1)
        {
            if (resize_type == 1) // nearest
            {
                const float ws = (output_width || !size_expr.empty()) ? w / (float)outw : 1.f / width_scale;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int y = 0; y < h; y++)
                {
                    const float* ptr = bottom_blob.row(y);
                    float* outptr = top_blob.row(y);
                    for (int x = 0; x < outw; x++)
                    {
                        int in_x = std::min((int)(x * ws), (w - 1));
                        const float* Sp = ptr + in_x * elempack;
                        size_t vl = __riscv_vsetvl_e32m1(elempack);
                        vfloat32m1_t _p = __riscv_vle32_v_f32m1(Sp, vl);
                        __riscv_vse32_v_f32m1(outptr, _p, vl);
                        outptr += elempack;
                    }
                }
            }

            if (resize_type == 2) // bilinear
            {
                int* buf = new int[outw + outw * 2];
                int* xofs = buf;
                float* alpha = (float*)(buf + outw);
                linear_coeffs(w, outw, xofs, alpha, align_corner);

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int y = 0; y < h; y++)
                {
                    const float* ptr = bottom_blob.row(y);
                    float* outptr = top_blob.row(y);
                    const float* alphap = alpha;

                    for (int x = 0; x < outw; x++)
                    {
                        int sx = xofs[x] * elempack;
                        const float* Sp = ptr + sx;

                        size_t vl = __riscv_vsetvl_e32m1(elempack);
                        vfloat32m1_t _a0 = __riscv_vfmv_v_f_f32m1(alphap[0], vl);
                        vfloat32m1_t _a1 = __riscv_vfmv_v_f_f32m1(alphap[1], vl);
                        vfloat32m1_t _S0 = __riscv_vle32_v_f32m1(Sp + 0, vl);
                        vfloat32m1_t _S1 = __riscv_vle32_v_f32m1(Sp + elempack, vl);
                        vfloat32m1_t _p = __riscv_vfmul_vv_f32m1(_S0, _a0, vl);
                        _p = __riscv_vfmacc_vv_f32m1(_p, _S1, _a1, vl);
                        __riscv_vse32_v_f32m1(outptr, _p, vl);

                        alphap += 2;
                        outptr += elempack;
                    }
                }

                delete[] buf;
            }

            if (resize_type == 3) // bicubic
            {
                int* buf = new int[outw + outw * 4];
                int* xofs = buf;
                float* alpha = (float*)(buf + outw);
                cubic_coeffs(w, outw, xofs, alpha, align_corner);

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int y = 0; y < h; y++)
                {
                    const float* ptr = bottom_blob.row(y);
                    float* outptr = top_blob.row(y);
                    const float* alphap = alpha;

                    for (int x = 0; x < outw; x++)
                    {
                        int sx = xofs[x] * elempack;
                        const float* Sp = ptr + sx;

                        size_t vl = __riscv_vsetvl_e32m1(elempack);
                        vfloat32m1_t _a0 = __riscv_vfmv_v_f_f32m1(alphap[0], vl);
                        vfloat32m1_t _a1 = __riscv_vfmv_v_f_f32m1(alphap[1], vl);
                        vfloat32m1_t _a2 = __riscv_vfmv_v_f_f32m1(alphap[2], vl);
                        vfloat32m1_t _a3 = __riscv_vfmv_v_f_f32m1(alphap[3], vl);
                        vfloat32m1_t _S0 = __riscv_vle32_v_f32m1(Sp - elempack, vl);
                        vfloat32m1_t _S1 = __riscv_vle32_v_f32m1(Sp + 0, vl);
                        vfloat32m1_t _S2 = __riscv_vle32_v_f32m1(Sp + elempack, vl);
                        vfloat32m1_t _S3 = __riscv_vle32_v_f32m1(Sp + elempack * 2, vl);
                        vfloat32m1_t _p = __riscv_vfmul_vv_f32m1(_S0, _a0, vl);
                        _p = __riscv_vfmacc_vv_f32m1(_p, _S1, _a1, vl);
                        _p = __riscv_vfmacc_vv_f32m1(_p, _S2, _a2, vl);
                        _p = __riscv_vfmacc_vv_f32m1(_p, _S3, _a3, vl);
                        __riscv_vse32_v_f32m1(outptr, _p, vl);

                        alphap += 4;
                        outptr += elempack;
                    }
                }

                delete[] buf;
            }

            return 0;
        }
#endif // __riscv_vector

        if (resize_type == 1) // nearest
        {
            const float ws = (output_width || !size_expr.empty()) ? w / (float)outw : 1.f / width_scale;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int y = 0; y < h; y++)
            {
                const float* ptr = bottom_blob.row(y);
                float* outptr = top_blob.row(y);
                for (int x = 0; x < outw; x++)
                {
                    int in_x = std::min((int)(x * ws), (w - 1));
                    *outptr++ = ptr[in_x];
                }
            }
        }

        if (resize_type == 2) // bilinear
        {
            int* buf = new int[outw + outw * 2];

            int* xofs = buf;
            float* alpha = (float*)(buf + outw);

            linear_coeffs(w, outw, xofs, alpha, align_corner);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int y = 0; y < h; y++)
            {
                const float* ptr = bottom_blob.row(y);
                float* outptr = top_blob.row(y);
                const float* alphap = alpha;

                for (int x = 0; x < outw; x++)
                {
                    int sx = xofs[x];
                    const float* Sp = ptr + sx;
                    float a0 = alphap[0];
                    float a1 = alphap[1];
                    *outptr++ = Sp[0] * a0 + Sp[1] * a1;
                    alphap += 2;
                }
            }

            delete[] buf;
        }

        if (resize_type == 3) // bicubic
        {
            int* buf = new int[outw + outw * 4];

            int* xofs = buf;
            float* alpha = (float*)(buf + outw);

            cubic_coeffs(w, outw, xofs, alpha, align_corner);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int y = 0; y < h; y++)
            {
                const float* ptr = bottom_blob.row(y);
                float* outptr = top_blob.row(y);
                const float* alphap = alpha;

                for (int x = 0; x < outw; x++)
                {
                    int sx = xofs[x];
                    const float* Sp = ptr + sx;
                    float a0 = alphap[0];
                    float a1 = alphap[1];
                    float a2 = alphap[2];
                    float a3 = alphap[3];
                    *outptr++ = Sp[-1] * a0 + Sp[0] * a1 + Sp[1] * a2 + Sp[2] * a3;
                    alphap += 4;
                }
            }

            delete[] buf;
        }

        return 0;
    }

    if (outw == w && outh == h)
    {
        top_blob = bottom_blob;
        return 0;
    }

    top_blob.create(outw, outh, channels, elemsize, elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

#if __riscv_vector
    if (elempack > 1)
    {
        if (resize_type == 1) // nearest
        {
            const float hs = (output_height || !size_expr.empty()) ? h / (float)outh : 1.f / height_scale;
            const float ws = (output_width || !size_expr.empty()) ? w / (float)outw : 1.f / width_scale;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat src = bottom_blob.channel(q);
                Mat dst = top_blob.channel(q);

                for (int y = 0; y < outh; y++)
                {
                    int in_y = std::min((int)(y * hs), (h - 1));

                    const float* ptr = src.row(in_y);
                    float* outptr = dst.row(y);
                    for (int x = 0; x < outw; x++)
                    {
                        int in_x = std::min((int)(x * ws), (w - 1));
                        const float* Sp = ptr + (in_y * w + in_x) * elempack; // ptr already row-indexed, so offset by in_x
                        size_t vl = __riscv_vsetvl_e32m1(elempack);
                        vfloat32m1_t _p = __riscv_vle32_v_f32m1(ptr + in_x * elempack, vl);
                        __riscv_vse32_v_f32m1(outptr, _p, vl);
                        outptr += elempack;
                    }
                }
            }
        }

        if (resize_type == 2) // bilinear
        {
            int* buf = new int[outw + outh + outw * 2 + outh * 2];

            int* xofs = buf;
            int* yofs = buf + outw;

            float* alpha = (float*)(buf + outw + outh);
            float* beta = (float*)(buf + outw + outh + outw * 2);

            linear_coeffs(w, outw, xofs, alpha, align_corner);
            linear_coeffs(h, outh, yofs, beta, align_corner);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat src = bottom_blob.channel(q);
                Mat dst = top_blob.channel(q);

                // loop body with temporary rows0/rows1 as vectors per pack
                Mat rowsbuf0(outw, elemsize, elempack);
                Mat rowsbuf1(outw, elemsize, elempack);
                float* rows0 = rowsbuf0;
                float* rows1 = rowsbuf1;

                int prev_sy1 = -2;

                for (int dy = 0; dy < outh; dy++)
                {
                    int sy = yofs[dy];

                    if (sy == prev_sy1)
                    {
                        // reuse all rows
                    }
                    else if (sy == prev_sy1 + 1)
                    {
                        float* rows0_old = rows0;
                        rows0 = rows1;
                        rows1 = rows0_old;
                        const float* S1 = src.row(sy + 1);

                        const float* alphap = alpha;
                        float* rows1p = rows1;
                        for (int dx = 0; dx < outw; dx++)
                        {
                            int sx = xofs[dx] * elempack;
                            const float* S1p = S1 + sx;

                            size_t vl = __riscv_vsetvl_e32m1(elempack);
                            vfloat32m1_t _a0 = __riscv_vfmv_v_f_f32m1(alphap[0], vl);
                            vfloat32m1_t _a1 = __riscv_vfmv_v_f_f32m1(alphap[1], vl);
                            vfloat32m1_t _S1v0 = __riscv_vle32_v_f32m1(S1p + 0, vl);
                            vfloat32m1_t _S1v1 = __riscv_vle32_v_f32m1(S1p + elempack, vl);
                            vfloat32m1_t _p = __riscv_vfmul_vv_f32m1(_S1v0, _a0, vl);
                            _p = __riscv_vfmacc_vv_f32m1(_p, _S1v1, _a1, vl);
                            __riscv_vse32_v_f32m1(rows1p, _p, vl);

                            alphap += 2;
                            rows1p += elempack;
                        }
                    }
                    else
                    {
                        const float* S0 = src.row(sy);
                        const float* S1 = src.row(sy + 1);

                        const float* alphap = alpha;
                        float* rows0p = rows0;
                        float* rows1p = rows1;
                        for (int dx = 0; dx < outw; dx++)
                        {
                            int sx = xofs[dx] * elempack;
                            const float* S0p = S0 + sx;
                            const float* S1p = S1 + sx;

                            size_t vl = __riscv_vsetvl_e32m1(elempack);
                            vfloat32m1_t _a0 = __riscv_vfmv_v_f_f32m1(alphap[0], vl);
                            vfloat32m1_t _a1 = __riscv_vfmv_v_f_f32m1(alphap[1], vl);
                            vfloat32m1_t _S0v0 = __riscv_vle32_v_f32m1(S0p + 0, vl);
                            vfloat32m1_t _S0v1 = __riscv_vle32_v_f32m1(S0p + elempack, vl);
                            vfloat32m1_t _S1v0 = __riscv_vle32_v_f32m1(S1p + 0, vl);
                            vfloat32m1_t _S1v1 = __riscv_vle32_v_f32m1(S1p + elempack, vl);
                            vfloat32m1_t _p0 = __riscv_vfmul_vv_f32m1(_S0v0, _a0, vl);
                            _p0 = __riscv_vfmacc_vv_f32m1(_p0, _S0v1, _a1, vl);
                            vfloat32m1_t _p1 = __riscv_vfmul_vv_f32m1(_S1v0, _a0, vl);
                            _p1 = __riscv_vfmacc_vv_f32m1(_p1, _S1v1, _a1, vl);
                            __riscv_vse32_v_f32m1(rows0p, _p0, vl);
                            __riscv_vse32_v_f32m1(rows1p, _p1, vl);

                            alphap += 2;
                            rows0p += elempack;
                            rows1p += elempack;
                        }
                    }

                    prev_sy1 = sy;

                    float b0 = beta[0];
                    float b1 = beta[1];

                    float* rows0p = rows0;
                    float* rows1p = rows1;
                    float* Dp = dst.row(dy);
                    for (int dx = 0; dx < outw; dx++)
                    {
                        size_t vl = __riscv_vsetvl_e32m1(elempack);
                        vfloat32m1_t _b0 = __riscv_vfmv_v_f_f32m1(b0, vl);
                        vfloat32m1_t _b1 = __riscv_vfmv_v_f_f32m1(b1, vl);
                        vfloat32m1_t _r0 = __riscv_vle32_v_f32m1(rows0p, vl);
                        vfloat32m1_t _r1 = __riscv_vle32_v_f32m1(rows1p, vl);
                        vfloat32m1_t _p = __riscv_vfmul_vv_f32m1(_r0, _b0, vl);
                        _p = __riscv_vfmacc_vv_f32m1(_p, _r1, _b1, vl);
                        __riscv_vse32_v_f32m1(Dp, _p, vl);
                        Dp += elempack;
                        rows0p += elempack;
                        rows1p += elempack;
                    }

                    beta += 2;
                }
            }

            delete[] buf;
        }

        if (resize_type == 3) // bicubic
        {
            int* buf = new int[outw + outh + outw * 4 + outh * 4];

            int* xofs = buf;
            int* yofs = buf + outw;

            float* alpha = (float*)(buf + outw + outh);
            float* beta = (float*)(buf + outw + outh + outw * 4);

            cubic_coeffs(w, outw, xofs, alpha, align_corner);
            cubic_coeffs(h, outh, yofs, beta, align_corner);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat src = bottom_blob.channel(q);
                Mat dst = top_blob.channel(q);

                // loop body
                Mat rowsbuf0(outw, elemsize, elempack);
                Mat rowsbuf1(outw, elemsize, elempack);
                Mat rowsbuf2(outw, elemsize, elempack);
                Mat rowsbuf3(outw, elemsize, elempack);
                float* rows0 = rowsbuf0;
                float* rows1 = rowsbuf1;
                float* rows2 = rowsbuf2;
                float* rows3 = rowsbuf3;

                int prev_sy1 = -3;

                for (int dy = 0; dy < outh; dy++)
                {
                    int sy = yofs[dy];

                    if (sy == prev_sy1)
                    {
                        // reuse all rows
                    }
                    else if (sy == prev_sy1 + 1)
                    {
                        float* rows0_old = rows0;
                        rows0 = rows1;
                        rows1 = rows2;
                        rows2 = rows3;
                        rows3 = rows0_old;
                        const float* S3 = src.row(sy + 2);

                        const float* alphap = alpha;
                        float* rows3p = rows3;
                        for (int dx = 0; dx < outw; dx++)
                        {
                            int sx = xofs[dx] * elempack;
                            const float* S3p = S3 + sx;

                            size_t vl = __riscv_vsetvl_e32m1(elempack);
                            vfloat32m1_t _a0 = __riscv_vfmv_v_f_f32m1(alphap[0], vl);
                            vfloat32m1_t _a1 = __riscv_vfmv_v_f_f32m1(alphap[1], vl);
                            vfloat32m1_t _a2 = __riscv_vfmv_v_f_f32m1(alphap[2], vl);
                            vfloat32m1_t _a3 = __riscv_vfmv_v_f_f32m1(alphap[3], vl);
                            vfloat32m1_t _S3m1 = __riscv_vle32_v_f32m1(S3p - elempack, vl);
                            vfloat32m1_t _S30 = __riscv_vle32_v_f32m1(S3p + 0, vl);
                            vfloat32m1_t _S31 = __riscv_vle32_v_f32m1(S3p + elempack, vl);
                            vfloat32m1_t _S32 = __riscv_vle32_v_f32m1(S3p + elempack * 2, vl);
                            vfloat32m1_t _p = __riscv_vfmul_vv_f32m1(_S3m1, _a0, vl);
                            _p = __riscv_vfmacc_vv_f32m1(_p, _S30, _a1, vl);
                            _p = __riscv_vfmacc_vv_f32m1(_p, _S31, _a2, vl);
                            _p = __riscv_vfmacc_vv_f32m1(_p, _S32, _a3, vl);
                            __riscv_vse32_v_f32m1(rows3p, _p, vl);

                            alphap += 4;
                            rows3p += elempack;
                        }
                    }
                    else if (sy == prev_sy1 + 2)
                    {
                        float* rows0_old = rows0;
                        float* rows1_old = rows1;
                        rows0 = rows2;
                        rows1 = rows3;
                        rows2 = rows0_old;
                        rows3 = rows1_old;
                        const float* S2 = src.row(sy + 1);
                        const float* S3 = src.row(sy + 2);

                        const float* alphap = alpha;
                        float* rows2p = rows2;
                        float* rows3p = rows3;
                        for (int dx = 0; dx < outw; dx++)
                        {
                            int sx = xofs[dx] * elempack;
                            const float* S2p = S2 + sx;
                            const float* S3p = S3 + sx;

                            size_t vl = __riscv_vsetvl_e32m1(elempack);
                            vfloat32m1_t _a0 = __riscv_vfmv_v_f_f32m1(alphap[0], vl);
                            vfloat32m1_t _a1 = __riscv_vfmv_v_f_f32m1(alphap[1], vl);
                            vfloat32m1_t _a2 = __riscv_vfmv_v_f_f32m1(alphap[2], vl);
                            vfloat32m1_t _a3 = __riscv_vfmv_v_f_f32m1(alphap[3], vl);
                            vfloat32m1_t _S2m1 = __riscv_vle32_v_f32m1(S2p - elempack, vl);
                            vfloat32m1_t _S20 = __riscv_vle32_v_f32m1(S2p + 0, vl);
                            vfloat32m1_t _S21 = __riscv_vle32_v_f32m1(S2p + elempack, vl);
                            vfloat32m1_t _S22 = __riscv_vle32_v_f32m1(S2p + elempack * 2, vl);
                            vfloat32m1_t _p2 = __riscv_vfmul_vv_f32m1(_S2m1, _a0, vl);
                            _p2 = __riscv_vfmacc_vv_f32m1(_p2, _S20, _a1, vl);
                            _p2 = __riscv_vfmacc_vv_f32m1(_p2, _S21, _a2, vl);
                            _p2 = __riscv_vfmacc_vv_f32m1(_p2, _S22, _a3, vl);
                            __riscv_vse32_v_f32m1(rows2p, _p2, vl);

                            vfloat32m1_t _S3m1 = __riscv_vle32_v_f32m1(S3p - elempack, vl);
                            vfloat32m1_t _S30 = __riscv_vle32_v_f32m1(S3p + 0, vl);
                            vfloat32m1_t _S31 = __riscv_vle32_v_f32m1(S3p + elempack, vl);
                            vfloat32m1_t _S32 = __riscv_vle32_v_f32m1(S3p + elempack * 2, vl);
                            vfloat32m1_t _p3 = __riscv_vfmul_vv_f32m1(_S3m1, _a0, vl);
                            _p3 = __riscv_vfmacc_vv_f32m1(_p3, _S30, _a1, vl);
                            _p3 = __riscv_vfmacc_vv_f32m1(_p3, _S31, _a2, vl);
                            _p3 = __riscv_vfmacc_vv_f32m1(_p3, _S32, _a3, vl);
                            __riscv_vse32_v_f32m1(rows3p, _p3, vl);

                            alphap += 4;
                            rows2p += elempack;
                            rows3p += elempack;
                        }
                    }
                    else if (sy == prev_sy1 + 3)
                    {
                        float* rows0_old = rows0;
                        float* rows1_old = rows1;
                        float* rows2_old = rows2;
                        rows0 = rows3;
                        rows1 = rows0_old;
                        rows2 = rows1_old;
                        rows3 = rows2_old;
                        const float* S1 = src.row(sy);
                        const float* S2 = src.row(sy + 1);
                        const float* S3 = src.row(sy + 2);

                        const float* alphap = alpha;
                        float* rows1p = rows1;
                        float* rows2p = rows2;
                        float* rows3p = rows3;
                        for (int dx = 0; dx < outw; dx++)
                        {
                            int sx = xofs[dx] * elempack;
                            const float* S1p = S1 + sx;
                            const float* S2p = S2 + sx;
                            const float* S3p = S3 + sx;

                            size_t vl = __riscv_vsetvl_e32m1(elempack);
                            vfloat32m1_t _a0 = __riscv_vfmv_v_f_f32m1(alphap[0], vl);
                            vfloat32m1_t _a1 = __riscv_vfmv_v_f_f32m1(alphap[1], vl);
                            vfloat32m1_t _a2 = __riscv_vfmv_v_f_f32m1(alphap[2], vl);
                            vfloat32m1_t _a3 = __riscv_vfmv_v_f_f32m1(alphap[3], vl);
                            vfloat32m1_t _S1m1 = __riscv_vle32_v_f32m1(S1p - elempack, vl);
                            vfloat32m1_t _S10 = __riscv_vle32_v_f32m1(S1p + 0, vl);
                            vfloat32m1_t _S11 = __riscv_vle32_v_f32m1(S1p + elempack, vl);
                            vfloat32m1_t _S12 = __riscv_vle32_v_f32m1(S1p + elempack * 2, vl);
                            vfloat32m1_t _p1 = __riscv_vfmul_vv_f32m1(_S1m1, _a0, vl);
                            _p1 = __riscv_vfmacc_vv_f32m1(_p1, _S10, _a1, vl);
                            _p1 = __riscv_vfmacc_vv_f32m1(_p1, _S11, _a2, vl);
                            _p1 = __riscv_vfmacc_vv_f32m1(_p1, _S12, _a3, vl);
                            __riscv_vse32_v_f32m1(rows1p, _p1, vl);

                            vfloat32m1_t _S2m1 = __riscv_vle32_v_f32m1(S2p - elempack, vl);
                            vfloat32m1_t _S20 = __riscv_vle32_v_f32m1(S2p + 0, vl);
                            vfloat32m1_t _S21 = __riscv_vle32_v_f32m1(S2p + elempack, vl);
                            vfloat32m1_t _S22 = __riscv_vle32_v_f32m1(S2p + elempack * 2, vl);
                            vfloat32m1_t _p2 = __riscv_vfmul_vv_f32m1(_S2m1, _a0, vl);
                            _p2 = __riscv_vfmacc_vv_f32m1(_p2, _S20, _a1, vl);
                            _p2 = __riscv_vfmacc_vv_f32m1(_p2, _S21, _a2, vl);
                            _p2 = __riscv_vfmacc_vv_f32m1(_p2, _S22, _a3, vl);
                            __riscv_vse32_v_f32m1(rows2p, _p2, vl);

                            vfloat32m1_t _S3m1 = __riscv_vle32_v_f32m1(S3p - elempack, vl);
                            vfloat32m1_t _S30 = __riscv_vle32_v_f32m1(S3p + 0, vl);
                            vfloat32m1_t _S31 = __riscv_vle32_v_f32m1(S3p + elempack, vl);
                            vfloat32m1_t _S32 = __riscv_vle32_v_f32m1(S3p + elempack * 2, vl);
                            vfloat32m1_t _p3 = __riscv_vfmul_vv_f32m1(_S3m1, _a0, vl);
                            _p3 = __riscv_vfmacc_vv_f32m1(_p3, _S30, _a1, vl);
                            _p3 = __riscv_vfmacc_vv_f32m1(_p3, _S31, _a2, vl);
                            _p3 = __riscv_vfmacc_vv_f32m1(_p3, _S32, _a3, vl);
                            __riscv_vse32_v_f32m1(rows3p, _p3, vl);

                            alphap += 4;
                            rows1p += elempack;
                            rows2p += elempack;
                            rows3p += elempack;
                        }
                    }
                    else
                    {
                        const float* S0 = src.row(sy - 1);
                        const float* S1 = src.row(sy);
                        const float* S2 = src.row(sy + 1);
                        const float* S3 = src.row(sy + 2);

                        const float* alphap = alpha;
                        float* rows0p = rows0;
                        float* rows1p = rows1;
                        float* rows2p = rows2;
                        float* rows3p = rows3;
                        for (int dx = 0; dx < outw; dx++)
                        {
                            int sx = xofs[dx] * elempack;
                            const float* S0p = S0 + sx;
                            const float* S1p = S1 + sx;
                            const float* S2p = S2 + sx;
                            const float* S3p = S3 + sx;

                            size_t vl = __riscv_vsetvl_e32m1(elempack);
                            vfloat32m1_t _a0 = __riscv_vfmv_v_f_f32m1(alphap[0], vl);
                            vfloat32m1_t _a1 = __riscv_vfmv_v_f_f32m1(alphap[1], vl);
                            vfloat32m1_t _a2 = __riscv_vfmv_v_f_f32m1(alphap[2], vl);
                            vfloat32m1_t _a3 = __riscv_vfmv_v_f_f32m1(alphap[3], vl);
                            vfloat32m1_t _S0m1 = __riscv_vle32_v_f32m1(S0p - elempack, vl);
                            vfloat32m1_t _S00 = __riscv_vle32_v_f32m1(S0p + 0, vl);
                            vfloat32m1_t _S01 = __riscv_vle32_v_f32m1(S0p + elempack, vl);
                            vfloat32m1_t _S02 = __riscv_vle32_v_f32m1(S0p + elempack * 2, vl);
                            vfloat32m1_t _r0 = __riscv_vfmul_vv_f32m1(_S0m1, _a0, vl);
                            _r0 = __riscv_vfmacc_vv_f32m1(_r0, _S00, _a1, vl);
                            _r0 = __riscv_vfmacc_vv_f32m1(_r0, _S01, _a2, vl);
                            _r0 = __riscv_vfmacc_vv_f32m1(_r0, _S02, _a3, vl);
                            __riscv_vse32_v_f32m1(rows0p, _r0, vl);

                            vfloat32m1_t _S1m1 = __riscv_vle32_v_f32m1(S1p - elempack, vl);
                            vfloat32m1_t _S10 = __riscv_vle32_v_f32m1(S1p + 0, vl);
                            vfloat32m1_t _S11 = __riscv_vle32_v_f32m1(S1p + elempack, vl);
                            vfloat32m1_t _S12 = __riscv_vle32_v_f32m1(S1p + elempack * 2, vl);
                            vfloat32m1_t _r1 = __riscv_vfmul_vv_f32m1(_S1m1, _a0, vl);
                            _r1 = __riscv_vfmacc_vv_f32m1(_r1, _S10, _a1, vl);
                            _r1 = __riscv_vfmacc_vv_f32m1(_r1, _S11, _a2, vl);
                            _r1 = __riscv_vfmacc_vv_f32m1(_r1, _S12, _a3, vl);
                            __riscv_vse32_v_f32m1(rows1p, _r1, vl);

                            vfloat32m1_t _S2m1 = __riscv_vle32_v_f32m1(S2p - elempack, vl);
                            vfloat32m1_t _S20 = __riscv_vle32_v_f32m1(S2p + 0, vl);
                            vfloat32m1_t _S21 = __riscv_vle32_v_f32m1(S2p + elempack, vl);
                            vfloat32m1_t _S22 = __riscv_vle32_v_f32m1(S2p + elempack * 2, vl);
                            vfloat32m1_t _r2 = __riscv_vfmul_vv_f32m1(_S2m1, _a0, vl);
                            _r2 = __riscv_vfmacc_vv_f32m1(_r2, _S20, _a1, vl);
                            _r2 = __riscv_vfmacc_vv_f32m1(_r2, _S21, _a2, vl);
                            _r2 = __riscv_vfmacc_vv_f32m1(_r2, _S22, _a3, vl);
                            __riscv_vse32_v_f32m1(rows2p, _r2, vl);

                            vfloat32m1_t _S3m1 = __riscv_vle32_v_f32m1(S3p - elempack, vl);
                            vfloat32m1_t _S30 = __riscv_vle32_v_f32m1(S3p + 0, vl);
                            vfloat32m1_t _S31 = __riscv_vle32_v_f32m1(S3p + elempack, vl);
                            vfloat32m1_t _S32 = __riscv_vle32_v_f32m1(S3p + elempack * 2, vl);
                            vfloat32m1_t _r3 = __riscv_vfmul_vv_f32m1(_S3m1, _a0, vl);
                            _r3 = __riscv_vfmacc_vv_f32m1(_r3, _S30, _a1, vl);
                            _r3 = __riscv_vfmacc_vv_f32m1(_r3, _S31, _a2, vl);
                            _r3 = __riscv_vfmacc_vv_f32m1(_r3, _S32, _a3, vl);
                            __riscv_vse32_v_f32m1(rows3p, _r3, vl);

                            alphap += 4;
                            rows0p += elempack;
                            rows1p += elempack;
                            rows2p += elempack;
                            rows3p += elempack;
                        }
                    }

                    prev_sy1 = sy;

                    float b0 = beta[0];
                    float b1 = beta[1];
                    float b2 = beta[2];
                    float b3 = beta[3];

                    float* rows0p = rows0;
                    float* rows1p = rows1;
                    float* rows2p = rows2;
                    float* rows3p = rows3;
                    float* Dp = dst.row(dy);
                    for (int dx = 0; dx < outw; dx++)
                    {
                        size_t vl = __riscv_vsetvl_e32m1(elempack);
                        vfloat32m1_t _b0 = __riscv_vfmv_v_f_f32m1(b0, vl);
                        vfloat32m1_t _b1 = __riscv_vfmv_v_f_f32m1(b1, vl);
                        vfloat32m1_t _b2 = __riscv_vfmv_v_f_f32m1(b2, vl);
                        vfloat32m1_t _b3 = __riscv_vfmv_v_f_f32m1(b3, vl);
                        vfloat32m1_t _r0 = __riscv_vle32_v_f32m1(rows0p, vl);
                        vfloat32m1_t _r1 = __riscv_vle32_v_f32m1(rows1p, vl);
                        vfloat32m1_t _r2 = __riscv_vle32_v_f32m1(rows2p, vl);
                        vfloat32m1_t _r3 = __riscv_vle32_v_f32m1(rows3p, vl);
                        vfloat32m1_t _p = __riscv_vfmul_vv_f32m1(_r0, _b0, vl);
                        _p = __riscv_vfmacc_vv_f32m1(_p, _r1, _b1, vl);
                        _p = __riscv_vfmacc_vv_f32m1(_p, _r2, _b2, vl);
                        _p = __riscv_vfmacc_vv_f32m1(_p, _r3, _b3, vl);
                        __riscv_vse32_v_f32m1(Dp, _p, vl);
                        Dp += elempack;
                        rows0p += elempack;
                        rows1p += elempack;
                        rows2p += elempack;
                        rows3p += elempack;
                    }

                    beta += 4;
                }
            }

            delete[] buf;
        }

        return 0;
    }
#endif // __riscv_vector

    if (resize_type == 1) // nearest
    {
        const float hs = (output_height || !size_expr.empty()) ? h / (float)outh : 1.f / height_scale;
        const float ws = (output_width || !size_expr.empty()) ? w / (float)outw : 1.f / width_scale;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const Mat src = bottom_blob.channel(q);
            Mat dst = top_blob.channel(q);

            for (int y = 0; y < outh; y++)
            {
                int in_y = std::min((int)(y * hs), (h - 1));

                const float* ptr = src.row(in_y);
                float* outptr = dst.row(y);
                for (int x = 0; x < outw; x++)
                {
                    int in_x = std::min((int)(x * ws), (w - 1));
                    *outptr++ = ptr[in_y * w + in_x];
                }
            }
        }
    }

    if (resize_type == 2) // bilinear
    {
        int* buf = new int[outw + outh + outw * 2 + outh * 2];

        int* xofs = buf;        //new int[outw];
        int* yofs = buf + outw; //new int[outh];

        float* alpha = (float*)(buf + outw + outh);           //new float[outw * 2];
        float* beta = (float*)(buf + outw + outh + outw * 2); //new float[outh * 2];

        linear_coeffs(w, outw, xofs, alpha, align_corner);
        linear_coeffs(h, outh, yofs, beta, align_corner);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; ++q)
        {
            const Mat src = bottom_blob.channel(q);
            Mat dst = top_blob.channel(q);

            resize_bilinear_image(src, dst, alpha, xofs, beta, yofs);
        }

        delete[] buf;
    }

    if (resize_type == 3) // bicubic
    {
        int* buf = new int[outw + outh + outw * 4 + outh * 4];

        int* xofs = buf;        //new int[outw];
        int* yofs = buf + outw; //new int[outh];

        float* alpha = (float*)(buf + outw + outh);           //new float[outw * 4];
        float* beta = (float*)(buf + outw + outh + outw * 4); //new float[outh * 4];

        cubic_coeffs(w, outw, xofs, alpha, align_corner);
        cubic_coeffs(h, outh, yofs, beta, align_corner);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const Mat src = bottom_blob.channel(q);
            Mat dst = top_blob.channel(q);

            resize_bicubic_image(src, dst, alpha, xofs, beta, yofs);
        }

        delete[] buf;
    }

    return 0;
}

} // namespace ncnn
