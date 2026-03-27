// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

#include <algorithm>

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

#include "riscv_usability.h"

namespace ncnn {

#include "interp_bicubic.h"
#include "interp_bilinear.h"

#if __riscv_vector
static void resize_bilinear_image_packn(const Mat& src, Mat& dst, float* alpha, int* xofs, float* beta, int* yofs)
{
    int w = dst.w;
    int h = dst.h;

    const int packn = csrr_vlenb() / 4;
    size_t vl = __riscv_vsetvl_e32m1(packn);

    Mat rowsbuf0(w, (size_t)packn * 4u, packn);
    Mat rowsbuf1(w, (size_t)packn * 4u, packn);
    float* rows0 = rowsbuf0;
    float* rows1 = rowsbuf1;

    int prev_sy1 = -2;

    for (int dy = 0; dy < h; dy++)
    {
        int sy = yofs[dy];

        if (sy == prev_sy1)
        {
        }
        else if (sy == prev_sy1 + 1)
        {
            float* rows0_old = rows0;
            rows0 = rows1;
            rows1 = rows0_old;
            const float* S1 = src.row(sy + 1);

            const float* alphap = alpha;
            float* rows1p = rows1;
            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx] * packn;
                const float* S1p = S1 + sx;

                vfloat32m1_t _a0 = __riscv_vfmv_v_f_f32m1(alphap[0], vl);
                vfloat32m1_t _a1 = __riscv_vfmv_v_f_f32m1(alphap[1], vl);

                vfloat32m1_t _S10 = __riscv_vle32_v_f32m1(S1p, vl);
                vfloat32m1_t _S11 = __riscv_vle32_v_f32m1(S1p + packn, vl);
                vfloat32m1_t _rows1 = __riscv_vfmul_vv_f32m1(_S10, _a0, vl);
                _rows1 = __riscv_vfmadd_vv_f32m1(_S11, _a1, _rows1, vl);
                __riscv_vse32_v_f32m1(rows1p + dx * packn, _rows1, vl);

                alphap += 2;
            }
        }
        else
        {
            const float* S0 = src.row(sy);
            const float* S1 = src.row(sy + 1);

            const float* alphap = alpha;
            float* rows0p = rows0;
            float* rows1p = rows1;
            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx] * packn;
                const float* S0p = S0 + sx;
                const float* S1p = S1 + sx;

                vfloat32m1_t _a0 = __riscv_vfmv_v_f_f32m1(alphap[0], vl);
                vfloat32m1_t _a1 = __riscv_vfmv_v_f_f32m1(alphap[1], vl);

                vfloat32m1_t _S00 = __riscv_vle32_v_f32m1(S0p, vl);
                vfloat32m1_t _S01 = __riscv_vle32_v_f32m1(S0p + packn, vl);
                vfloat32m1_t _S10 = __riscv_vle32_v_f32m1(S1p, vl);
                vfloat32m1_t _S11 = __riscv_vle32_v_f32m1(S1p + packn, vl);
                vfloat32m1_t _rows0 = __riscv_vfmul_vv_f32m1(_S00, _a0, vl);
                vfloat32m1_t _rows1 = __riscv_vfmul_vv_f32m1(_S10, _a0, vl);
                _rows0 = __riscv_vfmadd_vv_f32m1(_S01, _a1, _rows0, vl);
                _rows1 = __riscv_vfmadd_vv_f32m1(_S11, _a1, _rows1, vl);
                __riscv_vse32_v_f32m1(rows0p + dx * packn, _rows0, vl);
                __riscv_vse32_v_f32m1(rows1p + dx * packn, _rows1, vl);

                alphap += 2;
            }
        }

        prev_sy1 = sy;

        vfloat32m1_t _b0 = __riscv_vfmv_v_f_f32m1(beta[0], vl);
        vfloat32m1_t _b1 = __riscv_vfmv_v_f_f32m1(beta[1], vl);

        float* rows0p = rows0;
        float* rows1p = rows1;
        float* Dp = dst.row(dy);

        for (int dx = 0; dx < w; dx++)
        {
            vfloat32m1_t _rows0 = __riscv_vle32_v_f32m1(rows0p, vl);
            vfloat32m1_t _rows1 = __riscv_vle32_v_f32m1(rows1p, vl);
            vfloat32m1_t _Dp = __riscv_vfmul_vv_f32m1(_rows0, _b0, vl);
            _Dp = __riscv_vfmadd_vv_f32m1(_rows1, _b1, _Dp, vl);
            __riscv_vse32_v_f32m1(Dp, _Dp, vl);

            Dp += packn;
            rows0p += packn;
            rows1p += packn;
        }

        beta += 2;
    }
}

static void resize_bicubic_image_packn(const Mat& src, Mat& dst, float* alpha, int* xofs, float* beta, int* yofs)
{
    int w = dst.w;
    int h = dst.h;

    const int packn = csrr_vlenb() / 4;
    size_t vl = __riscv_vsetvl_e32m1(packn);

    Mat rowsbuf0(w, (size_t)packn * 4u, packn);
    Mat rowsbuf1(w, (size_t)packn * 4u, packn);
    Mat rowsbuf2(w, (size_t)packn * 4u, packn);
    Mat rowsbuf3(w, (size_t)packn * 4u, packn);
    float* rows0 = rowsbuf0;
    float* rows1 = rowsbuf1;
    float* rows2 = rowsbuf2;
    float* rows3 = rowsbuf3;

    int prev_sy1 = -3;

    for (int dy = 0; dy < h; dy++)
    {
        int sy = yofs[dy];

        if (sy == prev_sy1)
        {
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
            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx] * packn;
                const float* S3p = S3 + sx;

                vfloat32m1_t _a0 = __riscv_vfmv_v_f_f32m1(alphap[0], vl);
                vfloat32m1_t _a1 = __riscv_vfmv_v_f_f32m1(alphap[1], vl);
                vfloat32m1_t _a2 = __riscv_vfmv_v_f_f32m1(alphap[2], vl);
                vfloat32m1_t _a3 = __riscv_vfmv_v_f_f32m1(alphap[3], vl);

                vfloat32m1_t _S30 = __riscv_vle32_v_f32m1(S3p - packn, vl);
                vfloat32m1_t _S31 = __riscv_vle32_v_f32m1(S3p + 0, vl);
                vfloat32m1_t _S32 = __riscv_vle32_v_f32m1(S3p + packn, vl);
                vfloat32m1_t _S33 = __riscv_vle32_v_f32m1(S3p + packn * 2, vl);
                vfloat32m1_t _rows3 = __riscv_vfmul_vv_f32m1(_S30, _a0, vl);
                _rows3 = __riscv_vfmadd_vv_f32m1(_S31, _a1, _rows3, vl);
                _rows3 = __riscv_vfmadd_vv_f32m1(_S32, _a2, _rows3, vl);
                _rows3 = __riscv_vfmadd_vv_f32m1(_S33, _a3, _rows3, vl);
                __riscv_vse32_v_f32m1(rows3p + dx * packn, _rows3, vl);

                alphap += 4;
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
            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx] * packn;
                const float* S2p = S2 + sx;
                const float* S3p = S3 + sx;

                vfloat32m1_t _a0 = __riscv_vfmv_v_f_f32m1(alphap[0], vl);
                vfloat32m1_t _a1 = __riscv_vfmv_v_f_f32m1(alphap[1], vl);
                vfloat32m1_t _a2 = __riscv_vfmv_v_f_f32m1(alphap[2], vl);
                vfloat32m1_t _a3 = __riscv_vfmv_v_f_f32m1(alphap[3], vl);

                vfloat32m1_t _S20 = __riscv_vle32_v_f32m1(S2p - packn, vl);
                vfloat32m1_t _S21 = __riscv_vle32_v_f32m1(S2p + 0, vl);
                vfloat32m1_t _S22 = __riscv_vle32_v_f32m1(S2p + packn, vl);
                vfloat32m1_t _S23 = __riscv_vle32_v_f32m1(S2p + packn * 2, vl);
                vfloat32m1_t _S30 = __riscv_vle32_v_f32m1(S3p - packn, vl);
                vfloat32m1_t _S31 = __riscv_vle32_v_f32m1(S3p + 0, vl);
                vfloat32m1_t _S32 = __riscv_vle32_v_f32m1(S3p + packn, vl);
                vfloat32m1_t _S33 = __riscv_vle32_v_f32m1(S3p + packn * 2, vl);
                vfloat32m1_t _rows2 = __riscv_vfmul_vv_f32m1(_S20, _a0, vl);
                vfloat32m1_t _rows3 = __riscv_vfmul_vv_f32m1(_S30, _a0, vl);
                _rows2 = __riscv_vfmadd_vv_f32m1(_S21, _a1, _rows2, vl);
                _rows3 = __riscv_vfmadd_vv_f32m1(_S31, _a1, _rows3, vl);
                _rows2 = __riscv_vfmadd_vv_f32m1(_S22, _a2, _rows2, vl);
                _rows3 = __riscv_vfmadd_vv_f32m1(_S32, _a2, _rows3, vl);
                _rows2 = __riscv_vfmadd_vv_f32m1(_S23, _a3, _rows2, vl);
                _rows3 = __riscv_vfmadd_vv_f32m1(_S33, _a3, _rows3, vl);
                __riscv_vse32_v_f32m1(rows2p + dx * packn, _rows2, vl);
                __riscv_vse32_v_f32m1(rows3p + dx * packn, _rows3, vl);

                alphap += 4;
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
            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx] * packn;
                const float* S1p = S1 + sx;
                const float* S2p = S2 + sx;
                const float* S3p = S3 + sx;

                vfloat32m1_t _a0 = __riscv_vfmv_v_f_f32m1(alphap[0], vl);
                vfloat32m1_t _a1 = __riscv_vfmv_v_f_f32m1(alphap[1], vl);
                vfloat32m1_t _a2 = __riscv_vfmv_v_f_f32m1(alphap[2], vl);
                vfloat32m1_t _a3 = __riscv_vfmv_v_f_f32m1(alphap[3], vl);

                vfloat32m1_t _S10 = __riscv_vle32_v_f32m1(S1p - packn, vl);
                vfloat32m1_t _S11 = __riscv_vle32_v_f32m1(S1p + 0, vl);
                vfloat32m1_t _S12 = __riscv_vle32_v_f32m1(S1p + packn, vl);
                vfloat32m1_t _S13 = __riscv_vle32_v_f32m1(S1p + packn * 2, vl);
                vfloat32m1_t _S20 = __riscv_vle32_v_f32m1(S2p - packn, vl);
                vfloat32m1_t _S21 = __riscv_vle32_v_f32m1(S2p + 0, vl);
                vfloat32m1_t _S22 = __riscv_vle32_v_f32m1(S2p + packn, vl);
                vfloat32m1_t _S23 = __riscv_vle32_v_f32m1(S2p + packn * 2, vl);
                vfloat32m1_t _S30 = __riscv_vle32_v_f32m1(S3p - packn, vl);
                vfloat32m1_t _S31 = __riscv_vle32_v_f32m1(S3p + 0, vl);
                vfloat32m1_t _S32 = __riscv_vle32_v_f32m1(S3p + packn, vl);
                vfloat32m1_t _S33 = __riscv_vle32_v_f32m1(S3p + packn * 2, vl);
                vfloat32m1_t _rows1 = __riscv_vfmul_vv_f32m1(_S10, _a0, vl);
                vfloat32m1_t _rows2 = __riscv_vfmul_vv_f32m1(_S20, _a0, vl);
                vfloat32m1_t _rows3 = __riscv_vfmul_vv_f32m1(_S30, _a0, vl);
                _rows1 = __riscv_vfmadd_vv_f32m1(_S11, _a1, _rows1, vl);
                _rows2 = __riscv_vfmadd_vv_f32m1(_S21, _a1, _rows2, vl);
                _rows3 = __riscv_vfmadd_vv_f32m1(_S31, _a1, _rows3, vl);
                _rows1 = __riscv_vfmadd_vv_f32m1(_S12, _a2, _rows1, vl);
                _rows2 = __riscv_vfmadd_vv_f32m1(_S22, _a2, _rows2, vl);
                _rows3 = __riscv_vfmadd_vv_f32m1(_S32, _a2, _rows3, vl);
                _rows1 = __riscv_vfmadd_vv_f32m1(_S13, _a3, _rows1, vl);
                _rows2 = __riscv_vfmadd_vv_f32m1(_S23, _a3, _rows2, vl);
                _rows3 = __riscv_vfmadd_vv_f32m1(_S33, _a3, _rows3, vl);
                __riscv_vse32_v_f32m1(rows1p + dx * packn, _rows1, vl);
                __riscv_vse32_v_f32m1(rows2p + dx * packn, _rows2, vl);
                __riscv_vse32_v_f32m1(rows3p + dx * packn, _rows3, vl);

                alphap += 4;
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
            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx] * packn;
                const float* S0p = S0 + sx;
                const float* S1p = S1 + sx;
                const float* S2p = S2 + sx;
                const float* S3p = S3 + sx;

                vfloat32m1_t _a0 = __riscv_vfmv_v_f_f32m1(alphap[0], vl);
                vfloat32m1_t _a1 = __riscv_vfmv_v_f_f32m1(alphap[1], vl);
                vfloat32m1_t _a2 = __riscv_vfmv_v_f_f32m1(alphap[2], vl);
                vfloat32m1_t _a3 = __riscv_vfmv_v_f_f32m1(alphap[3], vl);

                vfloat32m1_t _S00 = __riscv_vle32_v_f32m1(S0p - packn, vl);
                vfloat32m1_t _S01 = __riscv_vle32_v_f32m1(S0p + 0, vl);
                vfloat32m1_t _S02 = __riscv_vle32_v_f32m1(S0p + packn, vl);
                vfloat32m1_t _S03 = __riscv_vle32_v_f32m1(S0p + packn * 2, vl);
                vfloat32m1_t _S10 = __riscv_vle32_v_f32m1(S1p - packn, vl);
                vfloat32m1_t _S11 = __riscv_vle32_v_f32m1(S1p + 0, vl);
                vfloat32m1_t _S12 = __riscv_vle32_v_f32m1(S1p + packn, vl);
                vfloat32m1_t _S13 = __riscv_vle32_v_f32m1(S1p + packn * 2, vl);
                vfloat32m1_t _S20 = __riscv_vle32_v_f32m1(S2p - packn, vl);
                vfloat32m1_t _S21 = __riscv_vle32_v_f32m1(S2p + 0, vl);
                vfloat32m1_t _S22 = __riscv_vle32_v_f32m1(S2p + packn, vl);
                vfloat32m1_t _S23 = __riscv_vle32_v_f32m1(S2p + packn * 2, vl);
                vfloat32m1_t _S30 = __riscv_vle32_v_f32m1(S3p - packn, vl);
                vfloat32m1_t _S31 = __riscv_vle32_v_f32m1(S3p + 0, vl);
                vfloat32m1_t _S32 = __riscv_vle32_v_f32m1(S3p + packn, vl);
                vfloat32m1_t _S33 = __riscv_vle32_v_f32m1(S3p + packn * 2, vl);
                vfloat32m1_t _rows0 = __riscv_vfmul_vv_f32m1(_S00, _a0, vl);
                vfloat32m1_t _rows1 = __riscv_vfmul_vv_f32m1(_S10, _a0, vl);
                vfloat32m1_t _rows2 = __riscv_vfmul_vv_f32m1(_S20, _a0, vl);
                vfloat32m1_t _rows3 = __riscv_vfmul_vv_f32m1(_S30, _a0, vl);
                _rows0 = __riscv_vfmadd_vv_f32m1(_S01, _a1, _rows0, vl);
                _rows1 = __riscv_vfmadd_vv_f32m1(_S11, _a1, _rows1, vl);
                _rows2 = __riscv_vfmadd_vv_f32m1(_S21, _a1, _rows2, vl);
                _rows3 = __riscv_vfmadd_vv_f32m1(_S31, _a1, _rows3, vl);
                _rows0 = __riscv_vfmadd_vv_f32m1(_S02, _a2, _rows0, vl);
                _rows1 = __riscv_vfmadd_vv_f32m1(_S12, _a2, _rows1, vl);
                _rows2 = __riscv_vfmadd_vv_f32m1(_S22, _a2, _rows2, vl);
                _rows3 = __riscv_vfmadd_vv_f32m1(_S32, _a2, _rows3, vl);
                _rows0 = __riscv_vfmadd_vv_f32m1(_S03, _a3, _rows0, vl);
                _rows1 = __riscv_vfmadd_vv_f32m1(_S13, _a3, _rows1, vl);
                _rows2 = __riscv_vfmadd_vv_f32m1(_S23, _a3, _rows2, vl);
                _rows3 = __riscv_vfmadd_vv_f32m1(_S33, _a3, _rows3, vl);
                __riscv_vse32_v_f32m1(rows0p + dx * packn, _rows0, vl);
                __riscv_vse32_v_f32m1(rows1p + dx * packn, _rows1, vl);
                __riscv_vse32_v_f32m1(rows2p + dx * packn, _rows2, vl);
                __riscv_vse32_v_f32m1(rows3p + dx * packn, _rows3, vl);

                alphap += 4;
            }
        }

        prev_sy1 = sy;

        vfloat32m1_t _b0 = __riscv_vfmv_v_f_f32m1(beta[0], vl);
        vfloat32m1_t _b1 = __riscv_vfmv_v_f_f32m1(beta[1], vl);
        vfloat32m1_t _b2 = __riscv_vfmv_v_f_f32m1(beta[2], vl);
        vfloat32m1_t _b3 = __riscv_vfmv_v_f_f32m1(beta[3], vl);

        float* rows0p = rows0;
        float* rows1p = rows1;
        float* rows2p = rows2;
        float* rows3p = rows3;
        float* Dp = dst.row(dy);

        for (int dx = 0; dx < w; dx++)
        {
            vfloat32m1_t _rows0 = __riscv_vle32_v_f32m1(rows0p, vl);
            vfloat32m1_t _rows1 = __riscv_vle32_v_f32m1(rows1p, vl);
            vfloat32m1_t _rows2 = __riscv_vle32_v_f32m1(rows2p, vl);
            vfloat32m1_t _rows3 = __riscv_vle32_v_f32m1(rows3p, vl);
            vfloat32m1_t _Dp = __riscv_vfmul_vv_f32m1(_rows0, _b0, vl);
            _Dp = __riscv_vfmadd_vv_f32m1(_rows1, _b1, _Dp, vl);
            _Dp = __riscv_vfmadd_vv_f32m1(_rows2, _b2, _Dp, vl);
            _Dp = __riscv_vfmadd_vv_f32m1(_rows3, _b3, _Dp, vl);
            __riscv_vse32_v_f32m1(Dp, _Dp, vl);

            Dp += packn;
            rows0p += packn;
            rows1p += packn;
            rows2p += packn;
            rows3p += packn;
        }

        beta += 4;
    }
}
#endif // __riscv_vector

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
        const int packn = csrr_vlenb() / 4;
        if (elempack == packn)
        {
            size_t vl = __riscv_vsetvl_e32m1(packn);
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < w; q++)
            {
                Mat top_blob_c = top_blob.channel(q);
                const float* srcp = (const float*)bottom_blob + q * packn;
                for (int yy = 0; yy < outh; yy++)
                {
                    float* outptr = top_blob_c.row(yy);
                    for (int xx = 0; xx < outw; xx++)
                    {
                        vfloat32m1_t _v = __riscv_vle32_v_f32m1(srcp, vl);
                        __riscv_vse32_v_f32m1(outptr, _v, vl);
                        outptr += packn;
                    }
                }
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
        const int packn = csrr_vlenb() / 4;
        if (elempack == packn)
        {
            if (resize_type == 1)
            {
                const float ws = (output_width || !size_expr.empty()) ? w / (float)outw : 1.f / width_scale;

                size_t vl = __riscv_vsetvl_e32m1(packn);
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int y = 0; y < h; y++)
                {
                    const float* ptr = bottom_blob.row(y);
                    float* outptr = top_blob.row(y);
                    for (int x = 0; x < outw; x++)
                    {
                        int in_x = std::min((int)(x * ws), (w - 1));
                        const float* Sp = ptr + in_x * packn;
                        vfloat32m1_t _p = __riscv_vle32_v_f32m1(Sp, vl);
                        __riscv_vse32_v_f32m1(outptr, _p, vl);
                        outptr += packn;
                    }
                }
            }

            if (resize_type == 2)
            {
                int* buf = new int[outw + outw * 2];
                int* xofs = buf;
                float* alpha = (float*)(buf + outw);
                linear_coeffs(w, outw, xofs, alpha, align_corner);

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int y = 0; y < h; y++)
                {
                    const float* S0 = bottom_blob.row(y);
                    const float* S1 = bottom_blob.row(y);
                    float* outptr = top_blob.row(y);
                    const float* alphap = alpha;
                    for (int x = 0; x < outw; x++)
                    {
                        int sx = xofs[x] * packn;
                        const float* S0p = S0 + sx;
                        size_t vl = __riscv_vsetvl_e32m1(packn);
                        vfloat32m1_t _a0 = __riscv_vfmv_v_f_f32m1(alphap[0], vl);
                        vfloat32m1_t _a1 = __riscv_vfmv_v_f_f32m1(alphap[1], vl);
                        vfloat32m1_t _S00 = __riscv_vle32_v_f32m1(S0p, vl);
                        vfloat32m1_t _S01 = __riscv_vle32_v_f32m1(S0p + packn, vl);
                        vfloat32m1_t _rows0 = __riscv_vfmul_vv_f32m1(_S00, _a0, vl);
                        _rows0 = __riscv_vfmadd_vv_f32m1(_S01, _a1, _rows0, vl);
                        __riscv_vse32_v_f32m1(outptr, _rows0, vl);
                        outptr += packn;
                        alphap += 2;
                    }
                }

                delete[] buf;
            }

            if (resize_type == 3)
            {
                int* buf = new int[outw + outw * 4];
                int* xofs = buf;
                float* alpha = (float*)(buf + outw);
                cubic_coeffs(w, outw, xofs, alpha, align_corner);

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int y = 0; y < h; y++)
                {
                    const float* S0 = bottom_blob.row(y);
                    float* outptr = top_blob.row(y);
                    const float* alphap = alpha;
                    for (int x = 0; x < outw; x++)
                    {
                        int sx = xofs[x] * packn;
                        const float* Sp = S0 + sx;
                        size_t vl = __riscv_vsetvl_e32m1(packn);
                        vfloat32m1_t _a0 = __riscv_vfmv_v_f_f32m1(alphap[0], vl);
                        vfloat32m1_t _a1 = __riscv_vfmv_v_f_f32m1(alphap[1], vl);
                        vfloat32m1_t _a2 = __riscv_vfmv_v_f_f32m1(alphap[2], vl);
                        vfloat32m1_t _a3 = __riscv_vfmv_v_f_f32m1(alphap[3], vl);
                        vfloat32m1_t _S0 = __riscv_vle32_v_f32m1(Sp - packn, vl);
                        vfloat32m1_t _S1 = __riscv_vle32_v_f32m1(Sp + 0, vl);
                        vfloat32m1_t _S2 = __riscv_vle32_v_f32m1(Sp + packn, vl);
                        vfloat32m1_t _S3 = __riscv_vle32_v_f32m1(Sp + packn * 2, vl);
                        vfloat32m1_t _p = __riscv_vfmul_vv_f32m1(_S0, _a0, vl);
                        _p = __riscv_vfmadd_vv_f32m1(_S1, _a1, _p, vl);
                        _p = __riscv_vfmadd_vv_f32m1(_S2, _a2, _p, vl);
                        _p = __riscv_vfmadd_vv_f32m1(_S3, _a3, _p, vl);
                        __riscv_vse32_v_f32m1(outptr, _p, vl);
                        outptr += packn;
                        alphap += 4;
                    }
                }

                delete[] buf;
            }

            return 0;
        }
#endif // __riscv_vector

        if (resize_type == 1)
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

        if (resize_type == 2)
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

        if (resize_type == 3)
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
    const int packn = csrr_vlenb() / 4;
    if (elempack == packn)
    {
        if (resize_type == 1)
        {
            const float hs = (output_height || !size_expr.empty()) ? h / (float)outh : 1.f / height_scale;
            const float ws = (output_width || !size_expr.empty()) ? w / (float)outw : 1.f / width_scale;

            size_t vl = __riscv_vsetvl_e32m1(packn);
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
                        const float* Sp = ptr + in_x * packn;
                        vfloat32m1_t _p = __riscv_vle32_v_f32m1(Sp, vl);
                        __riscv_vse32_v_f32m1(outptr, _p, vl);
                        outptr += packn;
                    }
                }
            }
        }

        if (resize_type == 2)
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
                resize_bilinear_image_packn(src, dst, alpha, xofs, beta, yofs);
            }

            delete[] buf;
        }

        if (resize_type == 3)
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
                resize_bicubic_image_packn(src, dst, alpha, xofs, beta, yofs);
            }

            delete[] buf;
        }

        return 0;
    }
#endif // __riscv_vector

    if (resize_type == 1)
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
                    *outptr++ = ptr[in_x];
                }
            }
        }
    }

    if (resize_type == 2)
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
            resize_bilinear_image(src, dst, alpha, xofs, beta, yofs);
        }

        delete[] buf;
    }

    if (resize_type == 3)
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
            resize_bicubic_image(src, dst, alpha, xofs, beta, yofs);
        }

        delete[] buf;
    }

    return 0;
}

} // namespace ncnn
