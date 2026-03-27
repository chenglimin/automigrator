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

#include "gemm_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#include "riscv_usability.h"
#endif // __riscv_vector

#include "cpu.h"

namespace ncnn {

Gemm_riscv::Gemm_riscv()
{
    // defer packing support to base implementation for correctness in tests
    support_packing = false;
}

// Pack A tile from row-major A into contiguous AT by rows of ii and columns of kk
static void pack_A_tile_rvv(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    const int elempack = A.elempack;
    const int A_hstep = A.dims == 3 ? (int)A.cstep : A.w;
    float* pp = AT;

    int ii = 0;
#if __riscv_vector
    const int packn = csrr_vlenb() / 4; // dynamic packn
    // Use m1 segments for transpose helpers but load/store with m8 when possible
    for (; ii + (packn - 1) < max_ii; ii += packn)
    {
        if (elempack == packn)
        {
            const float* p0 = (const float*)A + (i + ii) * A_hstep + k * packn;
            for (int kk = 0; kk < max_kk; kk++)
            {
                size_t vl = __riscv_vsetvl_e32m8(packn);
                vfloat32m8_t _v = __riscv_vle32_v_f32m8(p0, vl);
                __riscv_vse32_v_f32m8(pp, _v, vl);
                pp += packn;
                p0 += packn;
            }
            continue;
        }
        if (elempack == 1)
        {
            // load packn rows and transpose into contiguous columns
            const float* p[16]; // support up to packn<=16
            for (int r = 0; r < packn; r++) p[r] = (const float*)A + (i + ii + r) * A_hstep + k;

            int kk = 0;
            for (; kk + (packn - 1) < max_kk; kk += packn)
            {
                // load packn elements from packn rows
                // transpose packn x packn using vsseg
                size_t vl = __riscv_vsetvl_e32m1(packn);
                // store by segment to temporary buffer then reload per column
                float tmp[16][16];
                for (int r = 0; r < packn; r++) {
                    vfloat32m1_t row = __riscv_vle32_v_f32m1(p[r], vl);
                    __riscv_vsse32_v_f32m1(&tmp[r][0], sizeof(float) * packn, row, vl);
                }
                for (int c = 0; c < packn; c++)
                {
                    size_t vlc = __riscv_vsetvl_e32m1(packn);
                    vfloat32m1_t col = __riscv_vle32_v_f32m1(&tmp[c][0], vlc);
                    // widen write contiguously
                    // write packn elements for this column
                    float buf[16];
                    __riscv_vse32_v_f32m1(buf, col, vlc);
                    for (int t = 0; t < packn; t++) pp[t] = buf[t];
                    pp += packn;
                }
                for (int r = 0; r < packn; r++) p[r] += packn;
            }
            for (; kk < max_kk; kk++)
            {
                for (int r = 0; r < packn; r++)
                {
                    pp[r] = p[r][0];
                    p[r]++;
                }
                pp += packn;
            }
            continue;
        }
    }
    // handle remaining rows in scalar
#endif // __riscv_vector
    for (; ii < max_ii; ii++)
    {
        if (elempack == 1)
        {
            const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
            for (int kk = 0; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp += 1;
                p0 += 1;
            }
        }
        else
        {
            const float* p0 = (const float*)A + (i + ii) * A_hstep + k * elempack;
            for (int kk = 0; kk < max_kk; kk++)
            {
                for (int e = 0; e < elempack; e++) pp[e] = p0[e];
                pp += elempack;
                p0 += elempack;
            }
        }
    }
}

// Compute top = A * BT (BT is transposed B) with optional bias C
static void gemm_transB_rvv(const Mat& A, const Mat& BT, const Mat& C, Mat& top_blob, float alpha, float beta, int broadcast_type_C, int output_transpose, const Option& opt)
{
    const int M = A.dims == 3 ? A.c : A.h;
    const int N = BT.dims == 3 ? BT.c : BT.h;
    const int K = A.w;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int i = 0; i < M; i++)
    {
        const int out_hstep = top_blob.dims == 3 ? (int)top_blob.cstep : top_blob.w;
        const int A_hstep = A.dims == 3 ? (int)A.cstep : A.w;
        const int BT_hstep = BT.dims == 3 ? (int)BT.cstep : BT.w;

        const float* ptrA = (const float*)A + i * A_hstep;
        const float* ptrC = C;

        for (int j = 0; j < N; j++)
        {
            const float* ptrBT = (const float*)BT + j * BT_hstep;

            float sum = 0.f;
            if (ptrC)
            {
                if (broadcast_type_C == 0) sum = ptrC[0];
                if (broadcast_type_C == 1) sum = ptrC[i];
                if (broadcast_type_C == 2) sum = ptrC[i];
                if (broadcast_type_C == 3) sum = ptrC[i * N + j];
                if (broadcast_type_C == 4) sum = ptrC[j];
                sum *= beta;
            }

#if __riscv_vector
            int k = 0;
            vfloat32m1_t acc_v = __riscv_vfmv_v_f_f32m1(0.f, 1);
            for (; k < K; )
            {
                int n = K - k;
                size_t vl = __riscv_vsetvl_e32m8(n);
                vfloat32m8_t _a = __riscv_vle32_v_f32m8(ptrA + k, vl);
                vfloat32m8_t _b = __riscv_vle32_v_f32m8(ptrBT + k, vl);
                vfloat32m8_t _mul = __riscv_vfmul_vv_f32m8(_a, _b, vl);
                // reduce sum
                vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.f, vl);
                acc_v = __riscv_vfredusum_vs_f32m8_f32m1(_mul, zero, vl);
                float part = __riscv_vfmv_f_s_f32m1_f32(acc_v);
                sum += part;
                k += vl;
            }
#else
            for (int k2 = 0; k2 < K; k2++) sum += ptrA[k2] * ptrBT[k2];
#endif
            sum *= alpha;
            if (output_transpose) top_blob[j * out_hstep + i] = sum;
            else top_blob[i * out_hstep + j] = sum;
        }
    }
}

int Gemm_riscv::create_pipeline(const Option& opt)
{
    // nothing to precreate now; keep consistent with base Gemm
    return 0;
}

int Gemm_riscv::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    // Accept packed inputs but fallback to scalar base by unpacking then repacking per packn_spec
    std::vector<Mat> unpacked_bottom_blobs(bottom_blobs.size());
    for (size_t bi = 0; bi < bottom_blobs.size(); bi++)
    {
        if (bottom_blobs[bi].elempack != 1)
        {
            convert_packing(bottom_blobs[bi], unpacked_bottom_blobs[bi], 1, opt);
        }
        else
        {
            unpacked_bottom_blobs[bi] = bottom_blobs[bi];
        }
    }

    int ret = Gemm::forward(unpacked_bottom_blobs, top_blobs, opt);
    return ret;

    // RVV path to be enabled after validation
    const Mat& A0 = constantA ? A_data : bottom_blobs[0];
    const Mat& B0 = constantB ? B_data : constantA ? bottom_blobs[0] : bottom_blobs[1];

    size_t elemsize = A0.elemsize;

    Mat A;
    if (transA == 0) A = A0;
    else
    {
        A.create((A0.dims == 3 ? A0.c : A0.h), A0.w, elemsize, opt.workspace_allocator);
        const int A0_hstep = A0.dims == 3 ? (int)A0.cstep : A0.w;
        for (int i = 0; i < A.h; i++)
        {
            float* ptr = A.row(i);
            for (int j = 0; j < A.w; j++) ptr[j] = A0[j * A0_hstep + i];
        }
    }

    Mat BT;
    if (transB == 0)
    {
        BT.create((B0.dims == 3 ? B0.c : B0.h), B0.w, elemsize, opt.workspace_allocator);
        const int B0_hstep = B0.dims == 3 ? (int)B0.cstep : B0.w;
        for (int i = 0; i < BT.h; i++)
        {
            float* ptr = BT.row(i);
            for (int j = 0; j < BT.w; j++) ptr[j] = B0[j * B0_hstep + i];
        }
    }
    else BT = B0;

    const int M = A.dims == 3 ? A.c : A.h;
    const int N = BT.dims == 3 ? BT.c : BT.h;

    Mat C;
    int broadcast_type_C = 0;
    if (constantC)
    {
        C = C_data;
        broadcast_type_C = constant_broadcast_type_C;
    }
    else
    {
        if (constantA && constantB && bottom_blobs.size() == 1) C = bottom_blobs[0];
        else if ((constantA || constantB) && bottom_blobs.size() == 2) C = bottom_blobs[1];
        else if (bottom_blobs.size() == 3) C = bottom_blobs[2];

        if (!C.empty())
        {
            if (C.dims == 1 && C.w == 1) broadcast_type_C = 0;
            if (C.dims == 1 && C.w == M) broadcast_type_C = 1;
            if (C.dims == 1 && C.w == N) broadcast_type_C = 4;
            if (C.dims == 2 && C.w == 1 && C.h == M) broadcast_type_C = 2;
            if (C.dims == 2 && C.w == N && C.h == M) broadcast_type_C = 3;
            if (C.dims == 2 && C.w == N && C.h == 1) broadcast_type_C = 4;
        }
    }

    Mat& top_blob = top_blobs[0];
    if (output_transpose)
    {
        if (output_N1M) top_blob.create(M, 1, N, elemsize, opt.blob_allocator);
        else top_blob.create(M, N, elemsize, opt.blob_allocator);
    }
    else
    {
        if (output_N1M) top_blob.create(N, 1, M, elemsize, opt.blob_allocator);
        else top_blob.create(N, M, elemsize, opt.blob_allocator);
    }
    if (top_blob.empty()) return -100;

    gemm_transB_rvv(A, BT, C, top_blob, alpha, beta, broadcast_type_C, output_transpose, opt);
    return 0;
}

} // namespace ncnn
