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
#endif // __riscv_vector

#include "riscv_usability.h"

#include "cpu.h"

namespace ncnn {

static void gemm_transB_rvv(const Mat& A, const Mat& BT, const Mat& C, Mat& top_blob, float alpha, float beta, int broadcast_type_C, int output_transpose, const Option& opt)
{
    const int M = A.dims == 3 ? A.c : A.h;
    const int N = BT.dims == 3 ? BT.c : BT.h;
    const int K = A.w; // assert A.w == BT.w

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
                if (broadcast_type_C == 0)
                {
                    sum = ptrC[0];
                }
                if (broadcast_type_C == 1)
                {
                    sum = ptrC[i];
                }
                if (broadcast_type_C == 2)
                {
                    sum = ptrC[i];
                }
                if (broadcast_type_C == 3)
                {
                    sum = ptrC[i * N + j];
                }
                if (broadcast_type_C == 4)
                {
                    sum = ptrC[j];
                }

                sum *= beta;
            }

#if __riscv_vector
            int k = K;
            const float* pa = ptrA;
            const float* pb = ptrBT;
            while (k > 0)
            {
                size_t vl = __riscv_vsetvl_e32m8(k);
                vfloat32m8_t _a = __riscv_vle32_v_f32m8(pa, vl);
                vfloat32m8_t _b = __riscv_vle32_v_f32m8(pb, vl);
                vfloat32m8_t _mul = __riscv_vfmul_vv_f32m8(_a, _b, vl);
                vfloat32m1_t _chunk = __riscv_vfmv_s_f_f32m1(0.f, __riscv_vsetvlmax_e32m1());
                _chunk = __riscv_vfredusum_vs_f32m8_f32m1(_mul, _chunk, vl);
                sum += __riscv_vfmv_f_s_f32m1_f32(_chunk);
                pa += vl;
                pb += vl;
                k -= (int)vl;
            }
#else
            for (int k = 0; k < K; k++)
            {
                sum += ptrA[k] * ptrBT[k];
            }
#endif // __riscv_vector

            sum *= alpha;

            if (output_transpose)
            {
                top_blob[j * out_hstep + i] = sum;
            }
            else
            {
                top_blob[i * out_hstep + j] = sum;
            }
        }
    }
}

Gemm_riscv::Gemm_riscv()
{
    nT = 0;
}

int Gemm_riscv::create_pipeline(const Option& opt)
{
    (void)opt;
    return 0;
}

int Gemm_riscv::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
#if NCNN_INT8
    if (int8_scale_term)
    {
        // fall back to base int8 path
        return Gemm::forward_int8(bottom_blobs, top_blobs, opt);
    }
#endif // NCNN_INT8

    const Mat& A0 = constantA ? A_data : bottom_blobs[0];
    const Mat& B0 = constantB ? B_data : constantA ? bottom_blobs[0] : bottom_blobs[1];

    size_t elemsize = A0.elemsize;

    Mat A;
    if (transA == 0)
    {
        A = A0;
    }
    else
    {
        // transpose A to row-major
        A.create((A0.dims == 3 ? A0.c : A0.h), A0.w, elemsize, opt.workspace_allocator);

        const int A0_hstep = A0.dims == 3 ? (int)A0.cstep : A0.w;

        for (int i = 0; i < A.h; i++)
        {
            float* ptr = A.row(i);
            for (int j = 0; j < A.w; j++)
            {
                ptr[j] = A0[j * A0_hstep + i];
            }
        }
    }

    Mat BT;
    if (transB == 0)
    {
        // transpose B to col-major
        BT.create((B0.dims == 3 ? B0.c : B0.h), B0.w, elemsize, opt.workspace_allocator);

        const int B0_hstep = B0.dims == 3 ? (int)B0.cstep : B0.w;

        for (int i = 0; i < BT.h; i++)
        {
            float* ptr = BT.row(i);
            for (int j = 0; j < BT.w; j++)
            {
                ptr[j] = B0[j * B0_hstep + i];
            }
        }
    }
    else
    {
        BT = B0;
    }

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
        if (constantA && constantB && bottom_blobs.size() == 1)
        {
            C = bottom_blobs[0];
        }
        else if ((constantA || constantB) && bottom_blobs.size() == 2)
        {
            C = bottom_blobs[1];
        }
        else if (bottom_blobs.size() == 3)
        {
            C = bottom_blobs[2];
        }

        if (!C.empty())
        {
            if (C.dims == 1 && C.w == 1)
            {
                // scalar
                broadcast_type_C = 0;
            }
            if (C.dims == 1 && C.w == M)
            {
                // M
                // auto broadcast from h to w is the ncnn-style convention
                broadcast_type_C = 1;
            }
            if (C.dims == 1 && C.w == N)
            {
                // N
                broadcast_type_C = 4;
            }
            if (C.dims == 2 && C.w == 1 && C.h == M)
            {
                // Mx1
                broadcast_type_C = 2;
            }
            if (C.dims == 2 && C.w == N && C.h == M)
            {
                // MxN
                broadcast_type_C = 3;
            }
            if (C.dims == 2 && C.w == N && C.h == 1)
            {
                // 1xN
                broadcast_type_C = 4;
            }
        }
    }

    Mat& top_blob = top_blobs[0];
    if (output_transpose)
    {
        if (output_N1M)
            top_blob.create(M, 1, N, elemsize, opt.blob_allocator);
        else
            top_blob.create(M, N, elemsize, opt.blob_allocator);
    }
    else
    {
        if (output_N1M)
            top_blob.create(N, 1, M, elemsize, opt.blob_allocator);
        else
            top_blob.create(N, M, elemsize, opt.blob_allocator);
    }
    if (top_blob.empty())
        return -100;

    gemm_transB_rvv(A, BT, C, top_blob, alpha, beta, broadcast_type_C, output_transpose, opt);

    return 0;
}

#if NCNN_INT8
int Gemm_riscv::forward_int8(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    return Gemm::forward_int8(bottom_blobs, top_blobs, opt);
}
#endif // NCNN_INT8

} // namespace ncnn
