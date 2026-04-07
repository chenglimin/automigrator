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

#include "binaryop_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#include "rvv_mathfun.h"
#endif // __riscv_vector

#include "cpu.h"

namespace ncnn {

BinaryOp_riscv::BinaryOp_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector
}

#if __riscv_vector
namespace BinaryOp_riscv_functor {

struct binary_op_add
{
    vfloat32m8_t func_pack(const vfloat32m8_t& x, const vfloat32m8_t& y, size_t vl) const { return __riscv_vfadd_vv_f32m8(x, y, vl); }
    float func_scalar(const float& x, const float& y) const { return x + y; }
};

struct binary_op_sub
{
    vfloat32m8_t func_pack(const vfloat32m8_t& x, const vfloat32m8_t& y, size_t vl) const { return __riscv_vfsub_vv_f32m8(x, y, vl); }
    float func_scalar(const float& x, const float& y) const { return x - y; }
};

struct binary_op_mul
{
    vfloat32m8_t func_pack(const vfloat32m8_t& x, const vfloat32m8_t& y, size_t vl) const { return __riscv_vfmul_vv_f32m8(x, y, vl); }
    float func_scalar(const float& x, const float& y) const { return x * y; }
};

struct binary_op_div
{
    vfloat32m8_t func_pack(const vfloat32m8_t& x, const vfloat32m8_t& y, size_t vl) const { return __riscv_vfdiv_vv_f32m8(x, y, vl); }
    float func_scalar(const float& x, const float& y) const { return x / y; }
};

struct binary_op_max
{
    vfloat32m8_t func_pack(const vfloat32m8_t& x, const vfloat32m8_t& y, size_t vl) const { return __riscv_vfmax_vv_f32m8(x, y, vl); }
    float func_scalar(const float& x, const float& y) const { return std::max(x, y); }
};

struct binary_op_min
{
    vfloat32m8_t func_pack(const vfloat32m8_t& x, const vfloat32m8_t& y, size_t vl) const { return __riscv_vfmin_vv_f32m8(x, y, vl); }
    float func_scalar(const float& x, const float& y) const { return std::min(x, y); }
};

struct binary_op_pow
{
    vfloat32m8_t func_pack(const vfloat32m8_t& x, const vfloat32m8_t& y, size_t vl) const { return pow_ps(x, y, vl); }
    float func_scalar(const float& x, const float& y) const { return (float)powf(x, y); }
};

struct binary_op_rsub
{
    vfloat32m8_t func_pack(const vfloat32m8_t& x, const vfloat32m8_t& y, size_t vl) const { return __riscv_vfsub_vv_f32m8(y, x, vl); }
    float func_scalar(const float& x, const float& y) const { return y - x; }
};

struct binary_op_rdiv
{
    vfloat32m8_t func_pack(const vfloat32m8_t& x, const vfloat32m8_t& y, size_t vl) const { return __riscv_vfdiv_vv_f32m8(y, x, vl); }
    float func_scalar(const float& x, const float& y) const { return y / x; }
};

struct binary_op_rpow
{
    vfloat32m8_t func_pack(const vfloat32m8_t& x, const vfloat32m8_t& y, size_t vl) const { return pow_ps(y, x, vl); }
    float func_scalar(const float& x, const float& y) const { return (float)powf(y, x); }
};

struct binary_op_atan2
{
    vfloat32m8_t func_pack(const vfloat32m8_t& x, const vfloat32m8_t& y, size_t vl) const { return atan2_ps(x, y, vl); }
    float func_scalar(const float& x, const float& y) const { return (float)atan2f(x, y); }
};

struct binary_op_ratan2
{
    vfloat32m8_t func_pack(const vfloat32m8_t& x, const vfloat32m8_t& y, size_t vl) const { return atan2_ps(y, x, vl); }
    float func_scalar(const float& x, const float& y) const { return (float)atan2f(y, x); }
};

} // namespace BinaryOp_riscv_functor

// Vector helper operating on contiguous memory
template<typename Op>
static void binary_op_vector_contiguous(const float* a, const float* b, float* c, int size)
{
    Op op;
    int n = size;
    while (n > 0)
    {
        size_t vl = __riscv_vsetvl_e32m8(n);
        vfloat32m8_t _a = __riscv_vle32_v_f32m8(a, vl);
        vfloat32m8_t _b = __riscv_vle32_v_f32m8(b, vl);
        vfloat32m8_t _c = op.func_pack(_a, _b, vl);
        __riscv_vse32_v_f32m8(c, _c, vl);
        a += vl; b += vl; c += vl; n -= vl;
    }
}

// Broadcast second operand scalar across vector
template<typename Op>
static void binary_op_vector_broadcast_b_contiguous(const float* a, float b, float* c, int size)
{
    Op op;
    int n = size;
    while (n > 0)
    {
        size_t vl = __riscv_vsetvl_e32m8(n);
        vfloat32m8_t _a = __riscv_vle32_v_f32m8(a, vl);
        vfloat32m8_t _b = __riscv_vfmv_v_f_f32m8(b, vl);
        vfloat32m8_t _c = op.func_pack(_a, _b, vl);
        __riscv_vse32_v_f32m8(c, _c, vl);
        a += vl; c += vl; n -= vl;
    }
}

// Broadcast first operand scalar across vector
template<typename Op>
static void binary_op_vector_broadcast_a_contiguous(float a, const float* b, float* c, int size)
{
    Op op;
    int n = size;
    while (n > 0)
    {
        size_t vl = __riscv_vsetvl_e32m8(n);
        vfloat32m8_t _a = __riscv_vfmv_v_f_f32m8(a, vl);
        vfloat32m8_t _b = __riscv_vle32_v_f32m8(b, vl);
        vfloat32m8_t _c = op.func_pack(_a, _b, vl);
        __riscv_vse32_v_f32m8(c, _c, vl);
        b += vl; c += vl; n -= vl;
    }
}

// Generic vector wrapper matching x86 logic for aw,bw,ap,bp
template<typename Op>
static void binary_op_vector(const float* ptr, const float* ptr1, float* outptr, int aw, int bw, int ap, int bp)
{
    const int w = std::max(aw, bw);
    const int elempack = std::max(ap, bp);
    const int size = w * elempack;

    if (ap == bp)
    {
        if (aw == bw)
        {
            return binary_op_vector_contiguous<Op>(ptr, ptr1, outptr, size);
        }
        if (bw == 1)
        {
            return binary_op_vector_broadcast_b_contiguous<Op>(ptr, *ptr1, outptr, size);
        }
        if (aw == 1)
        {
            return binary_op_vector_broadcast_a_contiguous<Op>(*ptr, ptr1, outptr, size);
        }
    }

    // bp == 1 path: pack1 on b. Iterate per pack-size chunk
    if (bp == 1)
    {
        if (aw == bw)
        {
            // layout like [pack] sequential rows
            for (int i = 0; i < w; i++)
            {
                binary_op_vector_broadcast_b_contiguous<Op>(ptr, ptr1[i], outptr, elempack);
                ptr += elempack;
                outptr += elempack;
            }
            return;
        }
        if (bw == 1)
        {
            // single scalar b
            return binary_op_vector_broadcast_b_contiguous<Op>(ptr, *ptr1, outptr, size);
        }
        if (aw == 1)
        {
            // single a broadcast with varying b
            for (int i = 0; i < w; i++)
            {
                binary_op_vector_broadcast_a_contiguous<Op>(*ptr, ptr1 + i, outptr, elempack);
                outptr += elempack;
            }
            return;
        }
    }
}

static void binary_op_vector(const float* ptr, const float* ptr1, float* outptr, int aw, int bw, int ap, int bp, int op_type)
{
    using namespace BinaryOp_riscv_functor;

    if (op_type == BinaryOp::Operation_ADD) return binary_op_vector<binary_op_add>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_SUB) return binary_op_vector<binary_op_sub>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_MUL) return binary_op_vector<binary_op_mul>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_DIV) return binary_op_vector<binary_op_div>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_MAX) return binary_op_vector<binary_op_max>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_MIN) return binary_op_vector<binary_op_min>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_POW) return binary_op_vector<binary_op_pow>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_RSUB) return binary_op_vector<binary_op_rsub>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_RDIV) return binary_op_vector<binary_op_rdiv>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_RPOW) return binary_op_vector<binary_op_rpow>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_ATAN2) return binary_op_vector<binary_op_atan2>(ptr, ptr1, outptr, aw, bw, ap, bp);
    if (op_type == BinaryOp::Operation_RATAN2) return binary_op_vector<binary_op_ratan2>(ptr, ptr1, outptr, aw, bw, ap, bp);
}

static void binary_op_scalar(const Mat& a, float b, Mat& c, int op_type, const Option& opt)
{
    const int channels = a.c;
    const int size = a.w * a.h * a.d * a.elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* ptr = a.channel(q);
        float* outptr = c.channel(q);

        binary_op_vector(ptr, &b, outptr, size, 1, 1, 1, op_type);
    }
}

static void binary_op_no_broadcast(const Mat& a, const Mat& b, Mat& c, int op_type, const Option& opt)
{
    const int channels = a.c;
    const int size = a.w * a.h * a.d * a.elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* ptr = a.channel(q);
        const float* ptr1 = b.channel(q);
        float* outptr = c.channel(q);

        binary_op_vector(ptr, ptr1, outptr, size, size, 1, 1, op_type);
    }
}

static void binary_op_broadcast(const Mat& a, const Mat& b, Mat& c, int op_type, const Option& opt)
{
    if (b.w * b.h * b.d * b.c * b.elempack == 1)
    {
        return binary_op_scalar(a, b[0], c, op_type, opt);
    }

    if (a.dims == b.dims && a.w == b.w && a.h == b.h && a.d == b.d && a.c == b.c && a.elempack == b.elempack)
    {
        return binary_op_no_broadcast(a, b, c, op_type, opt);
    }

    const int dims = c.dims;

    if (dims == 2)
    {
        const int h = c.h;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int y = 0; y < h; y++)
        {
            const int y0 = std::min(y, a.h - 1);
            const int y1 = std::min(y, b.h - 1);

            const float* ptr = a.row(y0);
            const float* ptr1 = b.row(y1);
            float* outptr = c.row(y);

            binary_op_vector(ptr, ptr1, outptr, a.w, b.w, a.elempack, b.elempack, op_type);
        }
    }

    if (dims == 3 || dims == 4)
    {
        const int channels = c.c;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const int q0 = std::min(q, a.c - 1);
            const int q1 = std::min(q, b.c - 1);

            if (b.d * b.h * b.w == 1)
            {
                const float* ptr = a.channel(q0);
                const float* ptr1 = b.channel(q1);
                float* outptr = c.channel(q);

                binary_op_vector(ptr, ptr1, outptr, a.w * a.h * a.d, 1, a.elempack, b.elempack, op_type);
                continue;
            }

            if (b.h * b.w == 1)
            {
                for (int z = 0; z < c.d; z++)
                {
                    const int z0 = std::min(z, a.d - 1);
                    const int z1 = std::min(z, b.d - 1);

                    const float* ptr = a.channel(q0).depth(z0);
                    const float* ptr1 = b.channel(q1).depth(z1);
                    float* outptr = c.channel(q).depth(z);

                    binary_op_vector(ptr, ptr1, outptr, a.w * a.h, 1, a.elempack, b.elempack, op_type);
                }
                continue;
            }

            for (int z = 0; z < c.d; z++)
            {
                const int z0 = std::min(z, a.d - 1);
                const int z1 = std::min(z, b.d - 1);

                for (int y = 0; y < c.h; y++)
                {
                    const int y0 = std::min(y, a.h - 1);
                    const int y1 = std::min(y, b.h - 1);

                    const float* ptr = a.channel(q0).depth(z0).row(y0);
                    const float* ptr1 = b.channel(q1).depth(z1).row(y1);
                    float* outptr = c.channel(q).depth(z).row(y);

                    binary_op_vector(ptr, ptr1, outptr, a.w, b.w, a.elempack, b.elempack, op_type);
                }
            }
        }
    }
}

static void binary_op_scalar_inplace(Mat& a, float b, int op_type, const Option& opt)
{
    const int channels = a.c;
    const int size = a.w * a.h * a.d * a.elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = a.channel(q);

        binary_op_vector(ptr, &b, ptr, size, 1, 1, 1, op_type);
    }
}

static int get_reverse_op_type(int op_type)
{
    if (op_type == BinaryOp::Operation_SUB) return BinaryOp::Operation_RSUB;
    if (op_type == BinaryOp::Operation_DIV) return BinaryOp::Operation_RDIV;
    if (op_type == BinaryOp::Operation_POW) return BinaryOp::Operation_RPOW;
    if (op_type == BinaryOp::Operation_ATAN2) return BinaryOp::Operation_RATAN2;
    if (op_type == BinaryOp::Operation_RSUB) return BinaryOp::Operation_SUB;
    if (op_type == BinaryOp::Operation_RDIV) return BinaryOp::Operation_DIV;
    if (op_type == BinaryOp::Operation_RPOW) return BinaryOp::Operation_POW;
    if (op_type == BinaryOp::Operation_RATAN2) return BinaryOp::Operation_ATAN2;
    return op_type;
}
#endif // __riscv_vector

int BinaryOp_riscv::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
#if __riscv_vector
    const Mat& A = bottom_blobs[0];
    const Mat& B = bottom_blobs[1];
    const int outdims = std::max(A.dims, B.dims);

    Mat A2 = A;
    Mat B2 = B;
    if (A.dims < outdims)
    {
        if (outdims == 2)
        {
            if (A.w * A.elempack == B.h * B.elempack)
                A2 = A.reshape(1, A.w, opt.workspace_allocator);
            else
            {
                A2.dims = 2;
                A2.w = A.w * A.elempack;
                A2.elempack = 1;
                A2.elemsize = A.elemsize / A.elempack;
                A2.cstep = A2.w;
            }
        }
        if (outdims == 3 && A.dims == 1)
        {
            if (A.w * A.elempack == B.c * B.elempack)
                A2 = A.reshape(1, 1, A.w, opt.workspace_allocator);
            else
            {
                A2.dims = 3;
                A2.w = A.w * A.elempack;
                A2.elempack = 1;
                A2.elemsize = A.elemsize / A.elempack;
                A2.cstep = A2.w;
            }
        }
        if (outdims == 3 && A.dims == 2)
            A2 = A.reshape(1, A.w, A.h, opt.workspace_allocator);
        if (outdims == 4 && A.dims == 1)
        {
            if (A.w * A.elempack == B.c * B.elempack)
                A2 = A.reshape(1, 1, 1, A.w, opt.workspace_allocator);
            else
            {
                A2.dims = 4;
                A2.w = A.w * A.elempack;
                A2.elempack = 1;
                A2.elemsize = A.elemsize / A.elempack;
                A2.cstep = A2.w;
            }
        }
        if (outdims == 4 && A.dims == 2)
            A2 = A.reshape(1, 1, A.w, A.h, opt.workspace_allocator);
        if (outdims == 4 && A.dims == 3)
            A2 = A.reshape(1, A.w, A.h, A.c, opt.workspace_allocator);
    }
    if (B.dims < outdims)
    {
        if (outdims == 2)
        {
            if (B.w * B.elempack == A.h * A.elempack)
                B2 = B.reshape(1, B.w, opt.workspace_allocator);
            else
            {
                B2.dims = 2;
                B2.w = B.w * B.elempack;
                B2.elempack = 1;
                B2.elemsize = B.elemsize / B.elempack;
                B2.cstep = B2.w;
            }
        }
        if (outdims == 3 && B.dims == 1)
        {
            if (B.w * B.elempack == A.c * A.elempack)
                B2 = B.reshape(1, 1, B.w, opt.workspace_allocator);
            else
            {
                B2.dims = 3;
                B2.w = B.w * B.elempack;
                B2.elempack = 1;
                B2.elemsize = B.elemsize / B.elempack;
                B2.cstep = B2.w;
            }
        }
        if (outdims == 3 && B.dims == 2)
            B2 = B.reshape(1, B.w, B.h, opt.workspace_allocator);
        if (outdims == 4 && B.dims == 1)
        {
            if (B.w * B.elempack == A.c * A.elempack)
                B2 = B.reshape(1, 1, 1, B.w, opt.workspace_allocator);
            else
            {
                B2.dims = 4;
                B2.w = B.w * B.elempack;
                B2.elempack = 1;
                B2.elemsize = B.elemsize / B.elempack;
                B2.cstep = B2.w;
            }
        }
        if (outdims == 4 && B.dims == 2)
            B2 = B.reshape(1, 1, B.w, B.h, opt.workspace_allocator);
        if (outdims == 4 && B.dims == 3)
            B2 = B.reshape(1, B.w, B.h, B.c, opt.workspace_allocator);
    }

    const int outw = std::max(A2.w, B2.w);
    const int outh = std::max(A2.h, B2.h);
    const int outd = std::max(A2.d, B2.d);
    const int outc = std::max(A2.c, B2.c);
    const size_t out_elemsize = std::max(A2.elemsize, B2.elemsize);
    const int out_elempack = std::max(A2.elempack, B2.elempack);

    Mat& top_blob = top_blobs[0];
    if (outdims == 1)
    {
        top_blob.create(outw, out_elemsize, out_elempack, opt.blob_allocator);
    }
    if (outdims == 2)
    {
        top_blob.create(outw, outh, out_elemsize, out_elempack, opt.blob_allocator);
    }
    if (outdims == 3)
    {
        top_blob.create(outw, outh, outc, out_elemsize, out_elempack, opt.blob_allocator);
    }
    if (outdims == 4)
    {
        top_blob.create(outw, outh, outd, outc, out_elemsize, out_elempack, opt.blob_allocator);
    }
    if (top_blob.empty())
        return -100;

    const bool a_pack_is_lower = A2.elempack < B2.elempack;
    const bool a_pack_is_equal = A2.elempack == B2.elempack;
    const bool a_size_is_lower = A2.w * A2.h * A2.d * A2.c * A2.elempack < B2.w * B2.h * B2.d * B2.c * B2.elempack;
    if (a_pack_is_lower || (a_pack_is_equal && a_size_is_lower))
    {
        binary_op_broadcast(B2, A2, top_blob, get_reverse_op_type(op_type), opt);
    }
    else
    {
        binary_op_broadcast(A2, B2, top_blob, op_type, opt);
    }

    return 0;
#else
    return BinaryOp::forward(bottom_blobs, top_blobs, opt);
#endif // __riscv_vector
}

int BinaryOp_riscv::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
#if __riscv_vector
    binary_op_scalar_inplace(bottom_top_blob, b, op_type, opt);
    return 0;
#else
    return BinaryOp::forward_inplace(bottom_top_blob, opt);
#endif // __riscv_vector
}

} // namespace ncnn
