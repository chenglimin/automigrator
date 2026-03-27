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

#include "unaryop_riscv.h"

// #include <fenv.h>
#include <float.h>

#if __riscv_vector
#include <riscv_vector.h>
#include "rvv_mathfun.h"
#endif // __riscv_vector

#include "riscv_usability.h"
#include "riscv_activation.h"

namespace ncnn {

UnaryOp_riscv::UnaryOp_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector
}

template<typename Op>
static int unary_op_inplace(Mat& a, const Option& opt)
{
    Op op;

    int w = a.w;
    int h = a.h;
    int d = a.d;
    int channels = a.c;
    int elempack = a.elempack;
    int size = w * h * d * elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = a.channel(q);

#if __riscv_vector
        if (Op::has_vector)
        {
            int n = size;
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e32m8(n);
                vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                _p = op.func_pack8(_p, vl);
                __riscv_vse32_v_f32m8(ptr, _p, vl);

                ptr += vl;
                n -= vl;
            }
            continue;
        }
#endif // __riscv_vector
        for (int i = 0; i < size; i++)
        {
            *ptr = op.func(*ptr);
            ptr++;
        }
    }

    return 0;
}

namespace UnaryOp_riscv_functor {

struct unary_op_abs
{
    static constexpr bool has_vector =
#if __riscv_vector
        true;
#else
        false;
#endif
    float func(const float& x) const { return (float)fabsf(x); }
#if __riscv_vector
    vfloat32m8_t func_pack8(const vfloat32m8_t& x, size_t vl) const
    {
        vuint32m8_t bits = __riscv_vreinterpret_v_f32m8_u32m8(x);
        bits = __riscv_vand_vx_u32m8(bits, 0x7fffffff, vl);
        return __riscv_vreinterpret_v_u32m8_f32m8(bits);
    }
#endif
};

struct unary_op_neg
{
    static constexpr bool has_vector =
#if __riscv_vector
        true;
#else
        false;
#endif
    float func(const float& x) const { return -x; }
#if __riscv_vector
    vfloat32m8_t func_pack8(const vfloat32m8_t& x, size_t vl) const
    {
        return __riscv_vfsub_vf_f32m8(__riscv_vfmv_v_f_f32m8(0.f, vl), x, vl);
    }
#endif
};

struct unary_op_floor
{
    static constexpr bool has_vector =
#if __riscv_vector
        true;
#else
        false;
#endif
    float func(const float& x) const { return (float)floorf(x); }
#if __riscv_vector
    vfloat32m8_t func_pack8(const vfloat32m8_t& x, size_t vl) const
    {
        vfloat32m8_t xi = __riscv_vfcvt_f_x_v_f32m8(__riscv_vfcvt_x_f_v_i32m8(x, vl), vl);
        vbool4_t mask = __riscv_vmfgt_vv_f32m8_b4(xi, x, vl);
        return __riscv_vfsub_vf_f32m8_mu(mask, xi, xi, 1.f, vl);
    }
#endif
};

struct unary_op_ceil
{
    static constexpr bool has_vector =
#if __riscv_vector
        true;
#else
        false;
#endif
    float func(const float& x) const { return (float)ceilf(x); }
#if __riscv_vector
    vfloat32m8_t func_pack8(const vfloat32m8_t& x, size_t vl) const
    {
        vfloat32m8_t xi = __riscv_vfcvt_f_x_v_f32m8(__riscv_vfcvt_x_f_v_i32m8(x, vl), vl);
        vbool4_t mask = __riscv_vmfgt_vv_f32m8_b4(x, xi, vl);
        return __riscv_vfadd_vf_f32m8_mu(mask, xi, xi, 1.f, vl);
    }
#endif
};

struct unary_op_square
{
    static constexpr bool has_vector =
#if __riscv_vector
        true;
#else
        false;
#endif
    float func(const float& x) const { return x * x; }
#if __riscv_vector
    vfloat32m8_t func_pack8(const vfloat32m8_t& x, size_t vl) const
    {
        return __riscv_vfmul_vv_f32m8(x, x, vl);
    }
#endif
};

struct unary_op_sqrt
{
    static constexpr bool has_vector =
#if __riscv_vector
        true;
#else
        false;
#endif
    float func(const float& x) const { return (float)sqrtf(x); }
#if __riscv_vector
    vfloat32m8_t func_pack8(const vfloat32m8_t& x, size_t vl) const
    {
        return __riscv_vfsqrt_v_f32m8(x, vl);
    }
#endif
};

struct unary_op_rsqrt
{
    static constexpr bool has_vector =
#if __riscv_vector
        true;
#else
        false;
#endif
    float func(const float& x) const { return 1.f / sqrtf(x); }
#if __riscv_vector
    vfloat32m8_t func_pack8(const vfloat32m8_t& x, size_t vl) const
    {
        vfloat32m8_t r = __riscv_vfrsqrt_v_f32m8(x, vl);
        vfloat32m8_t rr = __riscv_vfmul_vv_f32m8(r, r, vl);
        vfloat32m8_t halfxr2 = __riscv_vfmul_vf_f32m8(__riscv_vfmul_vv_f32m8(x, rr, vl), 0.5f, vl);
        vfloat32m8_t three_halves = __riscv_vfmv_v_f_f32m8(1.5f, vl);
        vfloat32m8_t term = __riscv_vfsub_vv_f32m8(three_halves, halfxr2, vl);
        return __riscv_vfmul_vv_f32m8(r, term, vl);
    }
#endif
};

struct unary_op_exp
{
    static constexpr bool has_vector =
#if __riscv_vector
        true;
#else
        false;
#endif
    float func(const float& x) const { return (float)expf(x); }
#if __riscv_vector
    vfloat32m8_t func_pack8(const vfloat32m8_t& x, size_t vl) const { return exp_ps(x, vl); }
#endif
};

struct unary_op_log
{
    static constexpr bool has_vector =
#if __riscv_vector
        true;
#else
        false;
#endif
    float func(const float& x) const { return (float)logf(x); }
#if __riscv_vector
    vfloat32m8_t func_pack8(const vfloat32m8_t& x, size_t vl) const { return log_ps(x, vl); }
#endif
};

struct unary_op_sin
{
    static constexpr bool has_vector =
#if __riscv_vector
        true;
#else
        false;
#endif
    float func(const float& x) const { return (float)sinf(x); }
#if __riscv_vector
    vfloat32m8_t func_pack8(const vfloat32m8_t& x, size_t vl) const { return sin_ps(x, vl); }
#endif
};

struct unary_op_cos
{
    static constexpr bool has_vector =
#if __riscv_vector
        true;
#else
        false;
#endif
    float func(const float& x) const { return (float)cosf(x); }
#if __riscv_vector
    vfloat32m8_t func_pack8(const vfloat32m8_t& x, size_t vl) const { return cos_ps(x, vl); }
#endif
};

struct unary_op_tan
{
    static constexpr bool has_vector =
#if __riscv_vector
        true;
#else
        false;
#endif
    float func(const float& x) const { return (float)tanf(x); }
#if __riscv_vector
    vfloat32m8_t func_pack8(const vfloat32m8_t& x, size_t vl) const
    {
        vfloat32m8_t s = sin_ps(x, vl);
        vfloat32m8_t c = cos_ps(x, vl);
        return __riscv_vfdiv_vv_f32m8(s, c, vl);
    }
#endif
};

struct unary_op_asin
{
    static constexpr bool has_vector = false;
    float func(const float& x) const { return (float)asinf(x); }
};

struct unary_op_acos
{
    static constexpr bool has_vector = false;
    float func(const float& x) const { return (float)acosf(x); }
};

struct unary_op_atan
{
    static constexpr bool has_vector =
#if __riscv_vector
        true;
#else
        false;
#endif
    float func(const float& x) const { return (float)atanf(x); }
#if __riscv_vector
    vfloat32m8_t func_pack8(const vfloat32m8_t& x, size_t vl) const
    {
        vfloat32m8_t one = __riscv_vfmv_v_f_f32m8(1.f, vl);
        return atan2_ps(x, one, vl);
    }
#endif
};

struct unary_op_reciprocal
{
    static constexpr bool has_vector =
#if __riscv_vector
        true;
#else
        false;
#endif
    float func(const float& x) const { return 1.f / x; }
#if __riscv_vector
    vfloat32m8_t func_pack8(const vfloat32m8_t& x, size_t vl) const
    {
        return __riscv_vfdiv_vv_f32m8(__riscv_vfmv_v_f_f32m8(1.f, vl), x, vl);
    }
#endif
};

struct unary_op_tanh
{
    static constexpr bool has_vector =
#if __riscv_vector
        true;
#else
        false;
#endif
    float func(const float& x) const { return (float)tanhf(x); }
#if __riscv_vector
    vfloat32m8_t func_pack8(const vfloat32m8_t& x, size_t vl) const { return tanh_ps(x, vl); }
#endif
};

struct unary_op_log10
{
    static constexpr bool has_vector =
#if __riscv_vector
        true;
#else
        false;
#endif
    float func(const float& x) const { return (float)log10f(x); }
#if __riscv_vector
    vfloat32m8_t func_pack8(const vfloat32m8_t& x, size_t vl) const
    {
        vfloat32m8_t lx = log_ps(x, vl);
        return __riscv_vfdiv_vf_f32m8(lx, 2.30258509299f, vl);
    }
#endif
};

struct unary_op_round
{
    static constexpr bool has_vector = false;
    float func(const float& x) const { return nearbyintf(x); }
};

struct unary_op_trunc
{
    static constexpr bool has_vector =
#if __riscv_vector
        true;
#else
        false;
#endif
    float func(const float& x) const { return (float)truncf(x); }
#if __riscv_vector
    vfloat32m8_t func_pack8(const vfloat32m8_t& x, size_t vl) const
    {
        return __riscv_vfcvt_f_x_v_f32m8(__riscv_vfcvt_x_f_v_i32m8(x, vl), vl);
    }
#endif
};

} // namespace UnaryOp_riscv_functor

int UnaryOp_riscv::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    using namespace UnaryOp_riscv_functor;
    if (op_type == Operation_ABS)
        return unary_op_inplace<unary_op_abs>(bottom_top_blob, opt);

    if (op_type == Operation_NEG)
        return unary_op_inplace<unary_op_neg>(bottom_top_blob, opt);

    if (op_type == Operation_FLOOR)
        return unary_op_inplace<unary_op_floor>(bottom_top_blob, opt);

    if (op_type == Operation_CEIL)
        return unary_op_inplace<unary_op_ceil>(bottom_top_blob, opt);

    if (op_type == Operation_SQUARE)
        return unary_op_inplace<unary_op_square>(bottom_top_blob, opt);

    if (op_type == Operation_SQRT)
        return unary_op_inplace<unary_op_sqrt>(bottom_top_blob, opt);

    if (op_type == Operation_RSQRT)
        return unary_op_inplace<unary_op_rsqrt>(bottom_top_blob, opt);

    if (op_type == Operation_EXP)
        return unary_op_inplace<unary_op_exp>(bottom_top_blob, opt);

    if (op_type == Operation_LOG)
        return unary_op_inplace<unary_op_log>(bottom_top_blob, opt);

    if (op_type == Operation_SIN)
        return unary_op_inplace<unary_op_sin>(bottom_top_blob, opt);

    if (op_type == Operation_COS)
        return unary_op_inplace<unary_op_cos>(bottom_top_blob, opt);

    if (op_type == Operation_TAN)
        return unary_op_inplace<unary_op_tan>(bottom_top_blob, opt);

    if (op_type == Operation_ASIN)
        return unary_op_inplace<unary_op_asin>(bottom_top_blob, opt);

    if (op_type == Operation_ACOS)
        return unary_op_inplace<unary_op_acos>(bottom_top_blob, opt);

    if (op_type == Operation_ATAN)
        return unary_op_inplace<unary_op_atan>(bottom_top_blob, opt);

    if (op_type == Operation_RECIPROCAL)
        return unary_op_inplace<unary_op_reciprocal>(bottom_top_blob, opt);

    if (op_type == Operation_TANH)
        return unary_op_inplace<unary_op_tanh>(bottom_top_blob, opt);

    if (op_type == Operation_LOG10)
        return unary_op_inplace<unary_op_log10>(bottom_top_blob, opt);

    if (op_type == Operation_ROUND)
    {
#ifdef FE_TONEAREST
        int old_rm = fegetround();
        fesetround(FE_TONEAREST);
#endif
        int ret = unary_op_inplace<unary_op_round>(bottom_top_blob, opt);
#ifdef FE_TONEAREST
        fesetround(old_rm);
#endif
        return ret;
    }

    if (op_type == Operation_TRUNC)
        return unary_op_inplace<unary_op_trunc>(bottom_top_blob, opt);

    return 0;
}

} // namespace ncnn
