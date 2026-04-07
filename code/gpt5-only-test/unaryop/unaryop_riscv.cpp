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

        int i = 0;
#if __riscv_vector
        int n = size;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(n);
            vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
            _p = op.func_pack(_p, vl);
            __riscv_vse32_v_f32m8(ptr, _p, vl);
            ptr += vl;
            n -= vl;
        }
#else  // __riscv_vector
        for (; i < size; i++)
        {
            *ptr = op.func(*ptr);
            ptr++;
        }
#endif // __riscv_vector
    }

    return 0;
}

namespace UnaryOp_riscv_functor {

struct unary_op_abs
{
    float func(const float& x) const { return (float)fabsf(x); }
#if __riscv_vector
    vfloat32m8_t func_pack(const vfloat32m8_t& x, size_t vl) const
    {
        // __riscv_vfabs_v_f32m8 is implemented via vfsgnjx
        return __riscv_vfsgnjx_vv_f32m8(x, x, vl);
    }
#endif // __riscv_vector
};

struct unary_op_neg
{
    float func(const float& x) const { return -x; }
#if __riscv_vector
    vfloat32m8_t func_pack(const vfloat32m8_t& x, size_t vl) const
    {
        return __riscv_vfneg_v_f32m8(x, vl);
    }
#endif // __riscv_vector
};

struct unary_op_floor
{
    float func(const float& x) const { return (float)floorf(x); }
#if __riscv_vector
    vfloat32m8_t func_pack(const vfloat32m8_t& x, size_t vl) const
    {
        // emulate floor using trunc and fix-up when trunc(x) > x
        vint32m8_t xi = __riscv_vfcvt_x_f_v_i32m8(x, vl);
        vfloat32m8_t xf = __riscv_vfcvt_f_x_v_f32m8(xi, vl);
        vbool4_t gt = __riscv_vmfgt_vv_f32m8_b4(xf, x, vl);
        xi = __riscv_vadd_vx_i32m8_m(gt, xi, xi, 1, vl);
        return __riscv_vfcvt_f_x_v_f32m8(xi, vl);
    }
#endif // __riscv_vector
};

struct unary_op_ceil
{
    float func(const float& x) const { return (float)ceilf(x); }
#if __riscv_vector
    vfloat32m8_t func_pack(const vfloat32m8_t& x, size_t vl) const
    {
        // emulate ceil using trunc and fix-up when trunc(x) < x
        vint32m8_t xi = __riscv_vfcvt_x_f_v_i32m8(x, vl);
        vfloat32m8_t xf = __riscv_vfcvt_f_x_v_f32m8(xi, vl);
        vbool4_t lt = __riscv_vmflt_vv_f32m8_b4(xf, x, vl);
        xi = __riscv_vsub_vx_i32m8_m(lt, xi, xi, 1, vl);
        return __riscv_vfcvt_f_x_v_f32m8(xi, vl);
    }
#endif // __riscv_vector
};

struct unary_op_square
{
    float func(const float& x) const { return x * x; }
#if __riscv_vector
    vfloat32m8_t func_pack(const vfloat32m8_t& x, size_t vl) const
    {
        return __riscv_vfmul_vv_f32m8(x, x, vl);
    }
#endif // __riscv_vector
};

struct unary_op_sqrt
{
    float func(const float& x) const { return (float)sqrtf(x); }
#if __riscv_vector
    vfloat32m8_t func_pack(const vfloat32m8_t& x, size_t vl) const
    {
        return __riscv_vfsqrt_v_f32m8(x, vl);
    }
#endif // __riscv_vector
};

struct unary_op_rsqrt
{
    float func(const float& x) const { return 1.f / sqrtf(x); }
#if __riscv_vector
    vfloat32m8_t func_pack(const vfloat32m8_t& x, size_t vl) const
    {
        // rvv lacks direct rsqrt, compute 1/sqrt(x)
        vfloat32m8_t s = __riscv_vfsqrt_v_f32m8(x, vl);
        vfloat32m8_t one = __riscv_vfmv_v_f_f32m8(1.f, vl);
        return __riscv_vfdiv_vv_f32m8(one, s, vl);
    }
#endif // __riscv_vector
};

struct unary_op_exp
{
    float func(const float& x) const { return (float)expf(x); }
#if __riscv_vector
    vfloat32m8_t func_pack(const vfloat32m8_t& x, size_t vl) const
    {
        return exp_ps(x, vl);
    }
#endif // __riscv_vector
};

struct unary_op_log
{
    float func(const float& x) const { return (float)logf(x); }
#if __riscv_vector
    vfloat32m8_t func_pack(const vfloat32m8_t& x, size_t vl) const
    {
        return log_ps(x, vl);
    }
#endif // __riscv_vector
};

struct unary_op_sin
{
    float func(const float& x) const { return (float)sinf(x); }
#if __riscv_vector
    vfloat32m8_t func_pack(const vfloat32m8_t& x, size_t vl) const
    {
        return sin_ps(x, vl);
    }
#endif // __riscv_vector
};

struct unary_op_cos
{
    float func(const float& x) const { return (float)cosf(x); }
#if __riscv_vector
    vfloat32m8_t func_pack(const vfloat32m8_t& x, size_t vl) const
    {
        return cos_ps(x, vl);
    }
#endif // __riscv_vector
};

struct unary_op_tan
{
    float func(const float& x) const { return (float)tanf(x); }
#if __riscv_vector
    vfloat32m8_t func_pack(const vfloat32m8_t& x, size_t vl) const
    {
        // tan(x) = sin(x)/cos(x)
        vfloat32m8_t s = sin_ps(x, vl);
        vfloat32m8_t c = cos_ps(x, vl);
        return __riscv_vfdiv_vv_f32m8(s, c, vl);
    }
#endif // __riscv_vector
};

struct unary_op_asin
{
    float func(const float& x) const { return (float)asinf(x); }
};

struct unary_op_acos
{
    float func(const float& x) const { return (float)acosf(x); }
};

struct unary_op_atan
{
    float func(const float& x) const { return (float)atanf(x); }
};

struct unary_op_reciprocal
{
    float func(const float& x) const { return 1.f / x; }
#if __riscv_vector
    vfloat32m8_t func_pack(const vfloat32m8_t& x, size_t vl) const
    {
        // 1/x
        vfloat32m8_t one = __riscv_vfmv_v_f_f32m8(1.f, vl);
        return __riscv_vfdiv_vv_f32m8(one, x, vl);
    }
#endif // __riscv_vector
};

struct unary_op_tanh
{
    float func(const float& x) const { return (float)tanhf(x); }
#if __riscv_vector
    vfloat32m8_t func_pack(const vfloat32m8_t& x, size_t vl) const
    {
        // tanh(x) via activation helper
        // tanh(x) = (exp(2x)-1)/(exp(2x)+1)
        vfloat32m8_t ex = exp_ps(__riscv_vfmul_vf_f32m8(x, 2.f, vl), vl);
        vfloat32m8_t num = __riscv_vfsub_vf_f32m8(ex, 1.f, vl);
        vfloat32m8_t den = __riscv_vfadd_vf_f32m8(ex, 1.f, vl);
        return __riscv_vfdiv_vv_f32m8(num, den, vl);
    }
#endif // __riscv_vector
};

struct unary_op_log10
{
    float func(const float& x) const { return (float)log10f(x); }
#if __riscv_vector
    vfloat32m8_t func_pack(const vfloat32m8_t& x, size_t vl) const
    {
        // log10(x) = log(x) * 1/log(10)
        const float invlog10 = 0.434294481903f;
        return __riscv_vfmul_vf_f32m8(log_ps(x, vl), invlog10, vl);
    }
#endif // __riscv_vector
};

struct unary_op_round
{
    float func(const float& x) const { return nearbyintf(x); }
#if __riscv_vector
    vfloat32m8_t func_pack(const vfloat32m8_t& x, size_t vl) const
    {
        // round to nearest even via float->int->float conversion
        // RISC-V default rounding mode is ties-to-even
        vint32m8_t xi = __riscv_vfcvt_x_f_v_i32m8(x, vl);
        return __riscv_vfcvt_f_x_v_f32m8(xi, vl);
    }
#endif // __riscv_vector
};

struct unary_op_trunc
{
    float func(const float& x) const { return (float)truncf(x); }
#if __riscv_vector
    vfloat32m8_t func_pack(const vfloat32m8_t& x, size_t vl) const
    {
        // trunc towards zero
        vint32m8_t xi = __riscv_vfcvt_x_f_v_i32m8(x, vl);
        return __riscv_vfcvt_f_x_v_f32m8(xi, vl);
    }
#endif // __riscv_vector
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
