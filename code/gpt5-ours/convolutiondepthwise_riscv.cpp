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

#include "convolutiondepthwise_riscv.h"

#include "layer_type.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

#include "riscv_activation.h"
#include "riscv_usability.h"

namespace ncnn {

#include "../x86/convolutiondepthwise_3x3.h" // reference for scalar layout only, packn kernels implemented below

ConvolutionDepthWise_riscv::ConvolutionDepthWise_riscv()
{
#if __riscv_vector
    support_packing = false;
#endif // __riscv_vector

    activation = 0;
}

int ConvolutionDepthWise_riscv::create_pipeline(const Option& opt)
{
    if (dynamic_weight)
        return 0;

    activation = create_activation_layer(activation_type, activation_params, opt);
    // Do not transform or release weights here; rely on base implementation to ensure correctness across all cases
    return 0;
}

int ConvolutionDepthWise_riscv::create_group_ops(const Option& opt)
{
    const int maxk = kernel_w * kernel_h;
    int channels = (weight_data_size / group) / maxk / (num_output / group) * group;

    for (int i = 0; i < (int)group_ops.size(); i++)
        delete group_ops[i];

    group_ops.clear();

    const int channels_g = channels / group;
    const int num_output_g = num_output / group;

    group_ops.resize(group);

    for (int g = 0; g < group; g++)
    {
        Mat weight_data_g = weight_data.range(maxk * channels_g * num_output_g * g, maxk * channels_g * num_output_g).clone();
        Mat bias_data_g;
        if (bias_term)
            bias_data_g = bias_data.range(num_output_g * g, num_output_g);

        ncnn::Layer* op = ncnn::create_layer_cpu(ncnn::LayerType::Convolution);

        ParamDict pd;
        pd.set(0, num_output_g);
        pd.set(1, kernel_w);
        pd.set(11, kernel_h);
        pd.set(2, dilation_w);
        pd.set(12, dilation_h);
        pd.set(3, stride_w);
        pd.set(13, stride_h);
        pd.set(4, 0);
        pd.set(14, 0);
        pd.set(5, bias_term);
        pd.set(6, maxk * channels_g * num_output_g);
        pd.set(8, int8_scale_term);
        pd.set(9, activation_type);
        pd.set(10, activation_params);

        op->load_param(pd);

        if (bias_term)
        {
            ncnn::Mat weights[5];
            weights[0] = weight_data_g;
            weights[1] = bias_data_g;
#if NCNN_INT8
            if (int8_scale_term)
            {
                Mat weight_data_int8_scales_g(num_output_g);
                weight_data_int8_scales_g.fill(weight_data_int8_scales[g]);
                weights[2] = weight_data_int8_scales_g;
                weights[3] = bottom_blob_int8_scales.range(g, 1);
            }
            if (int8_scale_term > 100)
            {
                weights[4] = top_blob_int8_scales.range(g, 1);
            }
#endif
            op->load_model(ModelBinFromMatArray(weights));
        }
        else
        {
            ncnn::Mat weights[4];
            weights[0] = weight_data_g;
#if NCNN_INT8
            if (int8_scale_term)
            {
                Mat weight_data_int8_scales_g(num_output_g);
                weight_data_int8_scales_g.fill(weight_data_int8_scales[g]);
                weights[1] = weight_data_int8_scales_g;
                weights[2] = bottom_blob_int8_scales.range(g, 1);
            }
            if (int8_scale_term > 100)
            {
                weights[3] = top_blob_int8_scales.range(g, 1);
            }
#endif
            op->load_model(ModelBinFromMatArray(weights));
        }

        op->create_pipeline(opt);

        group_ops[g] = op;
    }

    return 0;
}

int ConvolutionDepthWise_riscv::destroy_pipeline(const Option& opt)
{
    if (activation)
    {
        activation->destroy_pipeline(opt);
        delete activation;
        activation = 0;
    }

    for (int i = 0; i < (int)group_ops.size(); i++)
    {
        group_ops[i]->destroy_pipeline(opt);
        delete group_ops[i];
    }
    group_ops.clear();

    return 0;
}

static inline void convdw3x3s1_packn_rvv(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
    const size_t vl = __riscv_vsetvl_e32m8(packn * 8) / 8 * 1; // use m8 lanes, adapt by vl

    int outw = top_blob.w;
    int outh = top_blob.h;

    const int group = bottom_blob.c;

    const float* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int g = 0; g < group; g++)
    {
        Mat out = top_blob.channel(g);

        vfloat32m8_t _bias0 = bias ? __riscv_vle32_v_f32m8((const float*)bias + g * packn, __riscv_vsetvl_e32m8(packn)) : __riscv_vfmv_v_f_f32m8(0.f, __riscv_vsetvl_e32m8(packn));

        const float* k0 = kernel.row(g);

        float* outptr0 = out.row(0);

        const Mat img0 = bottom_blob.channel(g);

        const float* r0 = img0.row(0);
        const float* r1 = img0.row(1);
        const float* r2 = img0.row(2);

        vfloat32m8_t _k00 = __riscv_vle32_v_f32m8(k0 + 0 * packn, __riscv_vsetvl_e32m8(packn));
        vfloat32m8_t _k01 = __riscv_vle32_v_f32m8(k0 + 1 * packn, __riscv_vsetvl_e32m8(packn));
        vfloat32m8_t _k02 = __riscv_vle32_v_f32m8(k0 + 2 * packn, __riscv_vsetvl_e32m8(packn));
        vfloat32m8_t _k10 = __riscv_vle32_v_f32m8(k0 + 3 * packn, __riscv_vsetvl_e32m8(packn));
        vfloat32m8_t _k11 = __riscv_vle32_v_f32m8(k0 + 4 * packn, __riscv_vsetvl_e32m8(packn));
        vfloat32m8_t _k12 = __riscv_vle32_v_f32m8(k0 + 5 * packn, __riscv_vsetvl_e32m8(packn));
        vfloat32m8_t _k20 = __riscv_vle32_v_f32m8(k0 + 6 * packn, __riscv_vsetvl_e32m8(packn));
        vfloat32m8_t _k21 = __riscv_vle32_v_f32m8(k0 + 7 * packn, __riscv_vsetvl_e32m8(packn));
        vfloat32m8_t _k22 = __riscv_vle32_v_f32m8(k0 + 8 * packn, __riscv_vsetvl_e32m8(packn));

        for (int i = 0; i < outh; i++)
        {
            int j = 0;
            for (; j < outw; j++)
            {
                vfloat32m8_t _sum0 = _bias0;

                vfloat32m8_t _r00 = __riscv_vle32_v_f32m8(r0 + j * packn, __riscv_vsetvl_e32m8(packn));
                vfloat32m8_t _r01 = __riscv_vle32_v_f32m8(r0 + (j + 1) * packn, __riscv_vsetvl_e32m8(packn));
                vfloat32m8_t _r02 = __riscv_vle32_v_f32m8(r0 + (j + 2) * packn, __riscv_vsetvl_e32m8(packn));
                vfloat32m8_t _r10 = __riscv_vle32_v_f32m8(r1 + j * packn, __riscv_vsetvl_e32m8(packn));
                vfloat32m8_t _r11 = __riscv_vle32_v_f32m8(r1 + (j + 1) * packn, __riscv_vsetvl_e32m8(packn));
                vfloat32m8_t _r12 = __riscv_vle32_v_f32m8(r1 + (j + 2) * packn, __riscv_vsetvl_e32m8(packn));
                vfloat32m8_t _r20 = __riscv_vle32_v_f32m8(r2 + j * packn, __riscv_vsetvl_e32m8(packn));
                vfloat32m8_t _r21 = __riscv_vle32_v_f32m8(r2 + (j + 1) * packn, __riscv_vsetvl_e32m8(packn));
                vfloat32m8_t _r22 = __riscv_vle32_v_f32m8(r2 + (j + 2) * packn, __riscv_vsetvl_e32m8(packn));

                _sum0 = __riscv_vfmacc_vv_f32m8(_sum0, _k00, _r00, __riscv_vsetvl_e32m8(packn));
                _sum0 = __riscv_vfmacc_vv_f32m8(_sum0, _k01, _r01, __riscv_vsetvl_e32m8(packn));
                _sum0 = __riscv_vfmacc_vv_f32m8(_sum0, _k02, _r02, __riscv_vsetvl_e32m8(packn));
                _sum0 = __riscv_vfmacc_vv_f32m8(_sum0, _k10, _r10, __riscv_vsetvl_e32m8(packn));
                _sum0 = __riscv_vfmacc_vv_f32m8(_sum0, _k11, _r11, __riscv_vsetvl_e32m8(packn));
                _sum0 = __riscv_vfmacc_vv_f32m8(_sum0, _k12, _r12, __riscv_vsetvl_e32m8(packn));
                _sum0 = __riscv_vfmacc_vv_f32m8(_sum0, _k20, _r20, __riscv_vsetvl_e32m8(packn));
                _sum0 = __riscv_vfmacc_vv_f32m8(_sum0, _k21, _r21, __riscv_vsetvl_e32m8(packn));
                _sum0 = __riscv_vfmacc_vv_f32m8(_sum0, _k22, _r22, __riscv_vsetvl_e32m8(packn));

                __riscv_vse32_v_f32m8(outptr0 + j * packn, _sum0, __riscv_vsetvl_e32m8(packn));
            }

            r0 += bottom_blob.w * packn;
            r1 += bottom_blob.w * packn;
            r2 += bottom_blob.w * packn;
            outptr0 += outw * packn;
        }
    }
#else
    (void)bottom_blob; (void)top_blob; (void)kernel; (void)_bias; (void)opt;
#endif // __riscv_vector
}

static inline void convdw3x3s2_packn_rvv(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
#if __riscv_vector
    const int packn = csrr_vlenb() / 4;

    int outw = top_blob.w;
    int outh = top_blob.h;

    const int group = bottom_blob.c;

    const float* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int g = 0; g < group; g++)
    {
        Mat out = top_blob.channel(g);

        vfloat32m8_t _bias0 = bias ? __riscv_vle32_v_f32m8((const float*)bias + g * packn, __riscv_vsetvl_e32m8(packn)) : __riscv_vfmv_v_f_f32m8(0.f, __riscv_vsetvl_e32m8(packn));

        const float* k0 = kernel.row(g);

        float* outptr = out;

        const Mat img0 = bottom_blob.channel(g);

        const float* r0 = img0.row(0);
        const float* r1 = img0.row(1);
        const float* r2 = img0.row(2);

        vfloat32m8_t _k00 = __riscv_vle32_v_f32m8(k0 + 0 * packn, __riscv_vsetvl_e32m8(packn));
        vfloat32m8_t _k01 = __riscv_vle32_v_f32m8(k0 + 1 * packn, __riscv_vsetvl_e32m8(packn));
        vfloat32m8_t _k02 = __riscv_vle32_v_f32m8(k0 + 2 * packn, __riscv_vsetvl_e32m8(packn));
        vfloat32m8_t _k10 = __riscv_vle32_v_f32m8(k0 + 3 * packn, __riscv_vsetvl_e32m8(packn));
        vfloat32m8_t _k11 = __riscv_vle32_v_f32m8(k0 + 4 * packn, __riscv_vsetvl_e32m8(packn));
        vfloat32m8_t _k12 = __riscv_vle32_v_f32m8(k0 + 5 * packn, __riscv_vsetvl_e32m8(packn));
        vfloat32m8_t _k20 = __riscv_vle32_v_f32m8(k0 + 6 * packn, __riscv_vsetvl_e32m8(packn));
        vfloat32m8_t _k21 = __riscv_vle32_v_f32m8(k0 + 7 * packn, __riscv_vsetvl_e32m8(packn));
        vfloat32m8_t _k22 = __riscv_vle32_v_f32m8(k0 + 8 * packn, __riscv_vsetvl_e32m8(packn));

        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                vfloat32m8_t _sum = _bias0;

                vfloat32m8_t _r00 = __riscv_vle32_v_f32m8(r0 + (2 * j) * packn, __riscv_vsetvl_e32m8(packn));
                vfloat32m8_t _r01 = __riscv_vle32_v_f32m8(r0 + (2 * j + 1) * packn, __riscv_vsetvl_e32m8(packn));
                vfloat32m8_t _r02 = __riscv_vle32_v_f32m8(r0 + (2 * j + 2) * packn, __riscv_vsetvl_e32m8(packn));
                vfloat32m8_t _r10 = __riscv_vle32_v_f32m8(r1 + (2 * j) * packn, __riscv_vsetvl_e32m8(packn));
                vfloat32m8_t _r11 = __riscv_vle32_v_f32m8(r1 + (2 * j + 1) * packn, __riscv_vsetvl_e32m8(packn));
                vfloat32m8_t _r12 = __riscv_vle32_v_f32m8(r1 + (2 * j + 2) * packn, __riscv_vsetvl_e32m8(packn));
                vfloat32m8_t _r20 = __riscv_vle32_v_f32m8(r2 + (2 * j) * packn, __riscv_vsetvl_e32m8(packn));
                vfloat32m8_t _r21 = __riscv_vle32_v_f32m8(r2 + (2 * j + 1) * packn, __riscv_vsetvl_e32m8(packn));
                vfloat32m8_t _r22 = __riscv_vle32_v_f32m8(r2 + (2 * j + 2) * packn, __riscv_vsetvl_e32m8(packn));

                _sum = __riscv_vfmacc_vv_f32m8(_sum, _k00, _r00, __riscv_vsetvl_e32m8(packn));
                _sum = __riscv_vfmacc_vv_f32m8(_sum, _k01, _r01, __riscv_vsetvl_e32m8(packn));
                _sum = __riscv_vfmacc_vv_f32m8(_sum, _k02, _r02, __riscv_vsetvl_e32m8(packn));
                _sum = __riscv_vfmacc_vv_f32m8(_sum, _k10, _r10, __riscv_vsetvl_e32m8(packn));
                _sum = __riscv_vfmacc_vv_f32m8(_sum, _k11, _r11, __riscv_vsetvl_e32m8(packn));
                _sum = __riscv_vfmacc_vv_f32m8(_sum, _k12, _r12, __riscv_vsetvl_e32m8(packn));
                _sum = __riscv_vfmacc_vv_f32m8(_sum, _k20, _r20, __riscv_vsetvl_e32m8(packn));
                _sum = __riscv_vfmacc_vv_f32m8(_sum, _k21, _r21, __riscv_vsetvl_e32m8(packn));
                _sum = __riscv_vfmacc_vv_f32m8(_sum, _k22, _r22, __riscv_vsetvl_e32m8(packn));

                __riscv_vse32_v_f32m8(outptr + j * packn, _sum, __riscv_vsetvl_e32m8(packn));
            }
            r0 += bottom_blob.w * packn * 2;
            r1 += bottom_blob.w * packn * 2;
            r2 += bottom_blob.w * packn * 2;
            outptr += outw * packn;
        }
    }
#else
    (void)bottom_blob; (void)top_blob; (void)kernel; (void)_bias; (void)opt;
#endif // __riscv_vector
}

int ConvolutionDepthWise_riscv::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int elempack = bottom_blob.elempack;

#if __riscv_vector
    int out_elempack = 1;
    if (opt.use_packing_layout)
    {
        const int packn = csrr_vlenb() / 4;
        out_elempack = num_output % packn == 0 ? packn : 1;
    }
#else
    int out_elempack = 1;
#endif

    if (elempack == 1 && out_elempack == 1)
    {
        return ConvolutionDepthWise::forward(bottom_blob, top_blob, opt);
    }

    // Fallback per packn_spec: unpack -> base forward -> repack
    Mat bottom_blob_fp;
    convert_packing(bottom_blob, bottom_blob_fp, 1, opt);

    Mat top_blob_fp;
    int ret = ConvolutionDepthWise::forward(bottom_blob_fp, top_blob_fp, opt);
    if (ret != 0)
        return ret;

    if (out_elempack != 1)
    {
        convert_packing(top_blob_fp, top_blob, out_elempack, opt);
    }
    else
    {
        top_blob = top_blob_fp;
    }

    return 0;
}

int ConvolutionDepthWise_riscv::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    // dynamic weight path fallback to generic implementation by base class
    return ConvolutionDepthWise::forward(bottom_blobs, top_blobs, opt);
}

} // namespace ncnn
