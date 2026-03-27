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

#include "convolution1d_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

#include "riscv_activation.h"
#include "riscv_usability.h"
#include "cpu.h"
#include "layer_type.h"

namespace ncnn {

// We reuse the generic pack layout and compute like x86/arm but implement the rvv kernels here

// Transform kernel to packed layout suitable for rvv compute
static void convolution1d_transform_kernel_packed(const Mat& kernel, Mat& kernel_tm, int inh, int outh, int kernel_w)
{
    // src = kw-inh-outh
    // dst = pb-pa-kw-inh/pa-outh/pb, where pa/pb are 1 or vector groups

    // follow arm/x86 logic but without x86/arm intrinsics; we pack in 4/8 blocks using plain loops
    // choose out pack = 8,4,2,1 and in pack = 8,4,2,1 to match runtime elempack possibilities

    if (outh >= 8)
    {
        if (inh >= 8) kernel_tm.create(8 * 8 * kernel_w, inh / 8 + (inh % 8) / 4 + (inh % 4) / 2 + inh % 2, outh / 8 + (outh % 8) / 4 + (outh % 4) / 2 + outh % 2);
        else if (inh >= 4) kernel_tm.create(8 * 4 * kernel_w, inh / 4 + (inh % 4) / 2 + inh % 2, outh / 8 + (outh % 8) / 4 + (outh % 4) / 2 + outh % 2);
        else if (inh >= 2) kernel_tm.create(8 * 2 * kernel_w, inh / 2 + inh % 2, outh / 8 + (outh % 8) / 4 + (outh % 4) / 2 + outh % 2);
        else kernel_tm.create(8 * kernel_w, inh, outh / 8 + (outh % 8) / 4 + (outh % 4) / 2 + outh % 2);
    }
    else if (outh >= 4)
    {
        if (inh >= 8) kernel_tm.create(4 * 8 * kernel_w, inh / 8 + (inh % 8) / 4 + (inh % 4) / 2 + inh % 2, outh / 4 + (outh % 4) / 2 + outh % 2);
        else if (inh >= 4) kernel_tm.create(4 * 4 * kernel_w, inh / 4 + (inh % 4) / 2 + inh % 2, outh / 4 + (outh % 4) / 2 + outh % 2);
        else if (inh >= 2) kernel_tm.create(4 * 2 * kernel_w, inh / 2 + inh % 2, outh / 4 + (outh % 4) / 2 + outh % 2);
        else kernel_tm.create(4 * kernel_w, inh, outh / 4 + (outh % 4) / 2 + outh % 2);
    }
    else if (outh >= 2)
    {
        if (inh >= 8) kernel_tm.create(2 * 8 * kernel_w, inh / 8 + (inh % 8) / 4 + (inh % 4) / 2 + inh % 2, outh / 2 + outh % 2);
        else if (inh >= 4) kernel_tm.create(2 * 4 * kernel_w, inh / 4 + (inh % 4) / 2 + inh % 2, outh / 2 + outh % 2);
        else if (inh >= 2) kernel_tm.create(2 * 2 * kernel_w, inh / 2 + inh % 2, outh / 2 + outh % 2);
        else kernel_tm.create(2 * kernel_w, inh, outh / 2 + outh % 2);
    }
    else
    {
        if (inh >= 8) kernel_tm.create(8 * kernel_w, inh / 8 + (inh % 8) / 4 + (inh % 4) / 2 + inh % 2, outh);
        else if (inh >= 4) kernel_tm.create(4 * kernel_w, inh / 4 + (inh % 4) / 2 + inh % 2, outh);
        else if (inh >= 2) kernel_tm.create(2 * kernel_w, inh / 2 + inh % 2, outh);
        else kernel_tm.create(kernel_w, inh, outh);
    }

    int q = 0;
    for (; q + 7 < outh; q += 8)
    {
        const float* kptr[8];
        for (int i = 0; i < 8; i++) kptr[i] = (const float*)kernel + (q + i) * inh * kernel_w;
        float* g00 = kernel_tm.channel(q / 8);

        int p = 0;
        for (; p + 7 < inh; p += 8)
        {
            for (int k = 0; k < kernel_w; k++)
            {
                const float* kk[8];
                for (int i = 0; i < 8; i++) kk[i] = kptr[i] + p * kernel_w + k;
                for (int i = 0; i < 8; i++)
                {
                    for (int r = 0; r < 8; r++)
                    {
                        g00[i * 8 + r] = kk[i][r * kernel_w];
                    }
                }
                g00 += 64;
            }
        }
        for (; p + 3 < inh; p += 4)
        {
            for (int k = 0; k < kernel_w; k++)
            {
                const float* kk[8];
                for (int i = 0; i < 8; i++) kk[i] = kptr[i] + p * kernel_w + k;
                for (int i = 0; i < 8; i++)
                {
                    for (int r = 0; r < 4; r++)
                        g00[i * 4 + r] = kk[i][r * kernel_w];
                }
                g00 += 32;
            }
        }
        for (; p + 1 < inh; p += 2)
        {
            for (int k = 0; k < kernel_w; k++)
            {
                const float* kk[8];
                for (int i = 0; i < 8; i++) kk[i] = kptr[i] + p * kernel_w + k;
                for (int i = 0; i < 8; i++)
                {
                    for (int r = 0; r < 2; r++)
                        g00[i * 2 + r] = kk[i][r * kernel_w];
                }
                g00 += 16;
            }
        }
        for (; p < inh; p++)
        {
            for (int k = 0; k < kernel_w; k++)
            {
                for (int i = 0; i < 8; i++)
                {
                    g00[i] = kptr[i][p * kernel_w + k];
                }
                g00 += 8;
            }
        }
    }
    for (; q + 3 < outh; q += 4)
    {
        const float* kptr[4];
        for (int i = 0; i < 4; i++) kptr[i] = (const float*)kernel + (q + i) * inh * kernel_w;
        float* g00 = kernel_tm.channel(q / 8 + (q % 8) / 4);
        int p = 0;
        for (; p + 7 < inh; p += 8)
        {
            for (int k = 0; k < kernel_w; k++)
            {
                const float* kk[4];
                for (int i = 0; i < 4; i++) kk[i] = kptr[i] + p * kernel_w + k;
                for (int i = 0; i < 4; i++)
                {
                    for (int r = 0; r < 8; r++) g00[i * 8 + r] = kk[i][r * kernel_w];
                }
                g00 += 32;
            }
        }
        for (; p + 3 < inh; p += 4)
        {
            for (int k = 0; k < kernel_w; k++)
            {
                const float* kk[4];
                for (int i = 0; i < 4; i++) kk[i] = kptr[i] + p * kernel_w + k;
                for (int i = 0; i < 4; i++)
                {
                    for (int r = 0; r < 4; r++) g00[i * 4 + r] = kk[i][r * kernel_w];
                }
                g00 += 16;
            }
        }
        for (; p + 1 < inh; p += 2)
        {
            for (int k = 0; k < kernel_w; k++)
            {
                const float* kk[4];
                for (int i = 0; i < 4; i++) kk[i] = kptr[i] + p * kernel_w + k;
                for (int i = 0; i < 4; i++)
                {
                    for (int r = 0; r < 2; r++) g00[i * 2 + r] = kk[i][r * kernel_w];
                }
                g00 += 8;
            }
        }
        for (; p < inh; p++)
        {
            for (int k = 0; k < kernel_w; k++)
            {
                for (int i = 0; i < 4; i++) g00[i] = kptr[i][p * kernel_w + k];
                g00 += 4;
            }
        }
    }
    for (; q + 1 < outh; q += 2)
    {
        const float* kptr0 = (const float*)kernel + q * inh * kernel_w;
        const float* kptr1 = (const float*)kernel + (q + 1) * inh * kernel_w;
        float* g00 = kernel_tm.channel(q / 8 + (q % 8) / 4 + (q % 4) / 2);
        int p = 0;
        for (; p + 7 < inh; p += 8)
        {
            for (int k = 0; k < kernel_w; k++)
            {
                const float* k0 = kptr0 + p * kernel_w + k;
                const float* k1 = kptr1 + p * kernel_w + k;
                for (int r = 0; r < 8; r++) { g00[r * 2 + 0] = k0[r * kernel_w]; g00[r * 2 + 1] = k1[r * kernel_w]; }
                g00 += 16;
            }
        }
        for (; p + 3 < inh; p += 4)
        {
            for (int k = 0; k < kernel_w; k++)
            {
                const float* k0 = kptr0 + p * kernel_w + k;
                const float* k1 = kptr1 + p * kernel_w + k;
                for (int r = 0; r < 4; r++) { g00[r * 2 + 0] = k0[r * kernel_w]; g00[r * 2 + 1] = k1[r * kernel_w]; }
                g00 += 8;
            }
        }
        for (; p + 1 < inh; p += 2)
        {
            for (int k = 0; k < kernel_w; k++)
            {
                const float* k0 = kptr0 + p * kernel_w + k;
                const float* k1 = kptr1 + p * kernel_w + k;
                for (int r = 0; r < 2; r++) { g00[r * 2 + 0] = k0[r * kernel_w]; g00[r * 2 + 1] = k1[r * kernel_w]; }
                g00 += 4;
            }
        }
        for (; p < inh; p++)
        {
            for (int k = 0; k < kernel_w; k++) { g00[0] = kptr0[p * kernel_w + k]; g00[1] = kptr1[p * kernel_w + k]; g00 += 2; }
        }
    }
    for (; q < outh; q++)
    {
        const float* kptr = (const float*)kernel + q * inh * kernel_w;
        float* g00 = kernel_tm.channel(q / 8 + (q % 8) / 4 + (q % 4) / 2 + q % 2);
        int p = 0;
        for (; p + 7 < inh; p += 8)
        {
            for (int k = 0; k < kernel_w; k++)
            {
                const float* k0 = kptr + p * kernel_w + k;
                for (int r = 0; r < 8; r++) g00[r] = k0[r * kernel_w];
                g00 += 8;
            }
        }
        for (; p + 3 < inh; p += 4)
        {
            for (int k = 0; k < kernel_w; k++)
            {
                const float* k0 = kptr + p * kernel_w + k;
                for (int r = 0; r < 4; r++) g00[r] = k0[r * kernel_w];
                g00 += 4;
            }
        }
        for (; p + 1 < inh; p += 2)
        {
            for (int k = 0; k < kernel_w; k++)
            {
                const float* k0 = kptr + p * kernel_w + k;
                for (int r = 0; r < 2; r++) g00[r] = k0[r * kernel_w];
                g00 += 2;
            }
        }
        for (; p < inh; p++)
        {
            for (int k = 0; k < kernel_w; k++) { g00[0] = kptr[p * kernel_w + k]; g00 += 1; }
        }
    }
}

static void convolution1d_packed(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_tm, const Mat& bias_data, int kernel_w, int dilation_w, int stride_w, int activation_type, const Mat& activation_params, const Option& opt)
{
    const int elempack = bottom_blob.elempack;
    const int inh = bottom_blob.h * elempack;
    const int outw = top_blob.w;
    const int out_elempack = top_blob.elempack;
    const int outh = top_blob.h * out_elempack;

    const float* bias_ptr = bias_data;

    const int packn = csrr_vlenb() / 4;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outh; p += (packn >= 8 ? 8 : packn >= 4 ? 4 : 1))
    {
        int vec = (packn >= 8 && p + 7 < outh) ? 8 : (packn >= 4 && p + 3 < outh) ? 4 : 1;
        float* outptr = top_blob.row(p / out_elempack);
        for (int j = 0; j < outw; j++)
        {
#if __riscv_vector
            size_t vl = vec == 8 ? __riscv_vsetvl_e32m2(8) : vec == 4 ? __riscv_vsetvl_e32m1(4) : __riscv_vsetvl_e32m1(1);
            vfloat32m2_t _sumv8;
            vfloat32m1_t _sumv4, _sumv1;
            if (vec == 8)
            {
                _sumv8 = __riscv_vfmv_v_f_f32m2(0.f, vl);
                if (bias_ptr) _sumv8 = __riscv_vle32_v_f32m2(bias_ptr + p, vl);
            }
            else if (vec == 4)
            {
                _sumv4 = __riscv_vfmv_v_f_f32m1(0.f, vl);
                if (bias_ptr) _sumv4 = __riscv_vle32_v_f32m1(bias_ptr + p, vl);
            }
            else
            {
                _sumv1 = __riscv_vfmv_v_f_f32m1(0.f, vl);
                if (bias_ptr) _sumv1 = __riscv_vfmv_v_f_f32m1(bias_ptr[p], vl);
            }

            const float* kptr = weight_data_tm.channel(p / vec);

            int q = 0;
            for (; q < inh; )
            {
                const float* r0 = bottom_blob.row(q / elempack) + j * stride_w * elempack;
                int step = elempack; // how many input scalars will be consumed per kstep
                for (int k = 0; k < kernel_w; k++)
                {
                    if (vec == 8)
                    {
                        vfloat32m2_t _w0 = __riscv_vle32_v_f32m2(kptr, vl);
                        vfloat32m2_t _val = __riscv_vfmv_v_f_f32m2(r0[0], vl);
                        _sumv8 = __riscv_vfmacc_vv_f32m2(_sumv8, _w0, _val, vl);
                        kptr += 8;
                        r0 += dilation_w * step;
                    }
                    else if (vec == 4)
                    {
                        vfloat32m1_t _w0 = __riscv_vle32_v_f32m1(kptr, vl);
                        vfloat32m1_t _val = __riscv_vfmv_v_f_f32m1(r0[0], vl);
                        _sumv4 = __riscv_vfmacc_vv_f32m1(_sumv4, _w0, _val, vl);
                        kptr += 4;
                        r0 += dilation_w * step;
                    }
                    else
                    {
                        float w0 = kptr[0];
                        _sumv1 = __riscv_vfmacc_vf_f32m1(_sumv1, w0, r0[0], vl);
                        kptr += 1;
                        r0 += dilation_w * step;
                    }
                }
                q += step;
            }

            // activation and store
            if (vec == 8)
            {
                _sumv8 = activation_ps(_sumv8, activation_type, activation_params, vl);
                if (out_elempack == 8)
                {
                    __riscv_vse32_v_f32m2(outptr, _sumv8, vl);
                    outptr += 8;
                }
                else if (out_elempack == 4)
                {
                    vfloat32m1_t lo = __riscv_vget_v_f32m2_f32m1(_sumv8, 0);
                    vfloat32m1_t hi = __riscv_vget_v_f32m2_f32m1(_sumv8, 1);
                    __riscv_vse32_v_f32m1(outptr, lo, __riscv_vsetvl_e32m1(4));
                    __riscv_vse32_v_f32m1(outptr + top_blob.w * out_elempack, hi, __riscv_vsetvl_e32m1(4));
                    outptr += 4;
                }
                else
                {
                    float tmp[8]; __riscv_vse32_v_f32m2(tmp, _sumv8, vl);
                    for (int i = 0; i < 8; i++) { outptr[i * top_blob.w * out_elempack] = tmp[i]; }
                    outptr += 1;
                }
            }
            else if (vec == 4)
            {
                _sumv4 = activation_ps(_sumv4, activation_type, activation_params, vl);
                if (out_elempack == 4)
                {
                    __riscv_vse32_v_f32m1(outptr, _sumv4, vl);
                    outptr += 4;
                }
                else if (out_elempack == 8)
                {
                    // place into two rows of 8-pack output
                    float tmp[4]; __riscv_vse32_v_f32m1(tmp, _sumv4, vl);
                    for (int i = 0; i < 4; i++) { outptr[i] = tmp[i]; outptr[i + top_blob.w * out_elempack] = 0.f; }
                    outptr += 4;
                }
                else
                {
                    float tmp[4]; __riscv_vse32_v_f32m1(tmp, _sumv4, vl);
                    for (int i = 0; i < 4; i++) { outptr[i * top_blob.w * out_elempack] = tmp[i]; }
                    outptr += 1;
                }
            }
            else
            {
                _sumv1 = activation_ps(_sumv1, activation_type, activation_params, vl);
                float tmp; __riscv_vse32_v_f32m1(&tmp, _sumv1, vl);
                outptr[0] = tmp;
                outptr += 1;
            }
#else
            float sum[8] = {0};
            if (bias_ptr) for (int i = 0; i < vec; i++) sum[i] = bias_ptr[p + i];
            const float* kptr = weight_data_tm.channel(p / vec);
            int q = 0;
            for (; q < inh; )
            {
                const float* r0 = bottom_blob.row(q / elempack) + j * stride_w * elempack;
                int step = elempack;
                for (int k = 0; k < kernel_w; k++)
                {
                    for (int i = 0; i < vec; i++) sum[i] += r0[0] * kptr[i];
                    kptr += vec;
                    r0 += dilation_w * step;
                }
                q += step;
            }
            for (int i = 0; i < vec; i++)
            {
                sum[i] = activation_ss(sum[i], activation_type, activation_params);
                outptr[i * top_blob.w * out_elempack] = sum[i];
            }
            outptr += 1;
#endif
        }
    }
}

Convolution1D_riscv::Convolution1D_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif
}

int Convolution1D_riscv::create_pipeline(const Option& opt)
{
    if (dynamic_weight)
        return 0;

    int num_input = weight_data_size / kernel_w / num_output;
    convolution1d_transform_kernel_packed(weight_data, weight_data_packed, num_input, num_output, kernel_w);

    if (opt.lightmode) weight_data.release();
    return 0;
}

int Convolution1D_riscv::destroy_pipeline(const Option&)
{
    return 0;
}

int Convolution1D_riscv::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
#if __riscv_vector
    int w = bottom_blob.w;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;

    Mat bottom_blob_bordered;
    make_padding(bottom_blob, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty()) return -100;

    w = bottom_blob_bordered.w;

    int out_elempack = 1;
    if (opt.use_packing_layout)
    {
        const int packn = csrr_vlenb() / 4;
        if (packn >= 8)
            out_elempack = num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;
        else if (packn >= 4)
            out_elempack = num_output % 4 == 0 ? 4 : 1;
        else
            out_elempack = 1;
    }
    size_t out_elemsize = elemsize / elempack * out_elempack;

    const int outw = (w - kernel_extent_w) / stride_w + 1;
    const int outh = num_output / out_elempack;

    top_blob.create(outw, outh, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty()) return -100;

    convolution1d_packed(bottom_blob_bordered, top_blob, weight_data_packed, bias_data, kernel_w, dilation_w, stride_w, activation_type, activation_params, opt);
    return 0;
#else
    // fallback to base implementation when rvv not available
    return Convolution1D::forward(bottom_blob, top_blob, opt);
#endif
}

int Convolution1D_riscv::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& _weight_data = bottom_blobs[1];
    Mat& top_blob = top_blobs[0];

    const int _kernel_w = _weight_data.w;
    const int _num_output = _weight_data.c * _weight_data.elempack;

    Mat weight_data_flattened;
    flatten(_weight_data, weight_data_flattened, opt);
    if (weight_data_flattened.empty()) return -100;

    // pack1
    weight_data_flattened.w *= weight_data_flattened.elempack;
    weight_data_flattened.elemsize /= weight_data_flattened.elempack;
    weight_data_flattened.elempack = 1;

    Mat bias_data_flattened;
    if (bias_term)
    {
        const Mat& _bias_data = bottom_blobs[2];
        flatten(_bias_data, bias_data_flattened, opt);
        if (bias_data_flattened.empty()) return -100;
        bias_data_flattened.w *= bias_data_flattened.elempack;
        bias_data_flattened.elemsize /= bias_data_flattened.elempack;
        bias_data_flattened.elempack = 1;
    }

    ncnn::Layer* op = ncnn::create_layer_cpu(ncnn::LayerType::Convolution1D);

    ncnn::ParamDict pd;
    pd.set(0, _num_output);
    pd.set(1, _kernel_w);
    pd.set(2, dilation_w);
    pd.set(3, stride_w);
    pd.set(4, pad_left);
    pd.set(15, pad_right);
    pd.set(18, pad_value);
    pd.set(5, bias_term);
    pd.set(6, weight_data_flattened.w);
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    op->load_param(pd);

    ncnn::Mat weights[2];
    weights[0] = weight_data_flattened;
    weights[1] = bias_data_flattened;

    op->load_model(ncnn::ModelBinFromMatArray(weights));
    op->create_pipeline(opt);
    op->forward(bottom_blob, top_blob, opt);
    op->destroy_pipeline(opt);
    delete op;

    return 0;
}

} // namespace ncnn
