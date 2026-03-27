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
#include "convolution1d_packed.h"

#include "layer_type.h"
#include "cpu.h"

namespace ncnn {

// We reuse the arm/x86 packed kernel transformer logic structure but implement RVV-friendly dynamic packn
// The transformer reorganizes weight layout into [packn-major] to match rvv loads
static void convolution1d_transform_kernel_packed_rvv(const Mat& kernel, Mat& kernel_tm, int inh, int outh, int kernel_w, const Option& opt)
{
    // src layout: kw-inh-outh
    // dst layout: packn-packnblocks-kw-inh/packn-outh/packn
#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
    int out_elempack = 1;
    if (opt.use_packing_layout)
        out_elempack = outh % packn == 0 ? packn : 1;
    if (out_elempack == 1)
    {
        kernel_tm = kernel;
        return;
    }

    // reshape to inh x outh x kw contiguous
    Mat kernel_r = kernel.reshape(kernel_w, inh, outh);
    // create tm as [kw*(inh) * out_blocks, out_elempack]
    kernel_tm.create(kernel_w * inh, outh / out_elempack, (size_t)4u * out_elempack, out_elempack);

    for (int ob = 0; ob < outh / out_elempack; ob++)
    {
        float* g = kernel_tm.row(ob);
        for (int q = 0; q < inh; q++)
        {
            for (int k = 0; k < kernel_w; k++)
            {
                for (int j = 0; j < out_elempack; j++)
                {
                    // read weight[k, q, ob*packn + j]
                    const float* src = kernel_r.channel(ob * out_elempack + j).row(q);
                    *g++ = src[k];
                }
            }
        }
    }
#else
    kernel_tm = kernel;
#endif
}

// keep symbol unused but compiled for future optimization paths
static void convolution1d_packed_rvv(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_tm, const Mat& bias_data, int kernel_w, int dilation_w, int stride_w, int activation_type, const Mat& activation_params, const Option& opt)
{
    const int elempack = bottom_blob.elempack;
    const int inh = bottom_blob.h * elempack;
    const int outw = top_blob.w;
    const int out_elempack = top_blob.elempack;
    const int outh = top_blob.h * out_elempack;

    const int N = bottom_blob.w * elempack;
    // const int M = top_blob.w * out_elempack;

    const float* bias_data_ptr = bias_data;

#if __riscv_vector
    const int packn = csrr_vlenb() / 4;

    int nn_outh = outh / packn;
    int remain_outh_start = 0;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outh; pp++)
    {
        const int p = pp * packn;
        float* outptr = top_blob.row(p / out_elempack);

        for (int j = 0; j < outw; j++)
        {
            size_t vl = __riscv_vsetvl_e32m8(packn);
            vfloat32m8_t _sum = __riscv_vfmv_v_f_f32m8(0.f, vl);
            if (bias_data_ptr)
            {
                _sum = __riscv_vle32_v_f32m8(bias_data_ptr + p, vl);
            }

            const float* kptr = weight_data_tm.channel(p / packn);

            int q = 0;
            // consume inh in blocks of elempack
            for (; q < inh; q++)
            {
                const float* r0 = bottom_blob.row(q / elempack) + j * stride_w * elempack;
                // load input lane according to elempack
                float v;
                if (elempack == 1)
                {
                    v = r0[0];
                    r0 += dilation_w;
                }
                else if (elempack == 2)
                {
                    v = r0[0]; // use first lane, next iteration uses next row element
                    r0 += dilation_w * 2;
                }
                else if (elempack == 4)
                {
                    v = r0[0];
                    r0 += dilation_w * 4;
                }
                else // 8 or 16 not typical for 1d h
                {
                    v = r0[0];
                    r0 += dilation_w * elempack;
                }

                vfloat32m8_t _w = __riscv_vle32_v_f32m8(kptr, vl);
                _sum = __riscv_vfmacc_vf_f32m8(_sum, v, _w, vl);
                kptr += packn;
            }

            _sum = activation_ps(_sum, activation_type, activation_params, vl);
            __riscv_vse32_v_f32m8(outptr, _sum, vl);
            outptr += (out_elempack == packn) ? packn : 1;
        }
    }
    remain_outh_start += nn_outh * packn;

    for (int p = remain_outh_start; p < outh; p++)
    {
        float* outptr = top_blob.row(p);

        for (int j = 0; j < outw; j++)
        {
            float sum = 0.f;
            if (bias_data_ptr) sum = bias_data_ptr[p];

            const float* kptr = weight_data_tm.channel(p / packn + (packn == 1 ? 0 : (p % packn)));

            int q = 0;
            for (; q < inh; q++)
            {
                const float* r0 = bottom_blob.row(q / elempack) + j * stride_w * elempack;
                float v = r0[0];
                sum += v * kptr[0];
                r0 += dilation_w;
                kptr += 1;
            }

            sum = activation_ss(sum, activation_type, activation_params);
            outptr[0] = sum;
            outptr += 1;
        }
    }
#else
    // fallback to generic
    const int bias_term = bias_data.empty() ? 0 : 1;
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outh; p++)
    {
        float* outptr = top_blob.row(p);
        for (int j = 0; j < outw; j++)
        {
            float sum = bias_term ? bias_data_ptr[p] : 0.f;
            const float* kptr = weight_data_tm.channel(p);
            for (int q = 0; q < inh; q++)
            {
                const float* r0 = bottom_blob.row(q / elempack) + j * stride_w * elempack;
                sum += r0[0] * kptr[0];
                r0 += dilation_w;
                kptr += 1;
            }
            sum = activation_ss(sum, activation_type, activation_params);
            outptr[j] = sum;
        }
    }
#endif
}

Convolution1D_riscv::Convolution1D_riscv()
{
#if __riscv_vector
    support_packing = true;
#else
    support_packing = false;
#endif
}

int Convolution1D_riscv::create_pipeline(const Option& opt)
{
    if (dynamic_weight)
        return 0;

#if __riscv_vector
    if (opt.use_packing_layout)
    {
        const int num_input = weight_data_size / kernel_w / num_output;
        convolution1d_transform_kernel_packed(weight_data, weight_data_tm, num_input, num_output, kernel_w);
    }
    else
#endif
    {
        weight_data_tm = weight_data;
    }

    return 0;
}

int Convolution1D_riscv::destroy_pipeline(const Option& /*opt*/)
{
    return 0;
}

int Convolution1D_riscv::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // If we can use RVV packn path, do it here; otherwise, defer to generic Convolution1D
#if __riscv_vector
    if (opt.use_packing_layout)
    {
        const int packn = csrr_vlenb() / 4;
        if (num_output % packn == 0)
        {
            const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;

            Mat bottom_blob_for_pad = bottom_blob;
            if (bottom_blob_for_pad.elempack != 1)
            {
                Mat tmp;
                ncnn::convert_packing(bottom_blob_for_pad, tmp, 1, opt);
                bottom_blob_for_pad = tmp;
            }

            Mat bottom_blob_bordered;
            make_padding(bottom_blob_for_pad, bottom_blob_bordered, opt);
            if (bottom_blob_bordered.empty())
                return -100;

            const int w = bottom_blob_bordered.w;

            const int out_elempack = packn;
            const size_t scalar_elemsize = (bottom_blob_bordered.elemsize / bottom_blob_bordered.elempack);
            const size_t out_elemsize = scalar_elemsize * out_elempack;

            const int outw = (w - kernel_extent_w) / stride_w + 1;
            const int outh = num_output / out_elempack;

            top_blob.create(outw, outh, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            // instrumentation: verify packing and elemsize
            fprintf(stderr, "[RVV-Conv1D] packn=%d out_elemsize=%zu out_elempack=%d bottom(elemsize=%zu,elempack=%d)\n", packn, out_elemsize, out_elempack, bottom_blob_bordered.elemsize, bottom_blob_bordered.elempack);

            convolution1d_packed(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, kernel_w, dilation_w, stride_w, activation_type, activation_params, opt);

            // instrumentation: report actual created top_blob
            fprintf(stderr, "[RVV-Conv1D] top_blob(elemsize=%zu, elempack=%d, w=%d, h=%d)\n", top_blob.elemsize, top_blob.elempack, top_blob.w, top_blob.h);
            return 0;
        }
    }
#endif

    // fallback to generic implementation (handles padding internally)
    if (bottom_blob.elempack != 1)
    {
        Mat bottom_blob_p1;
        ncnn::convert_packing(bottom_blob, bottom_blob_p1, 1, opt);
        return Convolution1D::forward(bottom_blob_p1, top_blob, opt);
    }
    return Convolution1D::forward(bottom_blob, top_blob, opt);
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
    if (weight_data_flattened.empty())
        return -100;

    // weight_data_flattened as pack1
    weight_data_flattened.w *= weight_data_flattened.elempack;
    weight_data_flattened.elemsize /= weight_data_flattened.elempack;
    weight_data_flattened.elempack = 1;

    Mat bias_data_flattened;
    if (bias_term)
    {
        const Mat& _bias_data = bottom_blobs[2];
        flatten(_bias_data, bias_data_flattened, opt);
        if (bias_data_flattened.empty())
            return -100;

        // bias_data_flattened as pack1
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
