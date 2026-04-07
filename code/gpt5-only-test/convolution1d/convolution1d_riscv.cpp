// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
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

namespace ncnn {

Convolution1D_riscv::Convolution1D_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector
}

int Convolution1D_riscv::create_pipeline(const Option& opt)
{
    if (dynamic_weight)
        return 0;

    const int num_input = weight_data_size / kernel_w / num_output;

    int elempack = 1;
    int out_elempack = 1;
#if __riscv_vector
    if (opt.use_packing_layout)
    {
        const int packn = csrr_vlenb() / 4;
        elempack = num_input % packn == 0 ? packn : 1;
        out_elempack = num_output % packn == 0 ? packn : 1;
    }
#endif

    // src = kw-inch-outch
    // dst = pb-pa-kw-inch/pa-outch/pb
    {
        Mat weight_data_r2 = weight_data.reshape(kernel_w, num_input, num_output);

        weight_data_packed.create(kernel_w, num_input / elempack, num_output / out_elempack, (size_t)4u * elempack * out_elempack, elempack * out_elempack);

        for (int q = 0; q + (out_elempack - 1) < num_output; q += out_elempack)
        {
            float* g00 = weight_data_packed.channel(q / out_elempack);

            for (int p = 0; p + (elempack - 1) < num_input; p += elempack)
            {
                for (int k = 0; k < kernel_w; k++)
                {
                    for (int i = 0; i < elempack; i++)
                    {
                        for (int j = 0; j < out_elempack; j++)
                        {
                            const float* k00 = weight_data_r2.channel(q + j).row(p + i);

                            g00[0] = k00[k];

                            g00++;
                        }
                    }
                }
            }
        }
    }

    if (opt.lightmode)
        weight_data.release();

    return 0;
}

int Convolution1D_riscv::destroy_pipeline(const Option& /*opt*/)
{
    return 0;
}

int Convolution1D_riscv::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;

    Mat bottom_blob_bordered;
    make_padding(bottom_blob, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    w = bottom_blob_bordered.w;
    h = bottom_blob_bordered.h;

    int out_elempack = 1;
#if __riscv_vector
    if (opt.use_packing_layout)
    {
        const int packn = csrr_vlenb() / 4;
        out_elempack = num_output % packn == 0 ? packn : 1;
    }
#endif
    size_t out_elemsize = elemsize / elempack * out_elempack;

    const int outw = (w - kernel_extent_w) / stride_w + 1;
    const int outh = num_output / out_elempack;

    top_blob.create(outw, outh, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

#if __riscv_vector
    // packn x packn
    if (elempack != 1 && out_elempack != 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outh; p++)
        {
            float* outptr = top_blob.row(p);
            for (int j = 0; j < outw; j++)
            {
                size_t vl = elempack; // same as packn
                vfloat32m1_t _sum = __riscv_vfmv_v_f_f32m1(0.f, vl);
                if (bias_term)
                {
                    _sum = __riscv_vle32_v_f32m1((const float*)bias_data + p * out_elempack, vl);
                }
                const float* kptr = weight_data_packed.channel(p);
                for (int q = 0; q < h; q++)
                {
                    const float* sptr = bottom_blob_bordered.row(q) + j * stride_w * elempack;
                    for (int k = 0; k < kernel_w; k++)
                    {
                        vfloat32m1_t _val = __riscv_vle32_v_f32m1(sptr, vl);
                        vfloat32m1_t _w = __riscv_vle32_v_f32m1(kptr, vl);
                        _sum = __riscv_vfmacc_vv_f32m1(_sum, _val, _w, vl);
                        sptr += dilation_w * elempack;
                        kptr += elempack * out_elempack;
                    }
                }
                _sum = activation_ps(_sum, activation_type, activation_params, vl);
                __riscv_vse32_v_f32m1(outptr, _sum, vl);
                outptr += out_elempack;
            }
        }
        return 0;
    }

    // pack1 to packn
    if (elempack == 1 && out_elempack != 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outh; p++)
        {
            float* outptr = top_blob.row(p);
            for (int j = 0; j < outw; j++)
            {
                size_t vl = out_elempack;
                vfloat32m1_t _sum = __riscv_vfmv_v_f_f32m1(0.f, vl);
                if (bias_term)
                {
                    _sum = __riscv_vle32_v_f32m1((const float*)bias_data + p * out_elempack, vl);
                }
                const float* kptr = weight_data_packed.channel(p);
                for (int q = 0; q < h; q++)
                {
                    const float* sptr = bottom_blob_bordered.row(q) + j * stride_w;
                    for (int k = 0; k < kernel_w; k++)
                    {
                        vfloat32m1_t _val = __riscv_vfmv_v_f_f32m1(sptr[0], vl);
                        vfloat32m1_t _w = __riscv_vle32_v_f32m1(kptr, vl);
                        _sum = __riscv_vfmacc_vv_f32m1(_sum, _val, _w, vl);
                        sptr += dilation_w;
                        kptr += out_elempack;
                    }
                }
                _sum = activation_ps(_sum, activation_type, activation_params, vl);
                __riscv_vse32_v_f32m1(outptr, _sum, vl);
                outptr += out_elempack;
            }
        }
        return 0;
    }

    // packn to pack1
    if (elempack != 1 && out_elempack == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outh; p++)
        {
            float* outptr = top_blob.row(p);
            for (int j = 0; j < outw; j++)
            {
                float sum = 0.f;
                if (bias_term)
                {
                    sum = bias_data[p];
                }
                const float* kptr = weight_data_packed.channel(p);
                for (int q = 0; q < h; q++)
                {
                    const float* sptr = bottom_blob_bordered.row(q) + j * stride_w * elempack;
                    for (int k = 0; k < kernel_w; k++)
                    {
                        size_t vl = elempack;
                        vfloat32m1_t _val = __riscv_vle32_v_f32m1(sptr, vl);
                        vfloat32m1_t _w = __riscv_vle32_v_f32m1(kptr, vl);
                        vfloat32m1_t _mul = __riscv_vfmul_vv_f32m1(_val, _w, vl);
                        vfloat32m1_t _zero = __riscv_vfmv_v_f_f32m1(0.f, vl);
                        vfloat32m1_t _rsum = __riscv_vfredsum_vs_f32m1_f32m1(_zero, _mul, _zero, vl);
                        sum += __riscv_vfmv_f_s_f32m1_f32(_rsum);
                        sptr += dilation_w * elempack;
                        kptr += elempack * out_elempack;
                    }
                }
                sum = activation_ss(sum, activation_type, activation_params);
                outptr[j] = sum;
            }
        }
        return 0;
    }
#endif // __riscv_vector

    // pack1 to pack1 fallback
    if (elempack == 1 && out_elempack == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outh; p++)
        {
            float* outptr = top_blob.row(p);
            for (int j = 0; j < outw; j++)
            {
                float sum = 0.f;
                if (bias_term)
                {
                    sum = bias_data[p];
                }
                const float* kptr = weight_data_packed.channel(p);
                for (int q = 0; q < h; q++)
                {
                    const float* sptr = bottom_blob_bordered.row(q) + j * stride_w;
                    for (int k = 0; k < kernel_w; k++)
                    {
                        float val = sptr[0];
                        float wt = kptr[0];
                        sum += val * wt;
                        sptr += dilation_w;
                        kptr += 1;
                    }
                }
                sum = activation_ss(sum, activation_type, activation_params);
                outptr[j] = sum;
            }
        }
        return 0;
    }

    return 0;
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

#if NCNN_RVV
    if (opt.use_fp16_storage && cpu_support_riscv_zvfh() && weight_data_flattened.elembits() == 16)
    {
        Mat weight_data_flattened_fp32;
        cast_float16_to_float32(weight_data_flattened, weight_data_flattened_fp32, opt);
        weight_data_flattened = weight_data_flattened_fp32;
    }
#endif // NCNN_RVV

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

#if NCNN_RVV
        if (opt.use_fp16_storage && cpu_support_riscv_zvfh() && bias_data_flattened.elembits() == 16)
        {
            Mat bias_data_flattened_fp32;
            cast_float16_to_float32(bias_data_flattened, bias_data_flattened_fp32, opt);
            bias_data_flattened = bias_data_flattened_fp32;
        }
#endif // NCNN_RVV

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
