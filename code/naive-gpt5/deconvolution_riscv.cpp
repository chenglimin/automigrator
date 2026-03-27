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

#include "deconvolution_riscv.h"

#include "layer_type.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

#include "riscv_activation.h"
#include "riscv_usability.h"

namespace ncnn {

Deconvolution_riscv::Deconvolution_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector

    activation = 0;
    gemm = 0;
}

int Deconvolution_riscv::create_pipeline(const Option& opt)
{
    if (dynamic_weight)
        return 0;

    activation = create_activation_layer(activation_type, activation_params, opt);

    const int maxk = kernel_w * kernel_h;
    int num_input = weight_data_size / maxk / num_output;

    int elempack = 1;
    int out_elempack = 1;
#if __riscv_vector
    if (opt.use_packing_layout)
    {
        const int packn = csrr_vlenb() / 4;
        elempack = num_input % packn == 0 ? packn : 1;
        out_elempack = num_output % packn == 0 ? packn : 1;
    }
#endif // __riscv_vector

    if (opt.use_sgemm_convolution)
    {
        const int maxk = kernel_w * kernel_h;

        gemm = ncnn::create_layer_cpu(ncnn::LayerType::Gemm);

        ncnn::ParamDict pd;
        pd.set(2, 1);                 // transA
        pd.set(3, 0);                 // transB
        pd.set(4, 1);                 // constantA
        pd.set(5, 0);                 // constantB
        pd.set(6, 1);                 // constantC
        pd.set(7, maxk * num_output); // M = maxk*num_output
        pd.set(8, 0);                 // N = size
        pd.set(9, num_input);         // K = inch
        pd.set(10, -1);               // constant_broadcast_type_C = null
        pd.set(11, 0);                // output_N1M
        pd.set(12, out_elempack);

        gemm->load_param(pd);

        // maxk-inch-outch to pa-maxk-outch/pa-inch
        Mat tmp;
        {
            Mat weight_data_r2 = weight_data.reshape(maxk, num_input, num_output);

            tmp.create(maxk * num_output, num_input);

            for (int p = 0; p < num_input; p += 1)
            {
                float* g00 = tmp.row(p);

                for (int q = 0; q + (out_elempack - 1) < num_output; q += out_elempack)
                {
                    for (int k = 0; k < maxk; k++)
                    {
                        for (int i = 0; i < out_elempack; i++)
                        {
                            const float* k00 = weight_data_r2.channel(q + i).row(p);
                            g00[0] = k00[k];
                            g00++;
                        }
                    }
                }
            }
        }

        ncnn::Mat weights[1];
        weights[0] = tmp;

        gemm->load_model(ModelBinFromMatArray(weights));

        gemm->create_pipeline(opt);
    }
    else
    {
        Mat weight_data_transposed(weight_data.w);
        {
            float* pt = weight_data_transposed;
            const float* p = weight_data;

            for (int i = 0; i < num_input * num_output; i++)
            {
                for (int k = 0; k < maxk; k++)
                {
                    pt[maxk - 1 - k] = p[k];
                }

                p += maxk;
                pt += maxk;
            }
        }

        // src = kw-kh-inch-outch
        // dst = pb-pa-kw-kh-inch/pa-outch/pb
        Mat weight_data_r2 = weight_data_transposed.reshape(maxk, num_input, num_output);

        weight_data_tm.create(maxk, num_input / elempack, num_output / out_elempack, (size_t)4u * elempack * out_elempack, elempack * out_elempack);

        for (int q = 0; q + (out_elempack - 1) < num_output; q += out_elempack)
        {
            float* g00 = weight_data_tm.channel(q / out_elempack);

            for (int p = 0; p + (elempack - 1) < num_input; p += elempack)
            {
                for (int k = 0; k < maxk; k++)
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

int Deconvolution_riscv::destroy_pipeline(const Option& opt)
{
    if (activation)
    {
        activation->destroy_pipeline(opt);
        delete activation;
        activation = 0;
    }

    if (gemm)
    {
        gemm->destroy_pipeline(opt);
        delete gemm;
        gemm = 0;
    }

    return 0;
}

int Deconvolution_riscv::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    int outw = (w - 1) * stride_w + kernel_extent_w + output_pad_right;
    int outh = (h - 1) * stride_h + kernel_extent_h + output_pad_bottom;
    int out_elempack = 1;
#if __riscv_vector
    if (opt.use_packing_layout)
    {
        const int packn = csrr_vlenb() / 4;
        out_elempack = num_output % packn == 0 ? packn : 1;
    }
#endif // __riscv_vector
    size_t out_elemsize = elemsize / elempack * out_elempack;

    int out_channels = num_output / out_elempack;

    Mat top_blob_bordered;
    if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0 || (output_w > 0 && output_h > 0))
    {
        top_blob_bordered.create(outw, outh, out_channels, out_elemsize, out_elempack, opt.workspace_allocator);
    }
    else
    {
        top_blob_bordered = top_blob;
        top_blob_bordered.create(outw, outh, out_channels, out_elemsize, out_elempack, opt.blob_allocator);
    }
    if (top_blob_bordered.empty())
        return -100;

    const int maxk = kernel_w * kernel_h;

    if (opt.use_sgemm_convolution)
    {
        Mat bottom_blob_2 = bottom_blob;
        {
            bottom_blob_2.w = bottom_blob.w * bottom_blob.h;
            bottom_blob_2.h = 1;
        }
        Mat top_col2im;
        Option opt_b = opt;
        opt_b.blob_allocator = top_blob_bordered.allocator;
        int ret = gemm->forward(bottom_blob_2, top_col2im, opt_b);
        if (ret != 0)
            return ret;

        {
            const int gap = (outw * stride_h - w * stride_w) * out_elempack;

#if __riscv_vector
            const int packn = csrr_vlenb() / 4;
            if (out_elempack == packn)
            {
                const size_t vl = __riscv_vsetvl_e32m1(packn);
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int p = 0; p < out_channels; p++)
                {
                    const float* sptr = top_col2im.row(p * maxk);
                    Mat outm = top_blob_bordered.channel(p);

                    vfloat32m1_t _bias;
                    if (bias_data.empty())
                    {
                        _bias = __riscv_vfmv_v_f_f32m1(0.f, vl);
                    }
                    else
                    {
                        _bias = __riscv_vle32_v_f32m1((const float*)bias_data + p * packn, vl);
                    }

                    // initialize output with bias
                    for (int i = 0; i < outh; i++)
                    {
                        float* ptr = outm.row(i);
                        for (int j = 0; j < outw; j++)
                        {
                            __riscv_vse32_v_f32m1(ptr, _bias, vl);
                            ptr += packn;
                        }
                    }

                    for (int u = 0; u < kernel_h; u++)
                    {
                        for (int v = 0; v < kernel_w; v++)
                        {
                            float* ptr = outm.row(dilation_h * u) + dilation_w * v * packn;

                            for (int i = 0; i < h; i++)
                            {
                                for (int j = 0; j < w; j++)
                                {
                                    vfloat32m1_t _val = __riscv_vle32_v_f32m1(ptr, vl);
                                    vfloat32m1_t _s = __riscv_vle32_v_f32m1(sptr, vl);
                                    _val = __riscv_vfadd_vv_f32m1(_val, _s, vl);
                                    __riscv_vse32_v_f32m1(ptr, _val, vl);

                                    ptr += stride_w * packn;
                                    sptr += packn;
                                }

                                ptr += gap;
                            }
                        }
                    }
                }
            }
#endif // __riscv_vector

            if (out_elempack == 1)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int p = 0; p < out_channels; p++)
                {
                    const float* sptr = top_col2im.row(p * maxk);
                    Mat outm = top_blob_bordered.channel(p);

                    const float bias = bias_data.empty() ? 0.f : bias_data[p];
                    outm.fill(bias);

                    for (int u = 0; u < kernel_h; u++)
                    {
                        for (int v = 0; v < kernel_w; v++)
                        {
                            float* ptr = outm.row(dilation_h * u) + dilation_w * v;

                            for (int i = 0; i < h; i++)
                            {
                                for (int j = 0; j < w; j++)
                                {
                                    ptr[0] += sptr[0];

                                    ptr += stride_w;
                                    sptr += 1;
                                }

                                ptr += gap;
                            }
                        }
                    }
                }
            }
        }

        if (activation)
        {
            activation->forward_inplace(top_blob_bordered, opt);
        }
    }
    else
    {
        if (elempack == 1 && out_elempack == 1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < num_output; p++)
            {
                float* outptr = top_blob_bordered.channel(p);

                const int channels = bottom_blob.c;
                const int outw_ = top_blob_bordered.w;
                const int outh_ = top_blob_bordered.h;

                for (int i = 0; i < outh_; i++)
                {
                    for (int j = 0; j < outw_; j++)
                    {
                        float sum = 0.f;

                        if (bias_term)
                        {
                            sum = bias_data[p];
                        }

                        const float* kptr = (const float*)weight_data_tm.channel(p);

                        for (int q = 0; q < channels; q++)
                        {
                            const Mat m = bottom_blob.channel(q);

                            for (int y = 0; y < kernel_h; y++)
                            {
                                int sys = (i + y * dilation_h - (kernel_extent_h - 1));
                                if (sys < 0 || sys % stride_h != 0)
                                    continue;

                                int sy = sys / stride_h;
                                if (sy >= h)
                                    continue;

                                const float* sptr = m.row(sy);

                                for (int x = 0; x < kernel_w; x++)
                                {
                                    int sxs = (j + x * dilation_w - (kernel_extent_w - 1));
                                    if (sxs < 0 || sxs % stride_w != 0)
                                        continue;

                                    int sx = sxs / stride_w;
                                    if (sx >= w)
                                        continue;

                                    float val = sptr[sx];

                                    int k = y * kernel_w + x;

                                    float wv = kptr[k];

                                    sum += val * wv;
                                }
                            }

                            kptr += maxk;
                        }

                        sum = activation_ss(sum, activation_type, activation_params);

                        outptr[j] = sum;
                    }

                    outptr += outw_;
                }
            }
        }
        else
        {
            // TODO: RVV optimized packed deconvolution kernels can be implemented here
            // Fallback to sgemm path is recommended for packed layouts
            // For now, compute via gemm if available
            if (gemm)
            {
                Mat bottom_blob_2 = bottom_blob;
                bottom_blob_2.w = bottom_blob.w * bottom_blob.h;
                bottom_blob_2.h = 1;
                Mat top_col2im;
                Option opt_b = opt;
                opt_b.blob_allocator = top_blob_bordered.allocator;
                int ret = gemm->forward(bottom_blob_2, top_col2im, opt_b);
                if (ret != 0)
                    return ret;

                const int gap = (outw * stride_h - w * stride_w) * out_elempack;

#if __riscv_vector
                const int packn = csrr_vlenb() / 4;
                if (out_elempack == packn)
                {
                    const size_t vl = __riscv_vsetvl_e32m1(packn);
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int p = 0; p < out_channels; p++)
                    {
                        const float* sptr = top_col2im.row(p * maxk);
                        Mat outm = top_blob_bordered.channel(p);

                        vfloat32m1_t _bias;
                        if (bias_data.empty())
                        {
                            _bias = __riscv_vfmv_v_f_f32m1(0.f, vl);
                        }
                        else
                        {
                            _bias = __riscv_vle32_v_f32m1((const float*)bias_data + p * packn, vl);
                        }

                        for (int i = 0; i < outh; i++)
                        {
                            float* ptr = outm.row(i);
                            for (int j = 0; j < outw; j++)
                            {
                                __riscv_vse32_v_f32m1(ptr, _bias, vl);
                                ptr += packn;
                            }
                        }

                        for (int u = 0; u < kernel_h; u++)
                        {
                            for (int v = 0; v < kernel_w; v++)
                            {
                                float* ptr = outm.row(dilation_h * u) + dilation_w * v * packn;

                                for (int i = 0; i < h; i++)
                                {
                                    for (int j = 0; j < w; j++)
                                    {
                                        vfloat32m1_t _val = __riscv_vle32_v_f32m1(ptr, vl);
                                        vfloat32m1_t _s = __riscv_vle32_v_f32m1(sptr, vl);
                                        _val = __riscv_vfadd_vv_f32m1(_val, _s, vl);
                                        __riscv_vse32_v_f32m1(ptr, _val, vl);

                                        ptr += stride_w * packn;
                                        sptr += packn;
                                    }

                                    ptr += gap;
                                }
                            }
                        }
                    }
                }
#endif // __riscv_vector

                if (out_elempack == 1)
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int p = 0; p < out_channels; p++)
                    {
                        const float* sptr = top_col2im.row(p * maxk);
                        Mat outm = top_blob_bordered.channel(p);

                        const float bias = bias_data.empty() ? 0.f : bias_data[p];
                        outm.fill(bias);

                        for (int u = 0; u < kernel_h; u++)
                        {
                            for (int v = 0; v < kernel_w; v++)
                            {
                                float* ptr = outm.row(dilation_h * u) + dilation_w * v;

                                for (int i = 0; i < h; i++)
                                {
                                    for (int j = 0; j < w; j++)
                                    {
                                        ptr[0] += sptr[0];

                                        ptr += stride_w;
                                        sptr += 1;
                                    }

                                    ptr += gap;
                                }
                            }
                        }
                    }
                }

                if (activation)
                {
                    activation->forward_inplace(top_blob_bordered, opt);
                }
            }
        }
    }

    cut_padding(top_blob_bordered, top_blob, opt);
    if (top_blob.empty())
        return -100;

    return 0;
}

int Deconvolution_riscv::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& _weight_data = bottom_blobs[1];
    Mat& top_blob = top_blobs[0];

    const int _num_input = bottom_blob.c * bottom_blob.elempack;
    const int _kernel_w = _weight_data.w;
    const int _kernel_h = _weight_data.h;
    const int _num_output = _weight_data.d * 1;

    Mat weight_data_flattened;
    flatten(_weight_data, weight_data_flattened, opt);
    if (weight_data_flattened.empty())
        return -100;

    // weight_data_flattened as pack1
    weight_data_flattened.w *= weight_data_flattened.elempack;
    weight_data_flattened.elemsize /= weight_data_flattened.elempack;
    weight_data_flattened.elempack = 1;

    Mat weight_data_transposed;
    {
        weight_data_transposed.create(_kernel_w * _kernel_h * _num_output * _num_input / 1, 4u, opt.workspace_allocator);
        if (weight_data_transposed.empty())
            return -100;

        const int outch_g = _num_output / 1;
        const int inch_g = _num_input / 1;
        const int maxk = _kernel_h * _kernel_w;

        for (int g = 0; g < 1; g++)
        {
            float* wg2 = (float*)weight_data_transposed + g * outch_g * inch_g * maxk;
            const float* wg = (const float*)weight_data_flattened + g * inch_g * outch_g * maxk;
            for (int i = 0; i < outch_g; i++)
            {
                for (int j = 0; j < inch_g; j++)
                {
                    for (int k = 0; k < maxk; k++)
                    {
                        wg2[(i * inch_g + j) * maxk + k] = wg[(j * outch_g + i) * maxk + k];
                    }
                }
            }
        }
    }

    Mat bias_data_flattened;
    if (bias_term)
    {
        const Mat& _bias_data = bottom_blobs[2];
        flatten(_bias_data, bias_data_flattened, opt);
        if (bias_data_flattened.empty())
            return -100;

        bias_data_flattened.w *= bias_data_flattened.elempack;
        bias_data_flattened.elemsize /= bias_data_flattened.elempack;
        bias_data_flattened.elempack = 1;
    }

    ncnn::Layer* op = ncnn::create_layer_cpu(ncnn::LayerType::Deconvolution);

    ncnn::ParamDict pd;
    pd.set(0, _num_output);
    pd.set(1, _kernel_w);
    pd.set(11, _kernel_h);
    pd.set(2, dilation_w);
    pd.set(12, dilation_h);
    pd.set(3, stride_w);
    pd.set(13, stride_h);
    pd.set(4, pad_left);
    pd.set(15, pad_right);
    pd.set(14, pad_top);
    pd.set(16, pad_bottom);
    pd.set(18, output_pad_right);
    pd.set(19, output_pad_bottom);
    pd.set(20, output_w);
    pd.set(21, output_h);
    pd.set(5, bias_term);
    pd.set(6, weight_data_transposed.w);
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    op->load_param(pd);

    ncnn::Mat weights[2];
    weights[0] = weight_data_transposed;
    weights[1] = bias_data_flattened;

    op->load_model(ncnn::ModelBinFromMatArray(weights));

    op->create_pipeline(opt);

    op->forward(bottom_blob, top_blob, opt);

    op->destroy_pipeline(opt);

    delete op;

    return 0;
}

} // namespace ncnn
