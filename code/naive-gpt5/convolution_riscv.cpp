// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2025 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "convolution_riscv.h"

#include "benchmark.h"
#include "cpu.h"
#include "layer_type.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

#include "riscv_activation.h"
#include "riscv_usability.h"

namespace ncnn {

// light-weight kernel transform using packn like x86's convolution_transform_kernel_packed_sse
static void convolution_transform_kernel_packed_rvv(const Mat& weight_data, Mat& weight_data_tm, int num_input, int num_output, int kernel_w, int kernel_h, int elempack, int out_elempack)
{
    const int maxk = kernel_w * kernel_h;

    Mat weight_data_r2 = weight_data.reshape(maxk, num_input, num_output);

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

Convolution_riscv::Convolution_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector
    activation = 0;
}

int Convolution_riscv::create_pipeline(const Option& opt)
{
    if (dynamic_weight)
        return 0;

    activation = create_activation_layer(activation_type, activation_params, opt);
    if (activation)
        activation->create_pipeline(opt);

    int kernel_size = kernel_w * kernel_h;
    int num_input = weight_data_size / kernel_size / num_output;

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

    if (opt.use_packing_layout && (elempack != 1 || out_elempack != 1))
    {
        convolution_transform_kernel_packed_rvv(weight_data, weight_data_tm, num_input, num_output, kernel_w, kernel_h, elempack, out_elempack);
    }

    if (opt.lightmode)
        weight_data.release();

    return 0;
}

int Convolution_riscv::destroy_pipeline(const Option& opt)
{
    if (activation)
    {
        activation->destroy_pipeline(opt);
        delete activation;
        activation = 0;
    }
    return 0;
}

static inline void fused_activation_inplace(Mat& m, int activation_type, const Mat& activation_params, const Option& opt)
{
    if (activation_type == 0) return;
    ncnn::Layer* act = create_activation_layer(activation_type, activation_params, opt);
    act->forward_inplace(m, opt);
    act->destroy_pipeline(opt);
    delete act;
}

// Minimal packed convolution using RVV loads and FMA accumulation
int Convolution_riscv::forward_pack_rvv(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    Mat bottom_blob_bordered;
    make_padding(bottom_blob, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    int w = bottom_blob_bordered.w;
    int h = bottom_blob_bordered.h;
    int elempack = bottom_blob_bordered.elempack;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    int outw = (w - kernel_extent_w) / stride_w + 1;
    int outh = (h - kernel_extent_h) / stride_h + 1;

int out_elempack = 1;
#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
    if (opt.use_packing_layout)
        out_elempack = num_output % packn == 0 ? packn : 1;
    const size_t vl = __riscv_vsetvl_e32m1(packn);
#endif
    size_t out_elemsize = bottom_blob.elemsize / elempack * out_elempack;

    top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    const int inch = bottom_blob.c * elempack;
    const int outch = top_blob.c * out_elempack;
    const int maxk = kernel_w * kernel_h;

    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w * dilation_h - kernel_w * dilation_w;
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2 * elempack;
                p1++;
                p2 += dilation_w;
            }
            p2 += gap;
        }
    }

int nn_outch = outch / out_elempack;
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < nn_outch; p++)
    {
        Mat outp = top_blob.channel(p);
        const float* kptr_base = weight_data_tm.empty() ? (const float*)weight_data + p * maxk * inch : weight_data_tm.channel(p);

        // fill bias or zero
        if (bias_data)
        {
#if __riscv_vector
            if (out_elempack > 1)
            {
                vfloat32m1_t _b = __riscv_vle32_v_f32m1((const float*)bias_data + p * out_elempack, vl);
                outp.fill(_b);
            }
            else
            {
                outp.fill(bias_data[p]);
            }
#else
            outp.fill(bias_data[p]);
#endif
        }
        else
        {
            outp.fill(0.f);
        }

        const float* kptr = kptr_base;

        for (int q = 0; q < inch; q += elempack)
        {
            const Mat img0 = bottom_blob_bordered.channel(q / elempack);

            for (int i = 0; i < outh; i++)
            {
                const float* r0row = img0.row(i * stride_h);
                for (int j = 0; j < outw; j++)
                {
                    const float* r0 = r0row + j * stride_w * elempack;
                    float* outptr = outp.row(i) + j * out_elempack;
#if __riscv_vector
                    vfloat32m1_t _sum = __riscv_vle32_v_f32m1(outptr, vl);
                    const float* kptr_k = kptr;
                    for (int k = 0; k < maxk; k++)
                    {
                        const float* r0s = r0 + space_ofs[k];
                        // accumulate over input pack lanes
                        for (int ii = 0; ii < elempack; ii++)
                        {
                            float val = r0s[ii];
                            vfloat32m1_t _val = __riscv_vfmv_v_f_f32m1(val, vl);
                            vfloat32m1_t _w = __riscv_vle32_v_f32m1(kptr_k + ii * out_elempack, vl);
                            _sum = __riscv_vfmacc_vv_f32m1(_sum, _w, _val, vl);
                        }
                        kptr_k += elempack * out_elempack;
                    }
                    __riscv_vse32_v_f32m1(outptr, _sum, vl);
#else
                    for (int e = 0; e < out_elempack; e++)
                    {
                        float sum = outptr[e];
                        for (int k = 0; k < maxk; k++)
                        {
                            const float* r0s = r0 + space_ofs[k];
                            for (int ii = 0; ii < elempack; ii++)
                            {
                                float val = r0s[ii];
                                float wt = kptr[k * elempack * out_elempack + ii * out_elempack + e];
                                sum += val * wt;
                            }
                        }
                        outptr[e] = sum;
                    }
#endif
                }
            }

            kptr += maxk * elempack * out_elempack;
        }
    }

    if (activation)
    {
        activation->forward_inplace(top_blob, opt);
    }

    return 0;
}

int Convolution_riscv::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
#if __riscv_vector
    if (opt.use_packing_layout && (bottom_blob.elempack > 1))
    {
        return forward_pack_rvv(bottom_blob, top_blob, opt);
    }
#endif
    return Convolution::forward(bottom_blob, top_blob, opt);
}

int Convolution_riscv::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& _weight_data = bottom_blobs[1];
    Mat& top_blob = top_blobs[0];

    const int _kernel_w = _weight_data.w;
    const int _kernel_h = _weight_data.h;
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

        bias_data_flattened.w *= bias_data_flattened.elempack;
        bias_data_flattened.elemsize /= bias_data_flattened.elempack;
        bias_data_flattened.elempack = 1;
    }

    ncnn::Layer* op = ncnn::create_layer_cpu(ncnn::LayerType::Convolution);

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
    pd.set(18, pad_value);
    pd.set(5, bias_term);
    pd.set(6, weight_data_flattened.w);
    pd.set(8, int8_scale_term);
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
