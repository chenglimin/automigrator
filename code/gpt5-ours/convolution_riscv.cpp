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

#include "convolution_riscv.h"

#include "benchmark.h"
#include "cpu.h"
#include "layer_type.h"
#include "fused_activation.h"

#if __riscv_vector
#include <riscv_vector.h>
#include "riscv_usability.h"
#endif // __riscv_vector

namespace ncnn {

Convolution_riscv::Convolution_riscv()
{
    // Disable packing until RVV kernels are fully implemented to avoid mismatched layouts in generic path
    support_packing = false;
    activation = 0;
    nT = 0;
}

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

int Convolution_riscv::create_pipeline(const Option& opt)
{
    if (dynamic_weight)
        return 0;

    // create fused activation helper from header (internal linkage)
    activation = create_activation_layer(activation_type, activation_params, opt);
    nT = opt.num_threads;

    const int maxk = kernel_w * kernel_h;
    const int num_input = weight_data_size / maxk / num_output;

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

#if __riscv_vector
    if (elempack != 1 || out_elempack != 1)
    {
        convolution_transform_kernel_packed_rvv(weight_data, weight_data_tm, num_input, num_output, kernel_w, kernel_h, elempack, out_elempack);
    }
    else
#endif
    {
        // pack1 path uses raw weights or gemm/winograd transformed as generic code does
        weight_data_tm = weight_data;
    }

    // keep original weight_data for generic forward path
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

static inline int compute_out_dims(const Mat& bottom_blob_bordered, int kernel_w, int kernel_h, int stride_w, int stride_h, int dilation_w, int dilation_h, int& outw, int& outh)
{
    const int w = bottom_blob_bordered.w;
    const int h = bottom_blob_bordered.h;
    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;
    outw = (w - kernel_extent_w) / stride_w + 1;
    outh = (h - kernel_extent_h) / stride_h + 1;
    return 0;
}

int Convolution_riscv::forward_pack1_fallback(const Mat& bottom_blob_bordered, Mat& top_blob, const Mat& weight_data_used, const Mat& bias_data_used, int _kernel_w, int _kernel_h, const Option& opt) const
{
    int outw = 0, outh = 0;
    compute_out_dims(bottom_blob_bordered, _kernel_w, _kernel_h, stride_w, stride_h, dilation_w, dilation_h, outw, outh);

    const int inch = bottom_blob_bordered.c;
    const int outch = num_output;
    const int maxk = _kernel_w * _kernel_h;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = bottom_blob_bordered.w * dilation_h - _kernel_w * dilation_w;
        for (int i = 0; i < _kernel_h; i++)
        {
            for (int j = 0; j < _kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2 += dilation_w;
            }
            p2 += gap;
        }
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        float* outptr = top_blob.channel(p);

        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                float sum = 0.f;
                if (!bias_data_used.empty()) sum = bias_data_used[p];

                const float* kptr = (const float*)weight_data_used + maxk * inch * p;

                for (int q = 0; q < inch; q++)
                {
                    const Mat m = bottom_blob_bordered.channel(q);
                    const float* sptr = m.row(i * stride_h) + j * stride_w;

                    for (int k = 0; k < maxk; k++)
                    {
                        sum += sptr[space_ofs[k]] * kptr[k];
                    }

                    kptr += maxk;
                }

                outptr[j] = sum;
            }

            outptr += outw;
        }
    }

    return 0;
}

int Convolution_riscv::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // Ensure pack1 input for generic fallback path
    if (bottom_blob.elempack != 1)
    {
        Mat bottom_blob_pack1;
        Option opt_nopack = opt;
        opt_nopack.use_packing_layout = false;
        convert_packing(bottom_blob, bottom_blob_pack1, 1, opt_nopack);
        return Convolution::forward(bottom_blob_pack1, top_blob, opt);
    }

    return Convolution::forward(bottom_blob, top_blob, opt);
}

int Convolution_riscv::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    // Delegate dynamic-weight path to base implementation for correctness
    return Convolution::forward(bottom_blobs, top_blobs, opt);
}

} // namespace ncnn
