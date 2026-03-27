// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "lstm_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

#include "cpu.h"

namespace ncnn {

#if __riscv_vector
static inline float sigmoid_scalar(float x)
{
    return 1.f / (1.f + expf(-x));
}

// RVV-optimized LSTM core. Vectorizes the dot-products over input/hidden using m8 groups and reduces to scalar.
static int lstm_rvv_core(const Mat& bottom_blob, Mat& top_blob, int reverse,
                         const Mat& weight_xc, const Mat& bias_c,
                         const Mat& weight_hc, const Mat& weight_hr,
                         Mat& hidden_state, Mat& cell_state, const Option& opt)
{
    const int size = bottom_blob.w;
    const int T = bottom_blob.h;

    const int num_output = top_blob.w;
    const int hidden_size = cell_state.w;

    Mat gates(4, hidden_size, 4u, opt.workspace_allocator);
    if (gates.empty())
        return -100;

    Mat tmp_hidden_state;
    if (num_output != hidden_size)
    {
        tmp_hidden_state.create(hidden_size, 4u, opt.workspace_allocator);
        if (tmp_hidden_state.empty())
            return -100;
    }

    for (int t = 0; t < T; t++)
    {
        const int ti = reverse ? (T - 1 - t) : t;
        const float* x = bottom_blob.row(ti);

        // compute I F O G for each hidden unit
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < hidden_size; q++)
        {
            const float* bias_c_I = bias_c.row(0);
            const float* bias_c_F = bias_c.row(1);
            const float* bias_c_O = bias_c.row(2);
            const float* bias_c_G = bias_c.row(3);

            float I = bias_c_I[q];
            float F = bias_c_F[q];
            float O = bias_c_O[q];
            float G = bias_c_G[q];

            const float* weight_xc_I = weight_xc.row(hidden_size * 0 + q);
            const float* weight_xc_F = weight_xc.row(hidden_size * 1 + q);
            const float* weight_xc_O = weight_xc.row(hidden_size * 2 + q);
            const float* weight_xc_G = weight_xc.row(hidden_size * 3 + q);

            int n = size;
            const float* ptr_x = x;
            const float* ptr_wi = weight_xc_I;
            const float* ptr_wf = weight_xc_F;
            const float* ptr_wo = weight_xc_O;
            const float* ptr_wg = weight_xc_G;
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e32m8(n);
                vfloat32m8_t _x = __riscv_vle32_v_f32m8(ptr_x, vl);
                vfloat32m8_t _wi = __riscv_vle32_v_f32m8(ptr_wi, vl);
                vfloat32m8_t _wf = __riscv_vle32_v_f32m8(ptr_wf, vl);
                vfloat32m8_t _wo = __riscv_vle32_v_f32m8(ptr_wo, vl);
                vfloat32m8_t _wg = __riscv_vle32_v_f32m8(ptr_wg, vl);

                vfloat32m8_t _p_i = __riscv_vfmul_vv_f32m8(_wi, _x, vl);
                vfloat32m8_t _p_f = __riscv_vfmul_vv_f32m8(_wf, _x, vl);
                vfloat32m8_t _p_o = __riscv_vfmul_vv_f32m8(_wo, _x, vl);
                vfloat32m8_t _p_g = __riscv_vfmul_vv_f32m8(_wg, _x, vl);

                vfloat32m1_t _acc_i = __riscv_vfmv_s_f_f32m1(I, vl);
                vfloat32m1_t _acc_f = __riscv_vfmv_s_f_f32m1(F, vl);
                vfloat32m1_t _acc_o = __riscv_vfmv_s_f_f32m1(O, vl);
                vfloat32m1_t _acc_g = __riscv_vfmv_s_f_f32m1(G, vl);

                _acc_i = __riscv_vfredusum_vs_f32m8_f32m1(_p_i, _acc_i, vl);
                _acc_f = __riscv_vfredusum_vs_f32m8_f32m1(_p_f, _acc_f, vl);
                _acc_o = __riscv_vfredusum_vs_f32m8_f32m1(_p_o, _acc_o, vl);
                _acc_g = __riscv_vfredusum_vs_f32m8_f32m1(_p_g, _acc_g, vl);

                I = __riscv_vfmv_f_s_f32m1_f32(_acc_i);
                F = __riscv_vfmv_f_s_f32m1_f32(_acc_f);
                O = __riscv_vfmv_f_s_f32m1_f32(_acc_o);
                G = __riscv_vfmv_f_s_f32m1_f32(_acc_g);

                ptr_x += vl;
                ptr_wi += vl;
                ptr_wf += vl;
                ptr_wo += vl;
                ptr_wg += vl;
                n -= (int)vl;
            }

            const float* weight_hc_I = weight_hc.row(hidden_size * 0 + q);
            const float* weight_hc_F = weight_hc.row(hidden_size * 1 + q);
            const float* weight_hc_O = weight_hc.row(hidden_size * 2 + q);
            const float* weight_hc_G = weight_hc.row(hidden_size * 3 + q);

            int n_out = num_output;
            const float* ptr_h = hidden_state;
            const float* ptr_hi = weight_hc_I;
            const float* ptr_hf = weight_hc_F;
            const float* ptr_ho = weight_hc_O;
            const float* ptr_hg = weight_hc_G;
            while (n_out > 0)
            {
                size_t vl = __riscv_vsetvl_e32m8(n_out);
                vfloat32m8_t _h = __riscv_vle32_v_f32m8(ptr_h, vl);
                vfloat32m8_t _hi = __riscv_vle32_v_f32m8(ptr_hi, vl);
                vfloat32m8_t _hf = __riscv_vle32_v_f32m8(ptr_hf, vl);
                vfloat32m8_t _ho = __riscv_vle32_v_f32m8(ptr_ho, vl);
                vfloat32m8_t _hg = __riscv_vle32_v_f32m8(ptr_hg, vl);

                vfloat32m8_t _p_i = __riscv_vfmul_vv_f32m8(_hi, _h, vl);
                vfloat32m8_t _p_f = __riscv_vfmul_vv_f32m8(_hf, _h, vl);
                vfloat32m8_t _p_o = __riscv_vfmul_vv_f32m8(_ho, _h, vl);
                vfloat32m8_t _p_g = __riscv_vfmul_vv_f32m8(_hg, _h, vl);

                vfloat32m1_t _acc_i = __riscv_vfmv_s_f_f32m1(I, vl);
                vfloat32m1_t _acc_f = __riscv_vfmv_s_f_f32m1(F, vl);
                vfloat32m1_t _acc_o = __riscv_vfmv_s_f_f32m1(O, vl);
                vfloat32m1_t _acc_g = __riscv_vfmv_s_f_f32m1(G, vl);

                _acc_i = __riscv_vfredusum_vs_f32m8_f32m1(_p_i, _acc_i, vl);
                _acc_f = __riscv_vfredusum_vs_f32m8_f32m1(_p_f, _acc_f, vl);
                _acc_o = __riscv_vfredusum_vs_f32m8_f32m1(_p_o, _acc_o, vl);
                _acc_g = __riscv_vfredusum_vs_f32m8_f32m1(_p_g, _acc_g, vl);

                I = __riscv_vfmv_f_s_f32m1_f32(_acc_i);
                F = __riscv_vfmv_f_s_f32m1_f32(_acc_f);
                O = __riscv_vfmv_f_s_f32m1_f32(_acc_o);
                G = __riscv_vfmv_f_s_f32m1_f32(_acc_g);

                ptr_h += vl;
                ptr_hi += vl;
                ptr_hf += vl;
                ptr_ho += vl;
                ptr_hg += vl;
                n_out -= (int)vl;
            }

            float* gates_data = gates.row(q);
            gates_data[0] = I;
            gates_data[1] = F;
            gates_data[2] = O;
            gates_data[3] = G;
        }

        float* output_data = top_blob.row(ti);
        float* cell_ptr = cell_state;
        float* hidden_ptr = hidden_state;
        float* tmp_hidden_ptr = tmp_hidden_state;

        // apply activations and update cell/hidden
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < hidden_size; q++)
        {
            const float* g = gates.row(q);
            float I = sigmoid_scalar(g[0]);
            float F = sigmoid_scalar(g[1]);
            float O = sigmoid_scalar(g[2]);
            float G = tanhf(g[3]);

            float cell2 = F * cell_ptr[q] + I * G;
            float H = O * tanhf(cell2);

            cell_ptr[q] = cell2;
            if (num_output == hidden_size)
            {
                hidden_ptr[q] = H;
                output_data[q] = H;
            }
            else
            {
                tmp_hidden_ptr[q] = H;
            }
        }

        if (num_output != hidden_size)
        {
            // project hidden to output
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < num_output; q++)
            {
                const float* hr = weight_hr.row(q);
                float H = 0.f;

                int n = hidden_size;
                const float* ptr_h = tmp_hidden_state;
                const float* ptr_hr = hr;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m8(n);
                    vfloat32m8_t _h = __riscv_vle32_v_f32m8(ptr_h, vl);
                    vfloat32m8_t _hr = __riscv_vle32_v_f32m8(ptr_hr, vl);
                    vfloat32m8_t _p = __riscv_vfmul_vv_f32m8(_h, _hr, vl);
                    vfloat32m1_t _acc = __riscv_vfmv_s_f_f32m1(H, vl);
                    _acc = __riscv_vfredusum_vs_f32m8_f32m1(_p, _acc, vl);
                    H = __riscv_vfmv_f_s_f32m1_f32(_acc);
                    ptr_h += vl;
                    ptr_hr += vl;
                    n -= (int)vl;
                }

                hidden_ptr[q] = H;
                output_data[q] = H;
            }
        }
    }

    return 0;
}
#endif // __riscv_vector

LSTM_riscv::LSTM_riscv()
{
    one_blob_only = false;
    support_inplace = false;
}

int LSTM_riscv::create_pipeline(const Option& opt)
{
    // No special weight transform for RVV path
    return 0;
}

int LSTM_riscv::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
#if NCNN_INT8
    if (int8_scale_term)
    {
        // int8 path not optimized here; fall back to generic implementation
        return LSTM::forward(bottom_blob, top_blob, opt);
    }
#endif

#if __riscv_vector
    const int T = bottom_blob.h;
    const int num_directions = direction == 2 ? 2 : 1;

    Mat hidden(num_output, 4u, opt.workspace_allocator);
    if (hidden.empty())
        return -100;
    hidden.fill(0.f);

    Mat cell(hidden_size, 4u, opt.workspace_allocator);
    if (cell.empty())
        return -100;
    cell.fill(0.f);

    top_blob.create(num_output * num_directions, T, 4u, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (direction == 0 || direction == 1)
    {
        return lstm_rvv_core(bottom_blob, top_blob, direction,
                              weight_xc_data.channel(0), bias_c_data.channel(0),
                              weight_hc_data.channel(0), num_output == hidden_size ? Mat() : weight_hr_data.channel(0),
                              hidden, cell, opt);
    }

    if (direction == 2)
    {
        Mat top_blob_forward(num_output, T, 4u, opt.workspace_allocator);
        if (top_blob_forward.empty())
            return -100;
        Mat top_blob_reverse(num_output, T, 4u, opt.workspace_allocator);
        if (top_blob_reverse.empty())
            return -100;

        int ret0 = lstm_rvv_core(bottom_blob, top_blob_forward, 0,
                                  weight_xc_data.channel(0), bias_c_data.channel(0),
                                  weight_hc_data.channel(0), num_output == hidden_size ? Mat() : weight_hr_data.channel(0),
                                  hidden, cell, opt);
        if (ret0 != 0)
            return ret0;

        hidden.fill(0.f);
        cell.fill(0.f);

        int ret1 = lstm_rvv_core(bottom_blob, top_blob_reverse, 1,
                                  weight_xc_data.channel(1), bias_c_data.channel(1),
                                  weight_hc_data.channel(1), num_output == hidden_size ? Mat() : weight_hr_data.channel(1),
                                  hidden, cell, opt);
        if (ret1 != 0)
            return ret1;

        for (int i = 0; i < T; i++)
        {
            const float* pf = top_blob_forward.row(i);
            const float* pr = top_blob_reverse.row(i);
            float* ptr = top_blob.row(i);
            memcpy(ptr, pf, num_output * sizeof(float));
            memcpy(ptr + num_output, pr, num_output * sizeof(float));
        }
        return 0;
    }
#endif
    return LSTM::forward(bottom_blob, top_blob, opt);
}

int LSTM_riscv::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
#if NCNN_INT8
    if (int8_scale_term)
    {
        return LSTM::forward(bottom_blobs, top_blobs, opt);
    }
#endif

#if __riscv_vector
    const Mat& bottom_blob = bottom_blobs[0];
    const int T = bottom_blob.h;
    const int num_directions = direction == 2 ? 2 : 1;

    Mat hidden;
    Mat cell;
    Allocator* hidden_cell_allocator = top_blobs.size() == 3 ? opt.blob_allocator : opt.workspace_allocator;
    if (bottom_blobs.size() == 3)
    {
        hidden = bottom_blobs[1].clone(hidden_cell_allocator);
        cell = bottom_blobs[2].clone(hidden_cell_allocator);
    }
    else
    {
        hidden.create(num_output, num_directions, 4u, hidden_cell_allocator);
        if (hidden.empty())
            return -100;
        hidden.fill(0.f);

        cell.create(hidden_size, num_directions, 4u, hidden_cell_allocator);
        if (cell.empty())
            return -100;
        cell.fill(0.f);
    }

    Mat& top_blob = top_blobs[0];
    top_blob.create(num_output * num_directions, T, 4u, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (direction == 0 || direction == 1)
    {
        Mat hidden0 = hidden.row_range(0, 1);
        Mat cell0 = cell.row_range(0, 1);
        int ret = lstm_rvv_core(bottom_blob, top_blob, direction,
                                 weight_xc_data.channel(0), bias_c_data.channel(0),
                                 weight_hc_data.channel(0), num_output == hidden_size ? Mat() : weight_hr_data.channel(0),
                                 hidden0, cell0, opt);
        if (ret != 0)
            return ret;
    }

    if (direction == 2)
    {
        Mat top_blob_forward(num_output, T, 4u, opt.workspace_allocator);
        if (top_blob_forward.empty())
            return -100;
        Mat top_blob_reverse(num_output, T, 4u, opt.workspace_allocator);
        if (top_blob_reverse.empty())
            return -100;

        Mat hidden0 = hidden.row_range(0, 1);
        Mat cell0 = cell.row_range(0, 1);
        int ret0 = lstm_rvv_core(bottom_blob, top_blob_forward, 0,
                                  weight_xc_data.channel(0), bias_c_data.channel(0),
                                  weight_hc_data.channel(0), num_output == hidden_size ? Mat() : weight_hr_data.channel(0),
                                  hidden0, cell0, opt);
        if (ret0 != 0)
            return ret0;

        Mat hidden1 = hidden.row_range(1, 1);
        Mat cell1 = cell.row_range(1, 1);
        int ret1 = lstm_rvv_core(bottom_blob, top_blob_reverse, 1,
                                  weight_xc_data.channel(1), bias_c_data.channel(1),
                                  weight_hc_data.channel(1), num_output == hidden_size ? Mat() : weight_hr_data.channel(1),
                                  hidden1, cell1, opt);
        if (ret1 != 0)
            return ret1;

        for (int i = 0; i < T; i++)
        {
            const float* pf = top_blob_forward.row(i);
            const float* pr = top_blob_reverse.row(i);
            float* ptr = top_blob.row(i);
            memcpy(ptr, pf, num_output * sizeof(float));
            memcpy(ptr + num_output, pr, num_output * sizeof(float));
        }
    }

    if (top_blobs.size() == 3)
    {
        top_blobs[1] = hidden;
        top_blobs[2] = cell;
    }

    return 0;
#endif

    return LSTM::forward(bottom_blobs, top_blobs, opt);
}

} // namespace ncnn
