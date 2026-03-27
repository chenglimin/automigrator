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
#include "rvv_mathfun.h"
#include "riscv_usability.h"
#include "riscv_activation.h"
#endif // __riscv_vector

#include "cpu.h"

namespace ncnn {

LSTM_riscv::LSTM_riscv()
{
    one_blob_only = false;
    support_inplace = false;
}

int LSTM_riscv::create_pipeline(const Option& opt)
{
#if NCNN_INT8
    if (int8_scale_term)
    {
        // int8 path not implemented for riscv in this migration
        // fallback to base class behavior if needed
        // return create_pipeline_int8(opt);
    }
#endif

    // pack IFOG in the same way as ARM implementation
    int num_directions = direction == 2 ? 2 : 1;
    int size = weight_data_size / num_directions / hidden_size / 4;

    weight_xc_data_packed.create(size, hidden_size, num_directions, 16u, 4);
    bias_c_data_packed.create(hidden_size, 1, num_directions, 16u, 4);
    weight_hc_data_packed.create(num_output, hidden_size, num_directions, 16u, 4);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int dr = 0; dr < num_directions; dr++)
    {
        const Mat weight_xc = weight_xc_data.channel(dr);
        const Mat bias_c = bias_c_data.channel(dr);
        const Mat weight_hc = weight_hc_data.channel(dr);

        Mat weight_xc_data_packed_dr = weight_xc_data_packed.channel(dr);
        Mat bias_c_data_packed_dr = bias_c_data_packed.channel(dr);
        Mat weight_hc_data_packed_dr = weight_hc_data_packed.channel(dr);

        const float* bias_c_I = bias_c.row(0);
        const float* bias_c_F = bias_c.row(1);
        const float* bias_c_O = bias_c.row(2);
        const float* bias_c_G = bias_c.row(3);

        float* bias_c_IFOG = bias_c_data_packed_dr.row(0);

        for (int q = 0; q < hidden_size; q++)
        {
            bias_c_IFOG[0] = bias_c_I[q];
            bias_c_IFOG[1] = bias_c_F[q];
            bias_c_IFOG[2] = bias_c_O[q];
            bias_c_IFOG[3] = bias_c_G[q];

            bias_c_IFOG += 4;

            const float* weight_xc_I = weight_xc.row(hidden_size * 0 + q);
            const float* weight_xc_F = weight_xc.row(hidden_size * 1 + q);
            const float* weight_xc_O = weight_xc.row(hidden_size * 2 + q);
            const float* weight_xc_G = weight_xc.row(hidden_size * 3 + q);

            const float* weight_hc_I = weight_hc.row(hidden_size * 0 + q);
            const float* weight_hc_F = weight_hc.row(hidden_size * 1 + q);
            const float* weight_hc_O = weight_hc.row(hidden_size * 2 + q);
            const float* weight_hc_G = weight_hc.row(hidden_size * 3 + q);

            float* weight_xc_IFOG = weight_xc_data_packed_dr.row(q);
            float* weight_hc_IFOG = weight_hc_data_packed_dr.row(q);

            for (int i = 0; i < size; i++)
            {
                weight_xc_IFOG[0] = weight_xc_I[i];
                weight_xc_IFOG[1] = weight_xc_F[i];
                weight_xc_IFOG[2] = weight_xc_O[i];
                weight_xc_IFOG[3] = weight_xc_G[i];

                weight_xc_IFOG += 4;
            }

            for (int i = 0; i < num_output; i++)
            {
                weight_hc_IFOG[0] = weight_hc_I[i];
                weight_hc_IFOG[1] = weight_hc_F[i];
                weight_hc_IFOG[2] = weight_hc_O[i];
                weight_hc_IFOG[3] = weight_hc_G[i];

                weight_hc_IFOG += 4;
            }
        }
    }

    if (opt.lightmode && !int8_scale_term)
    {
        weight_xc_data.release();
        bias_c_data.release();
        weight_hc_data.release();
    }

    return 0;
}

static int lstm(const Mat& bottom_blob, Mat& top_blob, int reverse, const Mat& weight_xc, const Mat& bias_c, const Mat& weight_hc, const Mat& weight_hr, Mat& hidden_state, Mat& cell_state, const Option& opt)
{
    int size = bottom_blob.w;
    int T = bottom_blob.h;

    int num_output = top_blob.w;
    int hidden_size = cell_state.w;

    // 4 x hidden_size
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

    // unroll
    for (int t = 0; t < T; t++)
    {
        int ti = reverse ? T - 1 - t : t;

#if __riscv_vector
        const float* x = bottom_blob.row(ti);
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < hidden_size; q++)
        {
            const float* bias_c_IFOG = (const float*)bias_c + q * 4;

            const float* weight_xc_IFOG = weight_xc.row(q);
            const float* weight_hc_IFOG = weight_hc.row(q);

            size_t vl = __riscv_vsetvl_e32m1(4);
            vfloat32m1_t _IFOG = __riscv_vle32_v_f32m1(bias_c_IFOG, vl);
            vfloat32m1_t _sum1 = __riscv_vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t _sum2 = __riscv_vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t _sum3 = __riscv_vfmv_v_f_f32m1(0.f, vl);

            int i = 0;
            for (; i + 3 < size; i += 4)
            {
                float xi0 = x[i];
                float xi1 = x[i + 1];
                float xi2 = x[i + 2];
                float xi3 = x[i + 3];

                vfloat32m1_t _w0 = __riscv_vle32_v_f32m1(weight_xc_IFOG, vl);
                vfloat32m1_t _w1 = __riscv_vle32_v_f32m1(weight_xc_IFOG + 4, vl);
                vfloat32m1_t _w2 = __riscv_vle32_v_f32m1(weight_xc_IFOG + 8, vl);
                vfloat32m1_t _w3 = __riscv_vle32_v_f32m1(weight_xc_IFOG + 12, vl);

                _IFOG = __riscv_vfmacc_vf_f32m1(_IFOG, xi0, _w0, vl);
                _sum1 = __riscv_vfmacc_vf_f32m1(_sum1, xi1, _w1, vl);
                _sum2 = __riscv_vfmacc_vf_f32m1(_sum2, xi2, _w2, vl);
                _sum3 = __riscv_vfmacc_vf_f32m1(_sum3, xi3, _w3, vl);

                weight_xc_IFOG += 16;
            }
            for (; i < size; i++)
            {
                float xi = x[i];
                vfloat32m1_t _w = __riscv_vle32_v_f32m1(weight_xc_IFOG, vl);
                _IFOG = __riscv_vfmacc_vf_f32m1(_IFOG, xi, _w, vl);
                weight_xc_IFOG += 4;
            }

            const float* hidden_ptr = hidden_state;
            i = 0;
            for (; i + 3 < num_output; i += 4)
            {
                float h0 = hidden_ptr[i];
                float h1 = hidden_ptr[i + 1];
                float h2 = hidden_ptr[i + 2];
                float h3 = hidden_ptr[i + 3];

                vfloat32m1_t _w0 = __riscv_vle32_v_f32m1(weight_hc_IFOG, vl);
                vfloat32m1_t _w1 = __riscv_vle32_v_f32m1(weight_hc_IFOG + 4, vl);
                vfloat32m1_t _w2 = __riscv_vle32_v_f32m1(weight_hc_IFOG + 8, vl);
                vfloat32m1_t _w3 = __riscv_vle32_v_f32m1(weight_hc_IFOG + 12, vl);

                _IFOG = __riscv_vfmacc_vf_f32m1(_IFOG, h0, _w0, vl);
                _sum1 = __riscv_vfmacc_vf_f32m1(_sum1, h1, _w1, vl);
                _sum2 = __riscv_vfmacc_vf_f32m1(_sum2, h2, _w2, vl);
                _sum3 = __riscv_vfmacc_vf_f32m1(_sum3, h3, _w3, vl);

                weight_hc_IFOG += 16;
            }
            for (; i < num_output; i++)
            {
                float h = hidden_ptr[i];
                vfloat32m1_t _w = __riscv_vle32_v_f32m1(weight_hc_IFOG, vl);
                _IFOG = __riscv_vfmacc_vf_f32m1(_IFOG, h, _w, vl);
                weight_hc_IFOG += 4;
            }

            _IFOG = __riscv_vfadd_vv_f32m1(_IFOG, _sum1, vl);
            _IFOG = __riscv_vfadd_vv_f32m1(_IFOG, _sum2, vl);
            _IFOG = __riscv_vfadd_vv_f32m1(_IFOG, _sum3, vl);

            float* gates_data = gates.row(q);
            __riscv_vse32_v_f32m1(gates_data, _IFOG, vl);
        }
#else
        const float* x = bottom_blob.row(ti);
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < hidden_size; q++)
        {
            const float* bias_c_I = bias_c.row(0);
            const float* bias_c_F = bias_c.row(1);
            const float* bias_c_O = bias_c.row(2);
            const float* bias_c_G = bias_c.row(3);

            float* gates_data = gates.row(q);

            const float* weight_xc_I = weight_xc.row(hidden_size * 0 + q);
            const float* weight_xc_F = weight_xc.row(hidden_size * 1 + q);
            const float* weight_xc_O = weight_xc.row(hidden_size * 2 + q);
            const float* weight_xc_G = weight_xc.row(hidden_size * 3 + q);

            const float* weight_hc_I = weight_hc.row(hidden_size * 0 + q);
            const float* weight_hc_F = weight_hc.row(hidden_size * 1 + q);
            const float* weight_hc_O = weight_hc.row(hidden_size * 2 + q);
            const float* weight_hc_G = weight_hc.row(hidden_size * 3 + q);

            float I = bias_c_I[q];
            float F = bias_c_F[q];
            float O = bias_c_O[q];
            float G = bias_c_G[q];

            for (int i = 0; i < size; i++)
            {
                float xi = x[i];
                I += weight_xc_I[i] * xi;
                F += weight_xc_F[i] * xi;
                O += weight_xc_O[i] * xi;
                G += weight_xc_G[i] * xi;
            }

            for (int i = 0; i < num_output; i++)
            {
                float h_cont = hidden_state[i];
                I += weight_hc_I[i] * h_cont;
                F += weight_hc_F[i] * h_cont;
                O += weight_hc_O[i] * h_cont;
                G += weight_hc_G[i] * h_cont;
            }

            gates_data[0] = I;
            gates_data[1] = F;
            gates_data[2] = O;
            gates_data[3] = G;
        }
#endif // __riscv_vector

        // lstm unit
        float* output_data = top_blob.row(ti);

        float* cell_ptr = cell_state;
        float* hidden_ptr = hidden_state;
        float* tmp_hidden_ptr = tmp_hidden_state;

        int remain_hidden_size_start = 0;
#if __riscv_vector
        int nn_hidden_size = hidden_size >> 2;
        remain_hidden_size_start = nn_hidden_size << 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int qq = 0; qq < nn_hidden_size; qq++)
        {
            int q = qq * 4;

            const float* gates_q0 = gates.row(q);
            const float* gates_q1 = gates.row(q + 1);
            const float* gates_q2 = gates.row(q + 2);
            const float* gates_q3 = gates.row(q + 3);

            size_t vl = __riscv_vsetvl_e32m1(4);
            vfloat32m1_t _r0 = __riscv_vle32_v_f32m1(gates_q0, vl);
            vfloat32m1_t _r1 = __riscv_vle32_v_f32m1(gates_q1, vl);
            vfloat32m1_t _r2 = __riscv_vle32_v_f32m1(gates_q2, vl);
            vfloat32m1_t _r3 = __riscv_vle32_v_f32m1(gates_q3, vl);

            transpose4x4_ps(_r0, _r1, _r2, _r3, vl);

            vfloat32m1_t _lstm_I = sigmoid_ps(_r0, vl);
            vfloat32m1_t _lstm_F = sigmoid_ps(_r1, vl);
            vfloat32m1_t _lstm_O = sigmoid_ps(_r2, vl);
            vfloat32m1_t _lstm_G = tanh_ps(_r3, vl);

            vfloat32m1_t _cell_prev = __riscv_vle32_v_f32m1(cell_ptr + q, vl);
            vfloat32m1_t _cell2 = __riscv_vfadd_vv_f32m1(__riscv_vfmul_vv_f32m1(_lstm_F, _cell_prev, vl), __riscv_vfmul_vv_f32m1(_lstm_I, _lstm_G, vl), vl);
            vfloat32m1_t _lstm_H = __riscv_vfmul_vv_f32m1(_lstm_O, tanh_ps(_cell2, vl), vl);

            __riscv_vse32_v_f32m1(cell_ptr + q, _cell2, vl);

            if (num_output == hidden_size)
            {
                __riscv_vse32_v_f32m1(hidden_ptr + q, _lstm_H, vl);
                __riscv_vse32_v_f32m1(output_data + q, _lstm_H, vl);
            }
            else
            {
                __riscv_vse32_v_f32m1(tmp_hidden_ptr + q, _lstm_H, vl);
            }
        }
#endif // __riscv_vector
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = remain_hidden_size_start; q < hidden_size; q++)
        {
            const float* gates_data = gates.row(q);

            float I = gates_data[0];
            float F = gates_data[1];
            float O = gates_data[2];
            float G = gates_data[3];

            I = 1.f / (1.f + expf(-I));
            F = 1.f / (1.f + expf(-F));
            O = 1.f / (1.f + expf(-O));
            G = tanhf(G);

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
            int remain_num_output_start = 0;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = remain_num_output_start; q < num_output; q++)
            {
                const float* hr = weight_hr.row(q);
                const float* th = tmp_hidden_state;

                float H = 0;
                for (int i = 0; i < hidden_size; i++)
                {
                    H += th[i] * hr[i];
                }

                hidden_ptr[q] = H;
                output_data[q] = H;
            }
        }
    }

    return 0;
}

int LSTM_riscv::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
#if NCNN_INT8
    if (int8_scale_term)
    {
        // int8 path not implemented for riscv in this migration
        // return forward_int8(bottom_blob, top_blob, opt);
    }
#endif

    int T = bottom_blob.h;
    int num_directions = direction == 2 ? 2 : 1;

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
        int ret = lstm(bottom_blob, top_blob, direction, weight_xc_data_packed.channel(0), bias_c_data_packed.channel(0), weight_hc_data_packed.channel(0), num_output == hidden_size ? Mat() : weight_hr_data.channel(0), hidden, cell, opt);
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

        {
            int ret = lstm(bottom_blob, top_blob_forward, 0, weight_xc_data_packed.channel(0), bias_c_data_packed.channel(0), weight_hc_data_packed.channel(0), num_output == hidden_size ? Mat() : weight_hr_data.channel(0), hidden, cell, opt);
            if (ret != 0)
                return ret;
        }

        hidden.fill(0.0f);
        cell.fill(0.0f);

        {
            int ret = lstm(bottom_blob, top_blob_reverse, 1, weight_xc_data_packed.channel(1), bias_c_data_packed.channel(1), weight_hc_data_packed.channel(1), num_output == hidden_size ? Mat() : weight_hr_data.channel(1), hidden, cell, opt);
            if (ret != 0)
                return ret;
        }

        for (int i = 0; i < T; i++)
        {
            const float* pf = top_blob_forward.row(i);
            const float* pr = top_blob_reverse.row(i);
            float* ptr = top_blob.row(i);

            memcpy(ptr, pf, num_output * sizeof(float));
            memcpy(ptr + num_output, pr, num_output * sizeof(float));
        }
    }

    return 0;
}

int LSTM_riscv::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
#if NCNN_INT8
    if (int8_scale_term)
    {
        // int8 path not implemented for riscv in this migration
        // return forward_int8(bottom_blobs, top_blobs, opt);
    }
#endif

    const Mat& bottom_blob = bottom_blobs[0];
    int T = bottom_blob.h;
    int num_directions = direction == 2 ? 2 : 1;

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
        int ret = lstm(bottom_blob, top_blob, direction, weight_xc_data_packed.channel(0), bias_c_data_packed.channel(0), weight_hc_data_packed.channel(0), num_output == hidden_size ? Mat() : weight_hr_data.channel(0), hidden, cell, opt);
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
        {
            int ret = lstm(bottom_blob, top_blob_forward, 0, weight_xc_data_packed.channel(0), bias_c_data_packed.channel(0), weight_hc_data_packed.channel(0), num_output == hidden_size ? Mat() : weight_hr_data.channel(0), hidden0, cell0, opt);
            if (ret != 0)
                return ret;
        }

        Mat hidden1 = hidden.row_range(1, 1);
        Mat cell1 = cell.row_range(1, 1);
        {
            int ret = lstm(bottom_blob, top_blob_reverse, 1, weight_xc_data_packed.channel(1), bias_c_data_packed.channel(1), weight_hc_data_packed.channel(1), num_output == hidden_size ? Mat() : weight_hr_data.channel(1), hidden1, cell1, opt);
            if (ret != 0)
                return ret;
        }

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
}

} // namespace ncnn
