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

#include "lrn_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#include "rvv_mathfun.h"
#endif // __riscv_vector

namespace ncnn {

int LRN_riscv::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    size_t elemsize = bottom_top_blob.elemsize;
    int size = w * h;

    // squared values with local_size padding
    Mat square_blob;
    square_blob.create(w, h, channels, elemsize, opt.workspace_allocator);
    if (square_blob.empty())
        return -100;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* ptr = bottom_top_blob.channel(q);
        float* outptr = square_blob.channel(q);

#if __riscv_vector
        int n = size;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(n);
            vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
            vfloat32m8_t _outp = __riscv_vfmul_vv_f32m8(_p, _p, vl);
            __riscv_vse32_v_f32m8(outptr, _outp, vl);
            ptr += vl;
            outptr += vl;
            n -= vl;
        }
#else
        for (int i = 0; i < size; i++)
        {
            *outptr = *ptr * *ptr;
            ptr++;
            outptr++;
        }
#endif // __riscv_vector
    }

    if (region_type == NormRegion_ACROSS_CHANNELS)
    {
        Mat square_sum;
        square_sum.create(w, h, channels, elemsize, opt.workspace_allocator);
        if (square_sum.empty())
            return -100;
        square_sum.fill(0.f);

        const float alpha_div_size = alpha / local_size;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            // square sum
            for (int p = q - local_size / 2; p <= q + local_size / 2; p++)
            {
                if (p < 0 || p >= channels)
                    continue;

                const float* sptr = square_blob.channel(p);
                float* ssptr = square_sum.channel(q);

#if __riscv_vector
                int n = size;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m8(n);
                    vfloat32m8_t _sp = __riscv_vle32_v_f32m8(sptr, vl);
                    vfloat32m8_t _ssp = __riscv_vle32_v_f32m8(ssptr, vl);
                    _ssp = __riscv_vfadd_vv_f32m8(_ssp, _sp, vl);
                    __riscv_vse32_v_f32m8(ssptr, _ssp, vl);
                    sptr += vl;
                    ssptr += vl;
                    n -= vl;
                }
#else
                for (int i = 0; i < size; i++)
                {
                    *ssptr += *sptr;
                    sptr++;
                    ssptr++;
                }
#endif // __riscv_vector
            }

            float* ptr = bottom_top_blob.channel(q);
            float* ssptr = square_sum.channel(q);

#if __riscv_vector
            int n = size;
            const float neg_beta = -beta;
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e32m8(n);
                vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                vfloat32m8_t _ssp = __riscv_vle32_v_f32m8(ssptr, vl);
                _ssp = __riscv_vfmul_vf_f32m8(_ssp, alpha_div_size, vl);
                _ssp = __riscv_vfadd_vf_f32m8(_ssp, bias, vl);
                vfloat32m8_t _pow = pow_ps(_ssp, __riscv_vfmv_v_f_f32m8(neg_beta, vl), vl);
                _p = __riscv_vfmul_vv_f32m8(_p, _pow, vl);
                __riscv_vse32_v_f32m8(ptr, _p, vl);
                ssptr += vl;
                ptr += vl;
                n -= vl;
            }
#else
            for (int i = 0; i < size; i++)
            {
                *ptr = *ptr * powf(bias + alpha_div_size * *ssptr, -beta);
                ssptr++;
                ptr++;
            }
#endif // __riscv_vector
        }
    }
    else if (region_type == NormRegion_WITHIN_CHANNEL)
    {
        int outw = w;
        int outh = h;

        Mat square_blob_bordered = square_blob;
        int pad = local_size / 2;
        if (pad > 0)
        {
            Option opt_b = opt;
            opt_b.blob_allocator = opt.workspace_allocator;
            copy_make_border(square_blob, square_blob_bordered, pad, local_size - pad - 1, pad, local_size - pad - 1, BORDER_CONSTANT, 0.f, opt_b);
            if (square_blob_bordered.empty())
                return -100;

            w = square_blob_bordered.w;
            h = square_blob_bordered.h;
        }

        const int maxk = local_size * local_size;

        const float alpha_div_size = alpha / maxk;

        // norm window offsets
        std::vector<int> _space_ofs(maxk);
        int* space_ofs = &_space_ofs[0];
        {
            int p1 = 0;
            int p2 = 0;
            int gap = w - local_size;
            for (int i = 0; i < local_size; i++)
            {
                for (int j = 0; j < local_size; j++)
                {
                    space_ofs[p1] = p2;
                    p1++;
                    p2++;
                }
                p2 += gap;
            }
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);
            const Mat m = square_blob_bordered.channel(q);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    const float* sptr = m.row(i) + j;

                    float ss = 0.f;

                    for (int k = 0; k < maxk; k++)
                    {
                        float val = sptr[space_ofs[k]];
                        ss += val;
                    }

                    ptr[j] = ptr[j] * powf(bias + alpha_div_size * ss, -beta);
                }

                ptr += outw;
            }
        }
    }

    return 0;
}

} // namespace ncnn
