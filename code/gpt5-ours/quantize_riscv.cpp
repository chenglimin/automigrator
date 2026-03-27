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

#include "quantize_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

#include "riscv_usability.h"
#include "packing.h"
#include <math.h>

namespace ncnn {

Quantize_riscv::Quantize_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector
}

static inline signed char float2int8_scalar(float v)
{
    int int32 = (int)roundf(v);
    if (int32 > 127) return 127;
    if (int32 < -127) return -127;
    return (signed char)int32;
}

#if __riscv_vector
static inline vint8m1_t float2int8_v_f32m4(vfloat32m4_t v, size_t vl)
{
    // Emulate std::round (ties away from zero): v + sign(v)*0.5 then truncate toward zero
    vfloat32m4_t half = __riscv_vfmv_v_f_f32m4(0.5f, vl);
    vfloat32m4_t offset = __riscv_vfsgnj_vv_f32m4(half, v, vl);
    vfloat32m4_t vround = __riscv_vfadd_vv_f32m4(v, offset, vl);
    vint32m4_t vi32 = __riscv_vfcvt_x_f_v_i32m4_rm(vround, __RISCV_FRM_RTZ, vl);
    // Narrow 32->16 then 16->8 with explicit VXRM RNU
    vint16m2_t vi16 = __riscv_vnclip_wx_i16m2(vi32, 0, __RISCV_VXRM_RNU, vl);
    vint8m1_t vi8 = __riscv_vnclip_wx_i8m1(vi16, 0, __RISCV_VXRM_RNU, vl);
    // Clamp -128 to -127 to match NCNN symmetric quantization
    vint8m1_t minv = __riscv_vmv_v_x_i8m1(-127, vl);
    vi8 = __riscv_vmax_vv_i8m1(vi8, minv, vl);
    return vi8;
}
#endif // __riscv_vector

static void quantize_scalar(const float* ptr, signed char* s8ptr, const Mat& scale_data, int elemcount, int elempack)
{
    const int sds = scale_data.w;
    const int size = elemcount * elempack;

    if (sds == 1)
    {
        const float scale = scale_data[0];
        for (int i = 0; i < size; i++)
            s8ptr[i] = float2int8_scalar(ptr[i] * scale);
        return;
    }

    if (elempack > 1 && sds == elempack)
    {
        for (int i = 0; i < elemcount; i++)
        {
            for (int p = 0; p < elempack; p++)
            {
                float v = ptr[i * elempack + p] * scale_data[p];
                s8ptr[i * elempack + p] = float2int8_scalar(v);
            }
        }
        return;
    }

    // per-element scales case
    for (int i = 0; i < size; i++)
    {
        float v = ptr[i] * scale_data[i];
        s8ptr[i] = float2int8_scalar(v);
    }
}

#if __riscv_vector
static void quantize_v(const float* ptr, signed char* s8ptr, const Mat& scale_data, int elemcount, int elempack)
{
    const int scale_data_size = scale_data.w;
    const int size = elemcount * elempack;

    int n = size;
    if (scale_data_size == 1)
    {
        float scale = scale_data[0];
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m4(n);
            vfloat32m4_t _v = __riscv_vle32_v_f32m4(ptr, vl);
            vfloat32m4_t _s = __riscv_vfmv_v_f_f32m4(scale, vl);
            _v = __riscv_vfmul_vv_f32m4(_v, _s, vl);
            vint8m1_t _q = float2int8_v_f32m4(_v, vl);
            __riscv_vse8_v_i8m1(s8ptr, _q, vl);
            ptr += vl;
            s8ptr += vl;
            n -= vl;
        }
    }
    else
    {
        // Fallback to scalar for per-lane scales to ensure correctness
        quantize_scalar(ptr, s8ptr, scale_data, elemcount, elempack);
    }
}
#endif // __riscv_vector

int Quantize_riscv::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    const int dims = bottom_blob.dims;
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;

    if (dims == 1)
    {
        int out_elempack = 1;
#if __riscv_vector
        if (opt.use_packing_layout)
        {
            const int packn = csrr_vlenb() / 4;
            out_elempack = (w * elempack % packn == 0) ? packn : 1;
        }
#endif
        const int outw = w * elempack / out_elempack;
        const size_t out_elemsize = out_elempack * 1u;

        top_blob.create(outw, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const float* ptr = (const float*)bottom_blob;
        signed char* s8ptr = (signed char*)top_blob;

        const int size = w * elempack;
#if __riscv_vector
        // Follow generic semantics: use single global scale for dims==1
        int n = size;
        float scale = scale_data[0];
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m4(n);
            vfloat32m4_t _v = __riscv_vle32_v_f32m4(ptr, vl);
            vfloat32m4_t _s = __riscv_vfmv_v_f_f32m4(scale, vl);
            _v = __riscv_vfmul_vv_f32m4(_v, _s, vl);
            vint8m1_t _q = float2int8_v_f32m4(_v, vl);
            __riscv_vse8_v_i8m1(s8ptr, _q, vl);
            ptr += vl;
            s8ptr += vl;
            n -= vl;
        }
#else
        // use first scale only as generic implementation
        Mat scale1; scale1.create(1); scale1[0] = scale_data[0];
        quantize_scalar(ptr, s8ptr, scale1, size, 1);
#endif
    }

    if (dims == 2)
    {
        int out_elempack = 1;
#if __riscv_vector
        if (opt.use_packing_layout)
        {
            const int packn = csrr_vlenb() / 4;
            out_elempack = (h * elempack % packn == 0) ? packn : 1;
        }
#endif
        const int outh = h * elempack / out_elempack;
        const size_t out_elemsize = out_elempack * 1u;

        top_blob.create(w, outh, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (elempack == out_elempack)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const float* ptr = bottom_blob.row(i);
                signed char* s8ptr = top_blob.row<signed char>(i);

                const Mat scale_data_i = scale_data_size > 1 ? scale_data.range(i * elempack, elempack) : scale_data;
#if __riscv_vector
                quantize_v(ptr, s8ptr, scale_data_i, w, elempack);
#else
                quantize_scalar(ptr, s8ptr, scale_data_i, w, elempack);
#endif
            }
        }
        else if (elempack > out_elempack)
        {
            // down-pack then quantize then repack if necessary per memcopy_map.txt guidance
            Mat unpacked;
            {
                // use generic Packing to unpack to elempack=1
                Option opt2 = opt;
                Packing pack;
                pack.out_elempack = 1;
                pack.use_padding = false;
                pack.forward(bottom_blob, unpacked, opt2);
            }
            Mat qblob;
            qblob.create(w, h * elempack, (size_t)1u, 1, opt.blob_allocator);
            if (qblob.empty()) return -100;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h * elempack; i++)
            {
                const float* ptr = unpacked.row(i);
                signed char* s8ptr = qblob.row<signed char>(i);
                const float scale = scale_data_size > 1 ? scale_data[i] : scale_data[0];
                for (int j = 0; j < w; j++) s8ptr[j] = float2int8_scalar(ptr[j] * scale);
            }
            // repack to out_elempack if needed
            if (out_elempack != 1)
            {
                Mat repacked;
                Option opt2 = opt;
                Packing pack2;
                pack2.out_elempack = out_elempack;
                pack2.use_padding = false;
                pack2.forward(qblob, top_blob, opt2);
            }
            else
            {
                top_blob = qblob;
                top_blob.h = outh;
            }
        }
    }

    if (dims == 3)
    {
        int out_elempack = 1;
#if __riscv_vector
        if (opt.use_packing_layout)
        {
            const int packn = csrr_vlenb() / 4;
            out_elempack = (channels * elempack % packn == 0) ? packn : 1;
        }
#endif
        const int outc = channels * elempack / out_elempack;
        const size_t out_elemsize = out_elempack * 1u;

        top_blob.create(w, h, outc, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (elempack == out_elempack)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                signed char* s8ptr = top_blob.channel(q);

                const Mat scale_data_q = scale_data_size > 1 ? scale_data.range(q * elempack, elempack) : scale_data;
#if __riscv_vector
                quantize_v(ptr, s8ptr, scale_data_q, w * h, elempack);
#else
                quantize_scalar(ptr, s8ptr, scale_data_q, w * h, elempack);
#endif
            }
        }
        else if (elempack > out_elempack)
        {
            Mat unpacked;
            {
                Option opt2 = opt;
                Packing pack;
                pack.out_elempack = 1;
                pack.use_padding = false;
                pack.forward(bottom_blob, unpacked, opt2);
            }
            Mat qblob;
            qblob.create(w, h, channels * elempack, (size_t)1u, 1, opt.blob_allocator);
            if (qblob.empty()) return -100;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels * elempack; q++)
            {
                const float* ptr = unpacked.channel(q);
                signed char* s8ptr = qblob.channel(q);
                const float scale = scale_data_size > 1 ? scale_data[q] : scale_data[0];
                for (int i = 0; i < w * h; i++) s8ptr[i] = float2int8_scalar(ptr[i] * scale);
            }
            if (out_elempack != 1)
            {
                Packing pack2;
                pack2.out_elempack = out_elempack;
                pack2.use_padding = false;
                pack2.forward(qblob, top_blob, opt);
            }
            else
            {
                top_blob = qblob;
                top_blob.c = outc;
            }
        }
    }

    return 0;
}

} // namespace ncnn
