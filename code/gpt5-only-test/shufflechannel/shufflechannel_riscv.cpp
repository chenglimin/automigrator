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

#include "shufflechannel_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#include "riscv_usability.h"
#endif // __riscv_vector

#include "cpu.h"

namespace ncnn {

ShuffleChannel_riscv::ShuffleChannel_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector
}

int ShuffleChannel_riscv::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int elembits = bottom_blob.elembits();

    // Only optimized for fp32 in this implementation. Fallback to generic otherwise.
    if (elembits != 32)
    {
        return ShuffleChannel::forward(bottom_blob, top_blob, opt);
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    // effective group considering current packing
    int _group = reverse ? channels * elempack / group : group;

    if (_group == 1)
    {
        top_blob = bottom_blob;
        return 0;
    }

    // Scalar planar path
    if (elempack == 1)
    {
        // identical with reference implementation
        if (channels % _group != 0)
            return -100;

        int channels_per_group = channels / _group;

        top_blob.create(w, h, channels, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const size_t feature_sz = (size_t)w * h * elemsize;
        for (int i = 0; i < _group; i++)
        {
            for (int j = 0; j < channels_per_group; j++)
            {
                int src_q = channels_per_group * i + j;
                int dst_q = _group * j + i;
                memcpy(top_blob.channel(dst_q), bottom_blob.channel(src_q), feature_sz);
            }
        }
        return 0;
    }

#if __riscv_vector
    // RVV optimized paths for common cases
    const int packn = csrr_vlenb() / 4;
    if (elempack == packn && channels % _group == 0)
    {
        int channels_per_group = channels / _group;
        int size = w * h;

        top_blob.create(w, h, channels, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (_group == 2)
        {
            // Interleave two channels into two outputs using segment stores
            for (int q = 0; q < channels_per_group; q++)
            {
                const float* ptr0 = bottom_blob.channel(q);
                const float* ptr1 = bottom_blob.channel(channels_per_group + q);
                float* outptr0 = top_blob.channel(q * 2);
                float* outptr1 = top_blob.channel(q * 2 + 1);

                for (int i = 0; i < size; i++)
                {
                    size_t vl = __riscv_vsetvl_e32m1(packn / 2);
                    vfloat32m1_t a0 = __riscv_vle32_v_f32m1(ptr0, vl);
                    vfloat32m1_t b0 = __riscv_vle32_v_f32m1(ptr1, vl);
                    vfloat32m1_t a1 = __riscv_vle32_v_f32m1(ptr0 + vl, vl);
                    vfloat32m1_t b1 = __riscv_vle32_v_f32m1(ptr1 + vl, vl);

                    __riscv_vsseg2e32_v_f32m1x2(outptr0, __riscv_vcreate_v_f32m1x2(a0, b0), vl);
                    __riscv_vsseg2e32_v_f32m1x2(outptr1, __riscv_vcreate_v_f32m1x2(a1, b1), vl);

                    ptr0 += packn;
                    ptr1 += packn;
                    outptr0 += packn;
                    outptr1 += packn;
                }
            }
            return 0;
        }
        if (_group == 4)
        {
            for (int q = 0; q < channels_per_group; q++)
            {
                const float* ptr0 = bottom_blob.channel(q);
                const float* ptr1 = bottom_blob.channel(channels_per_group + q);
                const float* ptr2 = bottom_blob.channel(channels_per_group * 2 + q);
                const float* ptr3 = bottom_blob.channel(channels_per_group * 3 + q);
                float* outptr0 = top_blob.channel(q * 4);
                float* outptr1 = top_blob.channel(q * 4 + 1);
                float* outptr2 = top_blob.channel(q * 4 + 2);
                float* outptr3 = top_blob.channel(q * 4 + 3);

                for (int i = 0; i < size; i++)
                {
                    size_t vl = __riscv_vsetvl_e32m1(packn / 4);
                    vfloat32m1_t a0 = __riscv_vle32_v_f32m1(ptr0, vl);
                    vfloat32m1_t b0 = __riscv_vle32_v_f32m1(ptr1, vl);
                    vfloat32m1_t c0 = __riscv_vle32_v_f32m1(ptr2, vl);
                    vfloat32m1_t d0 = __riscv_vle32_v_f32m1(ptr3, vl);
                    vfloat32m1_t a1 = __riscv_vle32_v_f32m1(ptr0 + vl, vl);
                    vfloat32m1_t b1 = __riscv_vle32_v_f32m1(ptr1 + vl, vl);
                    vfloat32m1_t c1 = __riscv_vle32_v_f32m1(ptr2 + vl, vl);
                    vfloat32m1_t d1 = __riscv_vle32_v_f32m1(ptr3 + vl, vl);
                    vfloat32m1_t a2 = __riscv_vle32_v_f32m1(ptr0 + vl * 2, vl);
                    vfloat32m1_t b2 = __riscv_vle32_v_f32m1(ptr1 + vl * 2, vl);
                    vfloat32m1_t c2 = __riscv_vle32_v_f32m1(ptr2 + vl * 2, vl);
                    vfloat32m1_t d2 = __riscv_vle32_v_f32m1(ptr3 + vl * 2, vl);
                    vfloat32m1_t a3 = __riscv_vle32_v_f32m1(ptr0 + vl * 3, vl);
                    vfloat32m1_t b3 = __riscv_vle32_v_f32m1(ptr1 + vl * 3, vl);
                    vfloat32m1_t c3 = __riscv_vle32_v_f32m1(ptr2 + vl * 3, vl);
                    vfloat32m1_t d3 = __riscv_vle32_v_f32m1(ptr3 + vl * 3, vl);

                    __riscv_vsseg4e32_v_f32m1x4(outptr0, __riscv_vcreate_v_f32m1x4(a0, b0, c0, d0), vl);
                    __riscv_vsseg4e32_v_f32m1x4(outptr1, __riscv_vcreate_v_f32m1x4(a1, b1, c1, d1), vl);
                    __riscv_vsseg4e32_v_f32m1x4(outptr2, __riscv_vcreate_v_f32m1x4(a2, b2, c2, d2), vl);
                    __riscv_vsseg4e32_v_f32m1x4(outptr3, __riscv_vcreate_v_f32m1x4(a3, b3, c3, d3), vl);

                    ptr0 += packn;
                    ptr1 += packn;
                    ptr2 += packn;
                    ptr3 += packn;
                    outptr0 += packn;
                    outptr1 += packn;
                    outptr2 += packn;
                    outptr3 += packn;
                }
            }
            return 0;
        }
    }
#endif // __riscv_vector

    // For other cases where elempack > 1 or unsupported group sizes
    // robust fallback via unpack -> generic -> repack
    Option opt_pack = opt;
    opt_pack.blob_allocator = opt.workspace_allocator;

    Mat bottom_blob_unpacked;
    convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt_pack);
    if (bottom_blob_unpacked.empty())
        return -100;

    Mat top_blob_unpacked;
    int ret = ShuffleChannel::forward(bottom_blob_unpacked, top_blob_unpacked, opt_pack);
    if (ret != 0)
        return ret;

    convert_packing(top_blob_unpacked, top_blob, elempack, opt);
    if (top_blob.empty())
        return -100;

    return 0;
}

} // namespace ncnn
