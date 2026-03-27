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
    if (elembits != 32)
    {
        return ShuffleChannel::forward(bottom_blob, top_blob, opt);
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;
    int size = w * h;

    int _group = reverse ? channels * elempack / group : group;
    int channels_per_group = channels / _group;

    if (_group == 1)
    {
        top_blob = bottom_blob;
        return 0;
    }

#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
    if (elempack == 1)
    {
        // safe generic path
        return ShuffleChannel::forward(bottom_blob, top_blob, opt);
    }

    if (elempack == packn)
    {
        if (_group > 4 || channels % _group != 0)
        {
            // robust fallback per updated-prompt-mig
            Option opt_pack = opt;
            opt_pack.blob_allocator = opt.workspace_allocator;
            Mat bottom_blob_unpacked;
            convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt_pack);
            Mat top_blob_unpacked;
            int ret = ShuffleChannel::forward(bottom_blob_unpacked, top_blob_unpacked, opt_pack);
            if (ret != 0) return ret;
            convert_packing(top_blob_unpacked, top_blob, elempack, opt);
            return 0;
        }

        top_blob.create(w, h, channels, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty()) return -100;

        if (_group == 2)
        {
            // interleave two channels into two outputs by lane
            for (int q = 0; q < channels_per_group; q++)
            {
                const float* ptr0 = bottom_blob.channel(q);
                const float* ptr1 = bottom_blob.channel(channels_per_group + q);
                float* out0 = top_blob.channel(q * 2);
                float* out1 = top_blob.channel(q * 2 + 1);

                for (int i = 0; i < size; i++)
                {
                    // lower half
                    size_t vlh = __riscv_vsetvl_e32m1(packn / 2);
                    vfloat32m1_t a0 = __riscv_vle32_v_f32m1(ptr0, vlh);
                    vfloat32m1_t b0 = __riscv_vle32_v_f32m1(ptr1, vlh);
                    __riscv_vsse32_v_f32m1(out0, sizeof(float) * 2, a0, vlh);
                    __riscv_vsse32_v_f32m1(out0 + 1, sizeof(float) * 2, b0, vlh);

                    // upper half
                    vfloat32m1_t a1 = __riscv_vle32_v_f32m1(ptr0 + packn / 2, vlh);
                    vfloat32m1_t b1 = __riscv_vle32_v_f32m1(ptr1 + packn / 2, vlh);
                    __riscv_vsse32_v_f32m1(out1, sizeof(float) * 2, a1, vlh);
                    __riscv_vsse32_v_f32m1(out1 + 1, sizeof(float) * 2, b1, vlh);

                    ptr0 += packn;
                    ptr1 += packn;
                    out0 += packn;
                    out1 += packn;
                }
            }
            return 0;
        }

        if (_group == 3)
        {
            // robust fallback
            Option opt_pack = opt;
            opt_pack.blob_allocator = opt.workspace_allocator;
            Mat bottom_blob_unpacked;
            convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt_pack);
            Mat top_blob_unpacked;
            int ret = ShuffleChannel::forward(bottom_blob_unpacked, top_blob_unpacked, opt_pack);
            if (ret != 0) return ret;
            convert_packing(top_blob_unpacked, top_blob, elempack, opt);
            return 0;
        }

        if (_group == 4 && packn == 4)
        {
            // optimize for VLEN=128 (packn=4): transpose 4x4 per spatial index
            for (int q = 0; q < channels_per_group; q++)
            {
                const float* ptr0 = bottom_blob.channel(q);
                const float* ptr1 = bottom_blob.channel(channels_per_group + q);
                const float* ptr2 = bottom_blob.channel(channels_per_group * 2 + q);
                const float* ptr3 = bottom_blob.channel(channels_per_group * 3 + q);
                float* out0 = top_blob.channel(q * 4);
                float* out1 = top_blob.channel(q * 4 + 1);
                float* out2 = top_blob.channel(q * 4 + 2);
                float* out3 = top_blob.channel(q * 4 + 3);

                for (int i = 0; i < size; i++)
                {
                    size_t vl = __riscv_vsetvl_e32m1(4);
                    vfloat32m1_t r0 = __riscv_vle32_v_f32m1(ptr0, vl);
                    vfloat32m1_t r1 = __riscv_vle32_v_f32m1(ptr1, vl);
                    vfloat32m1_t r2 = __riscv_vle32_v_f32m1(ptr2, vl);
                    vfloat32m1_t r3 = __riscv_vle32_v_f32m1(ptr3, vl);
                    float tmp[16];
                    vfloat32m1x4_t seg = __riscv_vcreate_v_f32m1x4(r0, r1, r2, r3);
                    __riscv_vsseg4e32_v_f32m1x4(tmp, seg, vl);
                    vfloat32m1_t t0 = __riscv_vle32_v_f32m1(tmp + 0 * 4, vl);
                    vfloat32m1_t t1 = __riscv_vle32_v_f32m1(tmp + 1 * 4, vl);
                    vfloat32m1_t t2 = __riscv_vle32_v_f32m1(tmp + 2 * 4, vl);
                    vfloat32m1_t t3 = __riscv_vle32_v_f32m1(tmp + 3 * 4, vl);
                    __riscv_vse32_v_f32m1(out0, t0, vl);
                    __riscv_vse32_v_f32m1(out1, t1, vl);
                    __riscv_vse32_v_f32m1(out2, t2, vl);
                    __riscv_vse32_v_f32m1(out3, t3, vl);
                    ptr0 += 4; ptr1 += 4; ptr2 += 4; ptr3 += 4;
                    out0 += 4; out1 += 4; out2 += 4; out3 += 4;
                }
            }
            return 0;
        }

        // other cases fallback
        Option opt_pack = opt;
        opt_pack.blob_allocator = opt.workspace_allocator;
        Mat bottom_blob_unpacked;
        convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt_pack);
        Mat top_blob_unpacked;
        int ret = ShuffleChannel::forward(bottom_blob_unpacked, top_blob_unpacked, opt_pack);
        if (ret != 0) return ret;
        convert_packing(top_blob_unpacked, top_blob, elempack, opt);
        return 0;
    }
#endif // __riscv_vector

    return ShuffleChannel::forward(bottom_blob, top_blob, opt);
}

} // namespace ncnn
