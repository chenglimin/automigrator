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

#include "gridsample_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif

#include "riscv/riscv_usability.h"
#include "cpu.h"

namespace ncnn {

// Helpers ported from x86 version but adapted to scalar generic path where appropriate
// The heavy lifting of SIMD happens in apply_interpolation helpers below

static inline float grid_sample_unormalize_scalar(int w, float coordx, int align_corner)
{
    return align_corner ? (coordx + 1) / 2.f * (w - 1) : ((coordx + 1) * w - 1) / 2.f;
}

static inline float border_coord_scalar(float x, float border)
{
    return std::min(border, std::max(x, 0.0f));
}

static inline float reflect_coord_scalar(float x, int high)
{
    x = fabs(x);
    x = high - fabs(x - high);
    return x;
}

static inline float compute_coord_scalar(float sx, int w, int padding_mode, int align_corner)
{
    if (padding_mode == 2)
    {
        sx = border_coord_scalar(sx, w - 1);
    }
    else if (padding_mode == 3)
    {
        if (align_corner)
            sx = reflect_coord_scalar(sx, w - 1);
        else
        {
            sx = reflect_coord_scalar(sx + 0.5, w) - 0.5;
            sx = border_coord_scalar(sx, w - 1);
        }
    }
    return sx;
}

// Compute blob generators are translated from x86 scalar paths (no x86 intrinsics needed here)
#ifndef NCNN_GRIDSAMPLE_COMPUTE_BLOB_INCLUDED_ONCE
#define NCNN_GRIDSAMPLE_COMPUTE_BLOB_INCLUDED_ONCE
#include "../x86/gridsample_compute_blob.h"
#endif

// include scalar p1 apply paths only once to avoid redefinition
// Only include headers once via global guard defined in their own file
#include "../x86/gridsample_bilinear_apply_interpolation.h"
#include "../x86/gridsample_nearest_apply_interpolation.h"
#include "../x86/gridsample_bicubic_apply_interpolation.h"

// RVV apply interpolation implementations

static void gridsample_2d_bilinear_apply_interpolation_packn(const Mat& src, Mat& dst, const Mat& offset_value, const Option& opt)
{
#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
    const size_t vl = __riscv_vsetvl_e32m8(packn); // set vl to packn elements
    const int channels = dst.c;
    const int grid_size = dst.w * dst.h;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* srcptr = src.channel(q);
        float* dstptr = dst.channel(q);
        const float* offset_ptr_base = offset_value.channel(0);

        for (int i = 0; i < grid_size; i++)
        {
            const int* offset_ptr = (const int*)offset_ptr_base;
            const float* value_ptr = offset_ptr_base + 4;

            // load neighbors or zero
            vfloat32m8_t v00 = __riscv_vfmv_v_f_f32m8(0.f, vl);
            vfloat32m8_t v01 = v00, v10 = v00, v11 = v00;
            // Note: gather needs indices per element; we use scalar offsets to compute contiguous loads
            int o0 = offset_ptr[0];
            int o1 = offset_ptr[1];
            int o2 = offset_ptr[2];
            int o3 = offset_ptr[3];
            if (o0 >= 0) v00 = __riscv_vle32_v_f32m8(srcptr + o0, vl);
            if (o1 >= 0) v01 = __riscv_vle32_v_f32m8(srcptr + o1, vl);
            if (o2 >= 0) v10 = __riscv_vle32_v_f32m8(srcptr + o2, vl);
            if (o3 >= 0) v11 = __riscv_vle32_v_f32m8(srcptr + o3, vl);

            vfloat32m8_t a = __riscv_vfmv_v_f_f32m8(value_ptr[0], vl);
            vfloat32m8_t b = __riscv_vfmv_v_f_f32m8(value_ptr[1], vl);

            // v0 = v00*(1-a) + v01*a  => v0 = v01*a + v00 - v00*a = v01*a + v00*(1-a)
            vfloat32m8_t v0 = __riscv_vfnmadd_vv_f32m8(v00, a, v00, vl); // v00 - v00*a
            v0 = __riscv_vfmadd_vv_f32m8(v01, a, v0, vl);               // + v01*a
            vfloat32m8_t v1 = __riscv_vfnmadd_vv_f32m8(v10, a, v10, vl);
            v1 = __riscv_vfmadd_vv_f32m8(v11, a, v1, vl);

            vfloat32m8_t v = __riscv_vfnmadd_vv_f32m8(v0, b, v0, vl);
            v = __riscv_vfmadd_vv_f32m8(v1, b, v, vl);

            __riscv_vse32_v_f32m8(dstptr, v, vl);
            dstptr += packn;
            offset_ptr_base += 6;
        }
    }
#else
    // scalar fallback
    const int channels = dst.c;
    const int outw = dst.w; const int outh = dst.h; const int grid_size = outw * outh;
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* srcptr = src.channel(q);
        float* dstptr = dst.channel(q);
        const float* offset_value_ptr = offset_value.channel(0);
        for (int i = 0; i < grid_size; i++)
        {
            const int* off = (const int*)offset_value_ptr;
            const float* val = offset_value_ptr + 4;
            float v00 = off[0] >= 0 ? srcptr[off[0]] : 0.f;
            float v01 = off[1] >= 0 ? srcptr[off[1]] : 0.f;
            float v10 = off[2] >= 0 ? srcptr[off[2]] : 0.f;
            float v11 = off[3] >= 0 ? srcptr[off[3]] : 0.f;
            float a = val[0];
            float b = val[1];
            float v0 = v00 * (1 - a) + v01 * a;
            float v1 = v10 * (1 - a) + v11 * a;
            float v = v0 * (1 - b) + v1 * b;
            for (int k = 0; k < dst.elempack; k++) dstptr[k] = v;
            dstptr += dst.elempack;
            offset_value_ptr += 6;
        }
    }
#endif
}

// nearest and 3d variants can fallback to scalar for correctness first; optimize iteratively
// Note: do not include scalar apply headers twice to avoid redefinition
// We rely on inclusion order from x86 compute_blob, so remove redundant includes

GridSample_riscv::GridSample_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif
}

int GridSample_riscv::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& grid = bottom_blobs[1];
    Mat& top_blob = top_blobs[0];
    int elempack = bottom_blob.elempack;

    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;

    Mat grid_p1 = grid;
    if (grid.elempack != 1)
    {
        convert_packing(grid, grid_p1, 1, opt);
    }

    int outw = 0, outh = 0, outd = 0;
    Mat offset_value_blob;

    if (dims == 3)
    {
        outw = permute_fusion == 0 ? grid_p1.h : grid_p1.w;
        outh = permute_fusion == 0 ? grid_p1.c : grid_p1.h;

        top_blob.create(outw, outh, channels, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty()) return -100;

        if (sample_type == GridSample::Interpolation_BILINEAR)
        {
            offset_value_blob.create(outw, outh, elemsize * 6, 6, opt.workspace_allocator);
            if (offset_value_blob.empty()) return -100;

            if (padding_mode == GridSample::Padding_ZEROS)
            {
                if (align_corner == 0)
                    gridsample_2d_bilinear_compute_blob<GridSample::Padding_ZEROS, false>(bottom_blob, grid_p1, offset_value_blob, permute_fusion);
                else
                    gridsample_2d_bilinear_compute_blob<GridSample::Padding_ZEROS, true>(bottom_blob, grid_p1, offset_value_blob, permute_fusion);
            }
            else if (padding_mode == GridSample::Padding_BORDER)
            {
                if (align_corner == 0)
                    gridsample_2d_bilinear_compute_blob<GridSample::Padding_BORDER, false>(bottom_blob, grid_p1, offset_value_blob, permute_fusion);
                else
                    gridsample_2d_bilinear_compute_blob<GridSample::Padding_BORDER, true>(bottom_blob, grid_p1, offset_value_blob, permute_fusion);
            }
            else if (padding_mode == GridSample::Padding_REFLECTION)
            {
                if (align_corner == 0)
                    gridsample_2d_bilinear_compute_blob<GridSample::Padding_REFLECTION, false>(bottom_blob, grid_p1, offset_value_blob, permute_fusion);
                else
                    gridsample_2d_bilinear_compute_blob<GridSample::Padding_REFLECTION, true>(bottom_blob, grid_p1, offset_value_blob, permute_fusion);
            }
            else
            {
                NCNN_LOGE("gridsample padding_mode error\n");
                return -100;
            }
        }

        if (sample_type == GridSample::Interpolation_NEAREST)
        {
            offset_value_blob.create(outw, outh, 1, elemsize, 1, opt.workspace_allocator);
            if (offset_value_blob.empty()) return -100;

            if (padding_mode == GridSample::Padding_ZEROS)
            {
                if (align_corner == 0)
                    gridsample_2d_nearest_compute_blob<GridSample::Padding_ZEROS, false>(bottom_blob, grid_p1, offset_value_blob, permute_fusion);
                else
                    gridsample_2d_nearest_compute_blob<GridSample::Padding_ZEROS, true>(bottom_blob, grid_p1, offset_value_blob, permute_fusion);
            }
            else if (padding_mode == GridSample::Padding_BORDER)
            {
                if (align_corner == 0)
                    gridsample_2d_nearest_compute_blob<GridSample::Padding_BORDER, false>(bottom_blob, grid_p1, offset_value_blob, permute_fusion);
                else
                    gridsample_2d_nearest_compute_blob<GridSample::Padding_BORDER, true>(bottom_blob, grid_p1, offset_value_blob, permute_fusion);
            }
            else if (padding_mode == GridSample::Padding_REFLECTION)
            {
                if (align_corner == 0)
                    gridsample_2d_nearest_compute_blob<GridSample::Padding_REFLECTION, false>(bottom_blob, grid_p1, offset_value_blob, permute_fusion);
                else
                    gridsample_2d_nearest_compute_blob<GridSample::Padding_REFLECTION, true>(bottom_blob, grid_p1, offset_value_blob, permute_fusion);
            }
            else
            {
                NCNN_LOGE("gridsample padding_mode error\n");
                return -100;
            }
        }

        if (sample_type == GridSample::Interpolation_BICUBIC)
        {
            offset_value_blob.create(outw, outh, elemsize * 18, 18, opt.workspace_allocator);
            if (offset_value_blob.empty()) return -100;

            if (padding_mode == GridSample::Padding_ZEROS)
            {
                if (align_corner == 0)
                    gridsample_2d_bicubic_compute_blob<GridSample::Padding_ZEROS, false>(bottom_blob, grid_p1, offset_value_blob, permute_fusion);
                else
                    gridsample_2d_bicubic_compute_blob<GridSample::Padding_ZEROS, true>(bottom_blob, grid_p1, offset_value_blob, permute_fusion);
            }
            else if (padding_mode == GridSample::Padding_BORDER)
            {
                if (align_corner == 0)
                    gridsample_2d_bicubic_compute_blob<GridSample::Padding_BORDER, false>(bottom_blob, grid_p1, offset_value_blob, permute_fusion);
                else
                    gridsample_2d_bicubic_compute_blob<GridSample::Padding_BORDER, true>(bottom_blob, grid_p1, offset_value_blob, permute_fusion);
            }
            else if (padding_mode == GridSample::Padding_REFLECTION)
            {
                if (align_corner == 0)
                    gridsample_2d_bicubic_compute_blob<GridSample::Padding_REFLECTION, false>(bottom_blob, grid_p1, offset_value_blob, permute_fusion);
                else
                    gridsample_2d_bicubic_compute_blob<GridSample::Padding_REFLECTION, true>(bottom_blob, grid_p1, offset_value_blob, permute_fusion);
            }
            else
            {
                NCNN_LOGE("gridsample padding_mode error\n");
                return -100;
            }
        }
    }

    if (dims == 4)
    {
        int outw_ = permute_fusion == 0 ? grid_p1.h : grid_p1.w;
        int outh_ = permute_fusion == 0 ? grid_p1.d : grid_p1.h;
        int outd_ = permute_fusion == 0 ? grid_p1.c : grid_p1.d;
        outw = outw_; outh = outh_; outd = outd_;

        top_blob.create(outw, outh, outd, channels, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty()) return -100;

        if (sample_type == GridSample::Interpolation_BILINEAR)
        {
            offset_value_blob.create(outw, outh, outd, elemsize * 11, 11, opt.workspace_allocator);
            if (offset_value_blob.empty()) return -100;

            if (padding_mode == GridSample::Padding_ZEROS)
            {
                if (align_corner == 0)
                    gridsample_3d_bilinear_compute_blob<GridSample::Padding_ZEROS, false>(bottom_blob, grid_p1, offset_value_blob, permute_fusion);
                else
                    gridsample_3d_bilinear_compute_blob<GridSample::Padding_ZEROS, true>(bottom_blob, grid_p1, offset_value_blob, permute_fusion);
            }
            else if (padding_mode == GridSample::Padding_BORDER)
            {
                if (align_corner == 0)
                    gridsample_3d_bilinear_compute_blob<GridSample::Padding_BORDER, false>(bottom_blob, grid_p1, offset_value_blob, permute_fusion);
                else
                    gridsample_3d_bilinear_compute_blob<GridSample::Padding_BORDER, true>(bottom_blob, grid_p1, offset_value_blob, permute_fusion);
            }
            else if (padding_mode == GridSample::Padding_REFLECTION)
            {
                if (align_corner == 0)
                    gridsample_3d_bilinear_compute_blob<GridSample::Padding_REFLECTION, false>(bottom_blob, grid_p1, offset_value_blob, permute_fusion);
                else
                    gridsample_3d_bilinear_compute_blob<GridSample::Padding_REFLECTION, true>(bottom_blob, grid_p1, offset_value_blob, permute_fusion);
            }
            else
            {
                NCNN_LOGE("gridsample padding_mode error\n");
                return -100;
            }
        }

        if (sample_type == GridSample::Interpolation_NEAREST)
        {
            offset_value_blob.create(outw, outh, outd, 1, elemsize, 1, opt.workspace_allocator);
            if (offset_value_blob.empty()) return -100;

            if (padding_mode == GridSample::Padding_ZEROS)
            {
                if (align_corner == 0)
                    gridsample_3d_nearest_compute_blob<GridSample::Padding_ZEROS, false>(bottom_blob, grid_p1, offset_value_blob, permute_fusion);
                else
                    gridsample_3d_nearest_compute_blob<GridSample::Padding_ZEROS, true>(bottom_blob, grid_p1, offset_value_blob, permute_fusion);
            }
            else if (padding_mode == GridSample::Padding_BORDER)
            {
                if (align_corner == 0)
                    gridsample_3d_nearest_compute_blob<GridSample::Padding_BORDER, false>(bottom_blob, grid_p1, offset_value_blob, permute_fusion);
                else
                    gridsample_3d_nearest_compute_blob<GridSample::Padding_BORDER, true>(bottom_blob, grid_p1, offset_value_blob, permute_fusion);
            }
            else if (padding_mode == GridSample::Padding_REFLECTION)
            {
                if (align_corner == 0)
                    gridsample_3d_nearest_compute_blob<GridSample::Padding_REFLECTION, false>(bottom_blob, grid_p1, offset_value_blob, permute_fusion);
                else
                    gridsample_3d_nearest_compute_blob<GridSample::Padding_REFLECTION, true>(bottom_blob, grid_p1, offset_value_blob, permute_fusion);
            }
            else
            {
                NCNN_LOGE("gridsample padding_mode error\n");
                return -100;
            }
        }

        if (sample_type == 3)
        {
            NCNN_LOGE("unsupported bicubic when dims == 4");
            return -100;
        }
    }

    // dispatch apply interpolation according to elempack similar to x86 but RVV uses packn
    const int packn = cpu_riscv_vlenb() / 4;
    if (elempack == 1)
    {
        if (dims == 3)
        {
            if (sample_type == GridSample::Interpolation_BILINEAR)
                gridsample_2d_bilinear_apply_interpolation_p1(bottom_blob, top_blob, offset_value_blob, opt);
            else if (sample_type == GridSample::Interpolation_NEAREST)
                gridsample_nearest_apply_interpolation_p1(bottom_blob, top_blob, offset_value_blob, opt);
            else if (sample_type == GridSample::Interpolation_BICUBIC)
                gridsample_2d_bicubic_apply_interpolation_p1(bottom_blob, top_blob, offset_value_blob, opt);
        }
        else if (dims == 4)
        {
            if (sample_type == GridSample::Interpolation_BILINEAR)
                gridsample_3d_bilinear_apply_interpolation_p1(bottom_blob, top_blob, offset_value_blob, opt);
            else if (sample_type == GridSample::Interpolation_NEAREST)
                gridsample_nearest_apply_interpolation_p1(bottom_blob, top_blob, offset_value_blob, opt);
        }
    }
    else
    {
        // Fallback: unpack -> generic -> repack
        Mat bottom_unpacked;
        convert_packing(bottom_blob, bottom_unpacked, 1, opt);
        std::vector<Mat> bb = {bottom_unpacked, grid_p1};
        // call generic GridSample
        GridSample generic;
        ParamDict pd;
        pd.set(0, sample_type);
        pd.set(1, padding_mode);
        pd.set(2, align_corner);
        pd.set(3, permute_fusion);
        generic.load_param(pd);
        std::vector<Mat> top_blobs2(1);
        int ret = generic.forward(bb, top_blobs2, opt);
        if (ret != 0) return ret;
        convert_packing(top_blobs2[0], top_blob, elempack, opt);
        return 0;
    }

    return 0;
}

} // namespace ncnn
