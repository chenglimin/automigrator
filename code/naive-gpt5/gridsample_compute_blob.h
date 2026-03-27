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

#include "riscv_usability.h"

// The following helpers mirror x86 gridsample_compute_blob.h but only rely on scalar paths when __AVX__ is not defined.
// This header intentionally avoids x86-specific includes and intrinsics.

template<bool align_corner>
struct grid_sample_unormalize;

template<>
struct grid_sample_unormalize</*align_corner*/ true>
{
    float operator()(int length, float coord)
    {
        return (coord + 1) / 2.f * (length - 1);
    }
};

template<>
struct grid_sample_unormalize</*align_corner*/ false>
{
    float operator()(int length, float coord)
    {
        return ((coord + 1) * length - 1) / 2.f;
    }
};

template<GridSample::PaddingMode pd, bool align_corner>
struct compute_coord
{
    float operator()(int /*length*/, float coord)
    {
        return coord;
    }
};

template<bool align_corner>
struct compute_coord<GridSample::Padding_BORDER, align_corner>
{
    float operator()(int length, float coord)
    {
        return std::min(length - 1.0f, std::max(coord, 0.0f));
    }
};

template<>
struct compute_coord<GridSample::Padding_REFLECTION, /*align_corner*/ true>
{
    float operator()(int length, float coord)
    {
        coord = fabs(coord);
        coord = (length - 1) - fabs(coord - (length - 1));
        return std::min(length - 1.0f, std::max(coord, 0.0f));
    }
};

template<>
struct compute_coord<GridSample::Padding_REFLECTION, /*align_corner*/ false>
{
    float operator()(int length, float coord)
    {
        coord = fabs(coord + 0.5f);
        coord = length - fabs(coord - length) - 0.5f;
        return std::min(length - 1.0f, std::max(coord, 0.0f));
    }
};

#include "gridsample_bilinear_compute_blob.h"
#include "gridsample_bicubic_compute_blob.h"
#include "gridsample_nearest_compute_blob.h"
