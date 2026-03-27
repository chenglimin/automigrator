// Copyright (C) 2026 OpenHands. All rights reserved.
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

#include "flatten_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

#include "riscv_usability.h"

namespace ncnn {

Flatten_riscv::Flatten_riscv()
{
    // Do not advertise packing to keep Flatten output planar and consistent across VLEN
    support_packing = false;
}

int Flatten_riscv::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // Preserve generic flatten semantics to ensure shape compatibility across VLEN
    return Flatten::forward(bottom_blob, top_blob, opt);
}

int Flatten_riscv::forward_int8(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    return Flatten::forward(bottom_blob, top_blob, opt);
}

} // namespace ncnn
