// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "yolov3detectionoutput_riscv.h"

#include <float.h>

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

namespace ncnn {

Yolov3DetectionOutput_riscv::Yolov3DetectionOutput_riscv()
{
}

static inline float sigmoid(float x)
{
    return 1.f / (1.f + expf(-x));
}

int Yolov3DetectionOutput_riscv::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    // gather all box
    std::vector<BBoxRect> all_bbox_rects;

    for (size_t b = 0; b < bottom_blobs.size(); b++)
    {
        std::vector<std::vector<BBoxRect> > all_box_bbox_rects;
        all_box_bbox_rects.resize(num_box);
        const Mat& bottom_top_blobs = bottom_blobs[b];

        int w = bottom_top_blobs.w;
        int h = bottom_top_blobs.h;
        int channels = bottom_top_blobs.c;
        const int channels_per_box = channels / num_box;

        // anchor coord + box score + num_class
        if (channels_per_box != 4 + 1 + num_class)
            return -1;
        size_t mask_offset = b * num_box;
        int net_w = (int)(anchors_scale[b] * w);
        int net_h = (int)(anchors_scale[b] * h);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int pp = 0; pp < num_box; pp++)
        {
            int p = pp * channels_per_box;
            int biases_index = static_cast<int>(mask[pp + mask_offset]);
            const float bias_w = biases[biases_index * 2];
            const float bias_h = biases[biases_index * 2 + 1];
            const float* xptr = bottom_top_blobs.channel(p);
            const float* yptr = bottom_top_blobs.channel(p + 1);
            const float* wptr = bottom_top_blobs.channel(p + 2);
            const float* hptr = bottom_top_blobs.channel(p + 3);

            const float* box_score_ptr = bottom_top_blobs.channel(p + 4);

            // class scores
            Mat scores = bottom_top_blobs.channel_range(p + 5, num_class);
            // softmax is disabled in original base implementation

            const int cs = (int)scores.cstep;

            for (int i = 0; i < h; i++)
            {
                for (int j = 0; j < w; j++)
                {
#if __riscv_vector
                    // vectorized max-reduction across classes in steps
                    int class_index = 0;
                    float class_score = -FLT_MAX;
                    float* ptr = ((float*)scores.data) + i * w + j;
                    int q = 0;
                    for (; q + 7 < num_class; q += 8)
                    {
                        size_t vl = 8; // fixed chunk of 8 classes
                        // gather with stride cstep
                        float tmp[8];
                        for (int t = 0; t < 8; t++) tmp[t] = ptr[t * cs];
                        vfloat32m2_t v = __riscv_vle32_v_f32m2(tmp, vl);
                        // horizontal max
                        vfloat32m1_t red = __riscv_vfmv_s_f_f32m1(-FLT_MAX, vl);
                        red = __riscv_vfredmax_vs_f32m2_f32m1(v, red, vl);
                        float score = __riscv_vfmv_f_s_f32m1_f32(red);
                        if (score > class_score)
                        {
                            // find index inside the 8 by scalar check
                            int local = 0;
                            for (int t = 0; t < 8; t++)
                                if (tmp[t] == score) { local = t; break; }
                            class_index = q + local;
                            class_score = score;
                        }
                        ptr += 8 * cs;
                    }
                    // tail
                    for (; q < num_class; q++)
                    {
                        float score = *ptr;
                        if (score > class_score)
                        {
                            class_index = q;
                            class_score = score;
                        }
                        ptr += cs;
                    }
#else
                    // scalar path
                    int class_index = 0;
                    float class_score = -FLT_MAX;
                    for (int q = 0; q < num_class; q++)
                    {
                        float score = scores.channel(q).row(i)[j];
                        if (score > class_score)
                        {
                            class_index = q;
                            class_score = score;
                        }
                    }
#endif
                    float confidence = 1.f / ((1.f + expf(-box_score_ptr[0]) * (1.f + expf(-class_score))));
                    if (confidence >= confidence_threshold)
                    {
                        float bbox_cx = (j + sigmoid(xptr[0])) / w;
                        float bbox_cy = (i + sigmoid(yptr[0])) / h;
                        float bbox_w = expf(wptr[0]) * bias_w / net_w;
                        float bbox_h = expf(hptr[0]) * bias_h / net_h;

                        float bbox_xmin = bbox_cx - bbox_w * 0.5f;
                        float bbox_ymin = bbox_cy - bbox_h * 0.5f;
                        float bbox_xmax = bbox_cx + bbox_w * 0.5f;
                        float bbox_ymax = bbox_cy + bbox_h * 0.5f;

                        float area = bbox_w * bbox_h;

                        BBoxRect c = {confidence, bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax, area, class_index};
                        all_box_bbox_rects[pp].push_back(c);
                    }

                    xptr++;
                    yptr++;
                    wptr++;
                    hptr++;

                    box_score_ptr++;
                }
            }
        }

        for (int i = 0; i < num_box; i++)
        {
            const std::vector<BBoxRect>& box_bbox_rects = all_box_bbox_rects[i];
            all_bbox_rects.insert(all_bbox_rects.end(), box_bbox_rects.begin(), box_bbox_rects.end());
        }
    }

    // global sort inplace
    qsort_descent_inplace(all_bbox_rects);

    // apply nms
    std::vector<size_t> picked;
    nms_sorted_bboxes(all_bbox_rects, picked, nms_threshold);

    // select
    std::vector<BBoxRect> bbox_rects;
    for (size_t i = 0; i < picked.size(); i++)
    {
        size_t z = picked[i];
        bbox_rects.push_back(all_bbox_rects[z]);
    }

    int num_detected = static_cast<int>(bbox_rects.size());
    if (num_detected == 0)
        return 0;

    Mat& top_blob = top_blobs[0];
    top_blob.create(6, num_detected, 4u, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    for (int i = 0; i < num_detected; i++)
    {
        const BBoxRect& r = bbox_rects[i];
        float score = r.score;
        float* outptr = top_blob.row(i);

        outptr[0] = r.label + 1.0f;
        outptr[1] = score;
        outptr[2] = r.xmin;
        outptr[3] = r.ymin;
        outptr[4] = r.xmax;
        outptr[5] = r.ymax;
    }

    return 0;
}

} // namespace ncnn
