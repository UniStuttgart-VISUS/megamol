#include "camera.h"
#include "raygen.h"

#include "optix/random.h"
#include "optix/utils_device.h"

namespace megamol {
namespace optix_hpg {
    namespace device {
        // OptiX SDK
        // Path tracer example

        //
        // Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
        //
        // Redistribution and use in source and binary forms, with or without
        // modification, are permitted provided that the following conditions
        // are met:
        //  * Redistributions of source code must retain the above copyright
        //    notice, this list of conditions and the following disclaimer.
        //  * Redistributions in binary form must reproduce the above copyright
        //    notice, this list of conditions and the following disclaimer in the
        //    documentation and/or other materials provided with the distribution.
        //  * Neither the name of NVIDIA CORPORATION nor the names of its
        //    contributors may be used to endorse or promote products derived
        //    from this software without specific prior written permission.
        //
        // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
        // EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
        // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
        // PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
        // CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
        // EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
        // PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
        // PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
        // OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
        // (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
        // OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
        //

        //
        // Modified 2021 MegaMol Dev Team
        //

        inline __device__ glm::vec4 traceRay(
            const RayGenData& self, Ray& ray /*, Random& rnd*/, PerRayData& prd, glm::vec4& bg, int maxBounces) {

            unsigned int p0 = 0;
            unsigned int p1 = 0;
            packPointer(&prd, p0, p1);

            glm::vec3 col(0.f);

            for (;;) {
                prd.wo = -ray.direction;
                optixTrace(self.world, (const float3&) ray.origin, (const float3&) ray.direction, ray.tmin, ray.tmax, 0,
                    (OptixVisibilityMask) -1,
                    /*rayFlags     */ OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                    /*SBToffset    */ 0,
                    /*SBTstride    */ 2,
                    /*missSBTIndex */ 0, p0, p1);


                /*if (prd.depth > 0) {
                    col += prd.attenuation * prd.result;
                } else {
                    col += prd.result;
                }*/

                // col += prd.emitted;
                col += prd.radiance * prd.beta;

                if (prd.done || prd.depth >= maxBounces)
                    break;

                ++prd.depth;

                ray.origin = prd.origin;
                ray.direction = prd.direction;
            }
            col += prd.emitted;
            return glm::vec4(col.x, col.y, col.z, 1.0f);
            //return glm::vec4(1, 1, 1, 1.0f);
            // return glm::vec4(prd.radiance, 1.0f);
        }

        MM_OPTIX_RAYGEN_KERNEL(raygen_program)() {
            // printf("RAYGEN1\n");
            const RayGenData& self = getProgramData<RayGenData>();
            auto const index = optixGetLaunchIndex();
            glm::ivec2 pixelID = glm::ivec2(index.x, index.y);

            if (pixelID.x >= self.fbSize.x)
                return;
            if (pixelID.y >= self.fbSize.y)
                return;
            const int pixelIdx = pixelID.x + self.fbSize.x * pixelID.y;

            const FrameState* fs = &self.frameStateBuffer[0];

            /*auto frame_idx = self.colorBufferPtr[pixelIdx].w;
            if (fs->changed) {
                frame_idx = 0.0f;
                self.colorBufferPtr[pixelIdx].w = 0.0f;
            }*/
            // auto const old_col = self.colorBufferPtr[pixelIdx];
            float4 old_col;
            surf2Dread(&old_col, self.col_surf, pixelID.x * sizeof(float4), pixelID.y, cudaBoundaryModeZero);

            unsigned int seed = tea<16>(pixelID.y * self.fbSize.x + pixelID.x, fs->frameIdx);


            glm::vec4 col(0.f);
            glm::vec4 bg = fs->background;

            // printf("RAYGEN FS %f\n", fs->near);

            auto i = fs->samplesPerPixel;

            float depth = FLT_MAX;

            do {
                PerRayData prd;

                prd.depth = 0;

                prd.radiance = glm::vec3(0.f);
                prd.pdf = 1.0f;

                prd.countDepth = true;
                prd.ray_depth = FLT_MAX;

                prd.beta = glm::vec3(1.f);

                prd.seed = seed;
                prd.done = false;

                prd.world = self.world;

                prd.countEmitted = true;
                prd.emitted = glm::vec3(0.f);

                prd.intensity = fs->intensity;

                // Random rnd(pixelIdx, 0);

                float u = -fs->rw + (fs->rw + fs->rw) * float(pixelID.x + rnd(seed)) / self.fbSize.x;
                float v = -(fs->th + (-fs->th - fs->th) * float(pixelID.y + rnd(seed)) / self.fbSize.y);
                /*float u = -fs->rw + (fs->rw + fs->rw) * float(pixelID.x) / self.fbSize.x;
                float v = -(fs->th + (-fs->th - fs->th) * float(pixelID.y) / self.fbSize.y);*/
                auto ray = generateRay(*fs, u, v);

                prd.origin = ray.origin;
                prd.direction = ray.direction;

                prd.lpos = ray.origin;
                prd.ldir = fs->camera_front;

                col += traceRay(self, ray /*, rnd*/, prd, bg, fs->maxBounces);

                depth = fminf(depth, prd.ray_depth);
            } while (--i);
            col /= (float) fs->samplesPerPixel;
            // col.w = frame_idx + 1;
            //++col.w;

            if (fs->frameIdx > 0) {
                const float a = 1.0f / static_cast<float>(fs->frameIdx + 1);
                col = lerp(glm::vec4(static_cast<float>(old_col.x), static_cast<float>(old_col.y),
                               static_cast<float>(old_col.z), static_cast<float>(old_col.w)),
                    col, a);
                // col.w = frame_idx + 1;
            }

            surf2Dwrite(make_float4(col.r, col.g, col.b, col.a), self.col_surf, pixelID.x * sizeof(float4), pixelID.y,
                cudaBoundaryModeZero);
            /*surf2Dwrite(make_float4(1, 1, 1, 1), self.col_surf, pixelID.x * sizeof(float4), pixelID.y,
                cudaBoundaryModeZero);*/

            if (depth < FLT_MAX) {
                depth = (fs->depth_params.z / depth) - (fs->depth_params.x);
                depth = 0.5f * (depth + 1.0f);
            } else {
                depth = 1.f;
            }
            surf2Dwrite(depth, self.depth_surf, pixelID.x * sizeof(float), pixelID.y, cudaBoundaryModeZero);
            //surf2Dwrite(1, self.depth_surf, pixelID.x * sizeof(float), pixelID.y, cudaBoundaryModeZero);
        }
    } // namespace device
} // namespace optix_hpg
} // namespace megamol
