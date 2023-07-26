#include "optix/utils_device.h"

#include "mesh.h"
#include "perraydata.h"

#include "optix/random.h"

#include <cuda.h>

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

        MM_OPTIX_CLOSESTHIT_KERNEL(mesh_closesthit)() {
            const int primID = optixGetPrimitiveIndex();
            PerRayData& prd = getPerRayData<PerRayData>();

            const auto& self = getProgramData<MeshGeoData>();

            const Ray ray(optixGetWorldRayOrigin(), optixGetWorldRayDirection(), optixGetRayTmin(), optixGetRayTmax());

            /*const float2 tmp_bary = optixGetTriangleBarycentrics();
            const glm::vec2 bary = glm::vec2(tmp_bary.x, tmp_bary.y);*/
            const glm::uvec3 indices = self.index_buffer[primID];
            const glm::vec3 v0 = self.vertex_buffer[indices.x];
            const glm::vec3 v1 = self.vertex_buffer[indices.y];
            const glm::vec3 v2 = self.vertex_buffer[indices.z];
            const glm::vec3 N = normalize(cross(v1 - v0, v2 - v0));

            // const auto tmp_N = optixTransformNormalFromObjectToWorldSpace(make_float3(normal.x, normal.y, normal.z));

            // const glm::vec3 N = normalize(glm::vec3(tmp_N.x, tmp_N.y, tmp_N.z));
            // const glm::vec3 N = normalize(normal);
            const glm::vec3 P = ray.origin + ray.tmax * ray.direction;

            glm::vec3 ffN = faceforward(N, -ray.direction, N);

            /*glm::vec3 geo_col = glm::vec3(self.globalColor);
            if (self.hasColorData) {
                geo_col = glm::vec3(self.colorBufferPtr[primID]);
            }*/

            glm::vec3 geo_col = glm::vec3(1.f, 1.f, 0.f);

            set_depth(prd, ray.tmax);
            lighting(prd, geo_col, P, ffN);
        }

        MM_OPTIX_CLOSESTHIT_KERNEL(mesh_closesthit_occlusion)() {
            optixSetPayload_0(1);
        }
    } // namespace device
} // namespace optix_hpg
} // namespace megamol
