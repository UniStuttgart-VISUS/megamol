#include "perraydata.h"
#include "sphere.h"

#include "glm/glm.hpp"
#include "optix/utils_device.h"

#include "optix/random.h"

#include "glm/gtx/component_wise.hpp"

// OptiX SDK and rtxpkd

namespace megamol {
namespace optix_hpg {
    namespace device {
        inline __device__ void intersectSphere(const Particle& particle, const float particleRadius, const Ray& ray) {
            // Raytracing Gems Intersection Code (Chapter 7)
            const glm::vec3 pos = glm::vec3(particle.pos);
            const glm::vec3 oc = ray.origin - pos;
            const float sqrRad = particleRadius * particleRadius;

            // const float  a = dot(ray.direction, ray.direction);
            const float b = glm::dot(-oc, ray.direction);
            const glm::vec3 temp = oc + b * ray.direction;
            const float delta = sqrRad - glm::dot(temp, temp);

            if (delta < 0.0f)
                return;

            const float c = glm::dot(oc, oc) - sqrRad;
            const float q = b + copysignf(sqrtf(delta), b);

            {
                const float t = fminf(c / q, q);
                if (t > ray.tmin && t < ray.tmax) {
                    optixReportIntersection(t, 0);
                }
            }
        }

        MM_OPTIX_INTERSECTION_KERNEL(sphere_intersect)() {
            // printf("ISEC\n");

            const int primID = optixGetPrimitiveIndex();

            const auto& self = getProgramData<SphereGeoData>();

            auto const ray =
                Ray(optixGetWorldRayOrigin(), optixGetWorldRayDirection(), optixGetRayTmin(), optixGetRayTmax());

            const Particle& particle = self.particleBufferPtr[primID];
            // float tmp_hit_t = ray.tmax;
            /*if (intersectSphere(particle, particle.pos.w, ray, tmp_hit_t)) {
                optixReportIntersection(tmp_hit_t, 0);
            }*/
            intersectSphere(particle, particle.pos.w, ray);
        }

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

        MM_OPTIX_CLOSESTHIT_KERNEL(sphere_closesthit)() {
            const int primID = optixGetPrimitiveIndex();
            PerRayData& prd = getPerRayData<PerRayData>();
            /*prd.primID = primID;
            prd.t = optixGetRayTmax();*/

            const auto& self = getProgramData<SphereGeoData>();

            Ray ray(optixGetWorldRayOrigin(), optixGetWorldRayDirection(), optixGetRayTmin(), optixGetRayTmax());


            const Particle& particle = self.particleBufferPtr[primID];
            glm::vec3 P = ray.origin + ray.tmax * ray.direction;
            glm::vec3 N = glm::normalize(P - glm::vec3(particle.pos));

            glm::vec3 ffN = faceforward(N, -ray.direction, N);

            glm::vec3 geo_col = glm::vec3(self.globalColor);
            if (self.hasColorData) {
                geo_col = glm::vec3(self.colorBufferPtr[primID]);
            }

            if (prd.countEmitted)
                prd.emitted = geo_col * 0.2f;
            else
                prd.emitted = glm::vec3(0.0f);


            unsigned int seed = prd.seed;

            {
                const float z1 = rnd(seed);
                const float z2 = rnd(seed);

                glm::vec3 w_in;
                w_in = CosineSampleHemisphere(glm::vec2(z1, z2));
                Onb onb(N);
                onb.inverse_transform(w_in);
                prd.direction = w_in;
                prd.origin = P;

                prd.beta *= geo_col;
                prd.countEmitted = false;
            }

            const float z1 = rnd(seed);
            const float z2 = rnd(seed);


            // Calculate properties of light sample (for area based pdf)
            const float Ldist = length(prd.lpos - P);
            const glm::vec3 L = normalize(prd.lpos - P);
            const float nDl = dot(N, L);
            const float LnDl = -dot(prd.ldir, L);

            float weight = 0.0f;
            if (nDl > 0.0f && LnDl > 0.0f) {
                // const bool occluded = traceOcclusion(params.handle, P, L,
                //    0.01f,        // tmin
                //    Ldist - 0.01f // tmax
                //);
                float3 Pn = make_float3(P.x, P.y, P.z);
                float3 Ln = make_float3(L.x, L.y, L.z);
                unsigned int occluded = 0;
                optixTrace(prd.world, Pn, Ln, 0.01f, Ldist - 0.01f, 0.0f, (OptixVisibilityMask) -1,
                    OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT, 1, 2, 1, occluded);

                if (!occluded) {
                    weight = nDl * LnDl / (MMO_PI * Ldist * Ldist);
                }
            }

            prd.radiance += glm::vec3(0.6f) * weight;
        }


        MM_OPTIX_CLOSESTHIT_KERNEL(sphere_closesthit_occlusion)() {
            /*PerRayData& prd = getPerRayData<PerRayData>();
            prd.inShadow = true;*/
            optixSetPayload_0(1);
        }


        MM_OPTIX_BOUNDS_KERNEL(sphere_bounds)(const void* geomData, box3f& primBounds, const unsigned int primID) {
            /*const SphereGeoData& self = *(const SphereGeoData*) geomData;

            const Particle& particle = self.particleBufferPtr[primID];*/
            Particle const* particles = (Particle const*) geomData;
            Particle const& particle = particles[primID];


            primBounds.lower = glm::vec3(particle.pos) - particle.pos.w;
            primBounds.upper = glm::vec3(particle.pos) + particle.pos.w;

            // printf("BOUNDS: %d with radius %f and box %f %f %f %f %f %f\n", primID, self.radius,
            // primBounds.lower.x, primBounds.lower.y, primBounds.lower.z, primBounds.upper.x, primBounds.upper.y,
            // primBounds.upper.z);
        }
    } // namespace device
} // namespace optix_hpg
} // namespace megamol
