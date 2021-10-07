#pragma once

#include "optix/utils_device.h"

#include "optix/random.h"

#include "glm/glm.hpp"
#include "glm/gtx/component_wise.hpp"

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
                const float ta = c / q;
                const float tb = q;
                const float t = fminf(ta, tb);
                if (t < 0.f)
                    return;
                if (t > ray.tmin && t < ray.tmax) {
                    optixReportIntersection(t, 0);
                }
            }
        }

        inline __device__ void kernel_sphere_intersect() {
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

        inline __device__ void kernel_sphere_closest_hit() {
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

            set_depth(prd, ray.tmax);
            lighting(prd, geo_col, P, ffN);
        }

        inline __device__ void kernel_bounds(const void* geomData, box3f& primBounds, const unsigned int primID) {
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
