//#include "owl/owl_device.h"

#include "perraydata.h"
#include "sphere.h"

#include "glm/glm.hpp"
#include "hpg/optix/utils_device.h"

//using namespace owl;

namespace megamol {
namespace hpg {
    namespace optix {
        namespace device {
            inline __device__ bool intersectSphere(
                const Particle& particle, const float particleRadius, const Ray& ray, float& hit_t) {
                // Raytracing Gems Intersection Code (Chapter 7)
                glm::vec3 pos = glm::vec3(particle.pos);
                const glm::vec3 oc = ray.origin - pos;
                const float sqrRad = particleRadius * particleRadius;

                // const float  a = dot(ray.direction, ray.direction);
                const float b = glm::dot(-oc, ray.direction);
                const glm::vec3 temp = oc + b * ray.direction;
                const float delta = sqrRad - glm::dot(temp, temp);

                if (delta < 0.0f)
                    return false;

                const float c = glm::dot(oc, oc) - sqrRad;
                const float sign = signbit(b) ? 1.0f : -1.0f;
                const float q = b + sign * sqrtf(delta);

                {
                    float temp = fminf(c / q, q);
                    if (temp < hit_t && temp > ray.tmin) {
                        hit_t = temp;
                        return true;
                    }
                }
            }

            MM_OPTIX_INTERSECTION_KERNEL(sphere_intersect)() {
                //printf("ISEC\n");

                const int primID = optixGetPrimitiveIndex();

                const auto& self = getProgramData<SphereGeoData>();

                auto ray =
                    Ray(optixGetWorldRayOrigin(), optixGetWorldRayDirection(), optixGetRayTmin(), optixGetRayTmax());

                const Particle& particle = self.particleBufferPtr[primID];
                float tmp_hit_t = ray.tmax;
                if (intersectSphere(particle, particle.pos.w, ray, tmp_hit_t)) {
                    optixReportIntersection(tmp_hit_t, 0);
                }
            }

            MM_OPTIX_CLOSESTHIT_KERNEL(sphere_closesthit)() {
                //printf("CH\n");
                const int primID = optixGetPrimitiveIndex();
                PerRayData& prd = getPerRayData<PerRayData>();
                prd.primID = primID;
                prd.t = optixGetRayTmax();

                const auto& self = getProgramData<SphereGeoData>();

                Ray ray(
                    optixGetWorldRayOrigin(), optixGetWorldRayDirection(), optixGetRayTmin(), optixGetRayTmax());


                const Particle& particle = self.particleBufferPtr[primID];
                glm::vec3 N = (ray.origin + prd.t * ray.direction) - glm::vec3(particle.pos);
                if (glm::dot(N, ray.direction) > 0.f)
                    N = -N;
                prd.N = glm::normalize(N);

                if (self.hasColorData) {
                    prd.albedo = self.colorBufferPtr[primID];
                } else {
                    prd.albedo = self.globalColor;
                }
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
    }     // namespace optix
} // namespace hpg
} // namespace megamol
