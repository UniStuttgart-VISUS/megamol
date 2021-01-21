//#include "owl/owl_device.h"

#include "perraydata.h"
#include "sphere.h"

#include "glm/glm.hpp"
#include "hpg/optix/utils_device.h"

#include "hpg/optix/random.h"

// using namespace owl;

#define MMO_PI 3.14159265358979323f

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
                // printf("ISEC\n");

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


            // Optix SDK

            struct Onb {
                inline __device__ Onb(const glm::vec3& normal) {
                    m_normal = normal;

                    if (fabs(m_normal.x) > fabs(m_normal.z)) {
                        m_binormal.x = -m_normal.y;
                        m_binormal.y = m_normal.x;
                        m_binormal.z = 0;
                    } else {
                        m_binormal.x = 0;
                        m_binormal.y = -m_normal.z;
                        m_binormal.z = m_normal.y;
                    }

                    m_binormal = normalize(m_binormal);
                    m_tangent = cross(m_binormal, m_normal);
                }

                inline __device__ void inverse_transform(glm::vec3& p) const {
                    p = p.x * m_tangent + p.y * m_binormal + p.z * m_normal;
                }

                glm::vec3 m_tangent;
                glm::vec3 m_binormal;
                glm::vec3 m_normal;
            };

            static __forceinline__ __device__ void cosine_sample_hemisphere(
                const float u1, const float u2, glm::vec3& p) {
                // Uniformly sample disk.
                const float r = sqrtf(u1);
                const float phi = 2.0f * MMO_PI * u2;
                p.x = r * cosf(phi);
                p.y = r * sinf(phi);

                // Project up to hemisphere.
                p.z = sqrtf(fmaxf(0.0f, 1.0f - p.x * p.x - p.y * p.y));
            }


            // https://github.com/knightcrawler25/Optix-PathTracer

            inline __device__ void Pdf(glm::vec3 const& n, PerRayData& prd) {
                // float3 n = state.ffnormal;
                auto L = prd.bsdfDir;

                float pdfDiff = abs(dot(L, n)) * (1.0f / MMO_PI);

                prd.pdf = pdfDiff;
            }

            inline __device__ void Sample(glm::vec3 const& N, glm::vec3 const& hp, PerRayData& prd) {
                // float3 N = state.ffnormal;
                prd.origin = hp;

                glm::vec3 dir;

                float r1 = rnd(prd.seed);
                float r2 = rnd(prd.seed);

                Onb onb(N);

                cosine_sample_hemisphere(r1, r2, dir);
                onb.inverse_transform(dir);

                prd.bsdfDir = dir;
            }


            inline __device__ glm::vec3 Eval(glm::vec3 const& N, glm::vec4 const& col, PerRayData& prd) {
                // float3 N = state.ffnormal;
                auto V = prd.wo;
                auto L = prd.bsdfDir;

                float NDotL = dot(N, L);
                float NDotV = dot(N, V);
                if (NDotL <= 0.0f || NDotV <= 0.0f)
                    return glm::vec3(0.0f);

                glm::vec3 out = (1.0f / MMO_PI) * glm::vec3(col);

                return out * glm::clamp(glm::dot(N, L), 0.0f, 1.0f);
            }


            inline __device__ float powerHeuristic(float fpdf, float gpdf) {
                return (fpdf * fpdf) / (fpdf * fpdf + gpdf * gpdf);
            }


            inline __device__ glm::vec3 DirectLight(
                glm::vec3 const& light_pos, glm::vec3 const& P, glm::vec3 const& N, PerRayData& prd) {
                const float Ldist = length(light_pos - P);
                const glm::vec3 L = normalize(light_pos - P);
                const float nDl = dot(N, L);

                float3 org = make_float3(P.x, P.y, P.z);
                float3 dir = make_float3(L.x, L.y, L.z);

                if (nDl > 0.0f) {
                    unsigned int occluded = 0;
                    optixTrace(prd.world, org, dir, 0.01f, Ldist - 0.01f, 0.0f, (OptixVisibilityMask) -1,
                        OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT, 1, 2, 1, occluded);
                    if (!occluded) {
                        float lightPdf = (Ldist * Ldist) / (nDl);
                        prd.bsdfDir = L;
                        auto f = Eval(-L, glm::vec4(1.0f), prd);

                        return powerHeuristic(lightPdf, prd.pdf) * prd.beta * f * glm::vec3(1.0f) / lightPdf;
                    }
                }

                return glm::vec3(0.0f);
            }


            MM_OPTIX_CLOSESTHIT_KERNEL(sphere_closesthit)() {
                const int primID = optixGetPrimitiveIndex();
                PerRayData& prd = getPerRayData<PerRayData>();
                prd.primID = primID;
                prd.t = optixGetRayTmax();

                const auto& self = getProgramData<SphereGeoData>();

                Ray ray(optixGetWorldRayOrigin(), optixGetWorldRayDirection(), optixGetRayTmin(), optixGetRayTmax());


                const Particle& particle = self.particleBufferPtr[primID];
                glm::vec3 hp = ray.origin + prd.t * ray.direction;
                glm::vec3 N = hp - glm::vec3(particle.pos);

                prd.radiance += glm::vec3(0.2f) * prd.beta;

                prd.radiance += DirectLight(prd.lpos, hp, N, prd);

                Sample(N, hp, prd);
                Pdf(N, prd);

                glm::vec4 col = self.globalColor;
                if (self.hasColorData) {
                    col = self.colorBufferPtr[primID];
                }
                auto f = Eval(N, col, prd);

                if (prd.pdf > 0.0f)
                    prd.beta *= f / prd.pdf;
                else
                    prd.done = true;

                /*if (glm::dot(N, ray.direction) > 0.f)
                    N = -N;
                prd.N = glm::normalize(N);

                if (self.hasColorData) {
                    prd.albedo = self.colorBufferPtr[primID];
                } else {
                    prd.albedo = self.globalColor;
                }*/
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
    }     // namespace optix
} // namespace hpg
} // namespace megamol
