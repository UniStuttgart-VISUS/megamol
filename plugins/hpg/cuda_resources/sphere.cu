//#include "owl/owl_device.h"

#include "perraydata.h"
#include "sphere.h"

#include "glm/glm.hpp"
#include "hpg/optix/utils_device.h"

#include "hpg/optix/random.h"

#include "glm/gtx/component_wise.hpp"

// using namespace owl;

namespace megamol {
namespace hpg {
    namespace optix {
        namespace device {
            inline __device__ void intersectSphere(
                const Particle& particle, const float particleRadius, const Ray& ray) {
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

            //static __forceinline__ __device__ void cosine_sample_hemisphere(
            //    const float u1, const float u2, glm::vec3& p) {
            //    // Uniformly sample disk.
            //    const float r = sqrtf(u1);
            //    const float phi = 2.0f * MMO_PI * u2;
            //    p.x = r * cosf(phi);
            //    p.y = r * sinf(phi);

            //    // Project up to hemisphere.
            //    p.z = sqrtf(fmaxf(0.0f, 1.0f - p.x * p.x - p.y * p.y));
            //}


            //// https://github.com/knightcrawler25/Optix-PathTracer

            // inline __device__ void Pdf(glm::vec3 const& n, PerRayData& prd) {
            //    // float3 n = state.ffnormal;
            //    auto L = prd.direction;

            //    float pdfDiff = abs(dot(L, n)) * (1.0f / MMO_PI);

            //    prd.pdf = pdfDiff;
            //}

            // inline __device__ void Sample(glm::vec3 const& N, glm::vec3 const& hp, PerRayData& prd) {
            //    // float3 N = state.ffnormal;
            //    prd.origin = hp;

            //    glm::vec3 dir;

            //    float r1 = rnd(prd.seed);
            //    float r2 = rnd(prd.seed);

            //    Onb onb(N);

            //    dir = CosineSampleHemisphere(glm::vec2(r1, r2));
            //    onb.inverse_transform(dir);

            //    prd.direction = dir;
            //}


            // inline __device__ glm::vec3 Eval(glm::vec3 const& N, glm::vec3 const& col, PerRayData& prd) {
            //    // float3 N = state.ffnormal;
            //    auto V = prd.wo;
            //    auto L = prd.direction;

            //    float NDotL = dot(N, L);
            //    float NDotV = dot(N, V);
            //    if (NDotL <= 0.0f || NDotV <= 0.0f)
            //        return glm::vec3(0.0f);

            //    glm::vec3 out = (1.0f / MMO_PI) * col;

            //    return out * glm::clamp(glm::dot(N, L), 0.0f, 1.0f);
            //}


            // inline __device__ float powerHeuristic(float fpdf, float gpdf) {
            //    return (fpdf * fpdf) / (fpdf * fpdf + gpdf * gpdf);
            //}


            // inline __device__ glm::vec3 DirectLight(
            //    glm::vec3 const& light_pos, glm::vec3 const& P, glm::vec3 const& N, PerRayData& prd, glm::vec3 const& col) {
            //    const float Ldist = length(light_pos - P);
            //    const glm::vec3 L = normalize(light_pos - P);
            //    const float nDl = dot(N, L);

            //    float3 org = make_float3(P.x, P.y, P.z);
            //    float3 dir = make_float3(L.x, L.y, L.z);

            //    if (nDl > 0.0f) {
            //        unsigned int occluded = 0;
            //        optixTrace(prd.world, org, dir, 0.01f, Ldist - 0.01f, 0.0f, (OptixVisibilityMask) -1,
            //            OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT, 1, 2, 1, occluded);
            //        if (!occluded) {
            //            float NdotL = glm::dot(prd.ldir, -L);
            //            //float lightPdf = (Ldist * Ldist) / (NdotL);
            //            float lightPdf = 1.0f / NdotL;
            //            prd.direction = L;
            //            Pdf(N, prd);
            //            auto f = Eval(prd.ldir, col, prd);
            //            return powerHeuristic(lightPdf, prd.pdf) * prd.beta * f * glm::vec3(0.4f) / fmaxf(0.001f, lightPdf);
            //        }
            //    }

            //    return glm::vec3(0.0f);
            //}


            //inline __device__ void phong(glm::vec3 const& Kd, glm::vec3 const& Ka, glm::vec3 const& Ks,
            //    glm::vec3 const& Kr, float exp, glm::vec3 const& N, Ray const& ray, PerRayData& prd) {
            //    glm::vec3 P = ray.origin + ray.tmax * ray.direction;

            //    // ambient contribution
            //    glm::vec3 result = Ka * glm::vec3(0.1f);

            //    // compute direct lighting
            //    auto const lpos = prd.lpos;
            //    float Ldist = length(lpos - P);
            //    glm::vec3 L = normalize(lpos - P);
            //    float nDl = dot(N, L);

            //    /*printf("TEST1 %f %f %f\n", lpos.x, lpos.y, lpos.z);
            //    printf("TEST2 %f %f %f\n", P.x, P.y, P.z);*/

            //    // cast shadow ray
            //    glm::vec3 light_attenuation = glm::vec3(static_cast<float>(nDl > 0.0f));
            //    unsigned int occluded = 0;
            //    if (nDl > 0.0f) {
            //        float3 Pn = make_float3(P.x, P.y, P.z);
            //        float3 Ln = make_float3(L.x, L.y, L.z);
            //        optixTrace(prd.world, Pn, Ln, 0.01f, Ldist - 0.01f, 0.0f, OptixVisibilityMask(-1),
            //            OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT, 1, 2, 1, occluded);
            //        if (occluded) {
            //            light_attenuation = glm::vec3(0.f);
            //        }
            //    }

            //    // If not completely shadowed, light the hit point
            //    if (glm::compMax(light_attenuation) > 0.f) {
            //        glm::vec3 Lc = glm::vec3(0.4f) * light_attenuation;

            //        result += Kd * nDl * Lc;

            //        glm::vec3 H = normalize(L - ray.direction);
            //        float nDh = dot(N, H);
            //        if (nDh > 0) {
            //            float power = pow(nDh, exp);
            //            result += Ks * power * Lc;
            //        }
            //    }

            //    /*if (!occluded) {
            //        result = glm::vec3(1.f, 0.0f, 0.f);
            //    }*/
            //    prd.done = true;
            //    if (glm::compMax(Kr) > 0.f) {
            //        // ray tree attenuation
            //        float new_importance = prd.importance * luminance(Kr);

            //        // reflection ray
            //        // compare new_depth to max_depth - 1 to leave room for a potential shadow ray trace
            //        if (new_importance >= 0.01f) {
            //            prd.origin = P;
            //            prd.direction = reflect(ray.direction, N);

            //            prd.attenuation = Kr;
            //            prd.done = false;
            //        }
            //    }

            //    prd.result = result;
            //}


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
                    prd.emitted = geo_col*0.2f;
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
                    //const bool occluded = traceOcclusion(params.handle, P, L,
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
    }     // namespace optix
} // namespace hpg
} // namespace megamol
