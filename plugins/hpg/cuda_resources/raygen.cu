//#include "owl/common/math/random.h"
//#include "owl/owl_device.h"

#include "camera.h"
#include "raygen.h"

#include "hpg/optix/random.h"
#include "hpg/optix/utils_device.h"

namespace megamol {
namespace hpg {
    namespace optix {
        namespace device {
            // using Random = owl::common::LCG<>;

            inline __device__ glm::vec4 traceRay(
                const RayGenData& self, Ray& ray /*, Random& rnd*/, PerRayData& prd, glm::vec4& bg) {
                /*glm::vec4 attenuation = glm::vec4(1.f);
                glm::vec4 ambientLight(.8f, .8f, .8f, 1.f);

                prd.primID = -1;*/

                // printf("TRACE: ray %f %f %f %f %f %f\n", ray.origin.x, ray.origin.y, ray.origin.z, ray.direction.x,
                // ray.direction.y, ray.direction.z);

                // owl::traceRay(/*accel to trace against*/ self.world,
                //    /*the ray to trace*/ ray,
                //    /*prd*/ prd, OPTIX_RAY_FLAG_DISABLE_ANYHIT);

                unsigned int p0 = 0;
                unsigned int p1 = 0;
                packPointer(&prd, p0, p1);

                glm::vec3 col(0.f);

                for (;;) {
                    prd.wo = -ray.direction;
                    optixTrace(self.world, (const float3&) ray.origin, (const float3&) ray.direction, ray.tmin,
                        ray.tmax, 0, (OptixVisibilityMask) -1,
                        /*rayFlags     */ OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                        /*SBToffset    */ 0,
                        /*SBTstride    */ 2,
                        /*missSBTIndex */ 0, p0, p1);

                    
                    /*if (prd.depth > 0) {
                        col += prd.attenuation * prd.result;
                    } else {
                        col += prd.result;
                    }*/   

                    col += prd.emitted;
                    col += prd.radiance * prd.beta;

                    if (prd.done || prd.depth >= 2)
                        break;

                    ++prd.depth;

                    ray.origin = prd.origin;
                    ray.direction = prd.direction;
                }
                //col += prd.emitted;
                return glm::vec4(col.x, col.y, col.z, 1.0f);
                //return glm::vec4(prd.radiance, 1.0f);


                // if (prd.primID == -1) {
                //    // miss...
                //    // return attenuation * ambientLight;
                //    return bg;
                //}

                //// const glm::vec4 albedo(0.f, 1.f, 0.f, 1.f);
                // const glm::vec4 albedo = prd.albedo;
                // auto const factor = (.2f + .6f * fabsf(glm::dot(prd.N, ray.direction)));
                // return albedo * glm::vec4(factor, factor, factor, 1.f);
            }

            MM_OPTIX_RAYGEN_KERNEL(raygen_program)() {
                // printf("RAYGEN1\n");
                const RayGenData& self = getProgramData<RayGenData>();
                auto const index = optixGetLaunchIndex();
                glm::ivec2 pixelID = glm::ivec2(index.x, index.y);
                // const owl::vec2i pixelID = owl::getLaunchIndex();
                // const owl::vec2i launchDim = owl::getLaunchDims();

                if (pixelID.x >= self.fbSize.x)
                    return;
                if (pixelID.y >= self.fbSize.y)
                    return;
                const int pixelIdx = pixelID.x + self.fbSize.x * pixelID.y;

                const FrameState* fs = &self.frameStateBuffer[0];

                auto frame_idx = self.colorBufferPtr[pixelIdx].w;
                if (fs->changed) {
                    frame_idx = 0.0f;
                    self.colorBufferPtr[pixelIdx].w = 0.0f;
                }
                auto const old_col = self.colorBufferPtr[pixelIdx];

                unsigned int seed = tea<16>(pixelID.y * 4200 + pixelID.x, frame_idx);

                

                
                glm::vec4 col(0.f);
                glm::vec4 bg = fs->background;

                // printf("RAYGEN FS %f\n", fs->near);

                auto i = 4;

                do {

                    PerRayData prd;
                    prd.depth = 0;

                    prd.radiance = glm::vec3(0.f);
                    prd.pdf = 1.0f;

                    prd.beta = glm::vec3(1.f);

                    prd.seed = seed;
                    prd.done = false;

                    prd.world = self.world;

                    prd.countEmitted = true;
                    prd.emitted = glm::vec3(0.f);


                    // Random rnd(pixelIdx, 0);

                    float u = -fs->rw + (fs->rw + fs->rw) * float(pixelID.x + rnd(seed)) / self.fbSize.x;
                    float v = -(fs->th + (-fs->th - fs->th) * float(pixelID.y + rnd(seed)) / self.fbSize.y);
                    /*float u = -fs->rw + (fs->rw + fs->rw) * float(pixelID.x) / self.fbSize.x;
                    float v = -(fs->th + (-fs->th - fs->th) * float(pixelID.y) / self.fbSize.y);*/
                    auto ray = generateRay(*fs, u, v);

                    prd.origin = ray.origin;
                    prd.direction = ray.direction;

                    prd.lpos = ray.origin;
                    prd.ldir = fs->camera_screen_00;

                    col += traceRay(self, ray /*, rnd*/, prd, bg);
                } while (--i);
                col /= 4.0f;
                col.w = frame_idx + 1;
                //++col.w;

                // printf("RAYGEN2\n");

                // for (int s = 0; s < fs->samplesPerPixel; ++s) {
                //     float u = -fs->rw + (fs->rw+fs->rw)*float(pixelID.x + rnd())/self.fbSize.x;
                //     float v = -(fs->th + (-fs->th-fs->th)*float(pixelID.y + rnd())/self.fbSize.y);
                //     //float u = float(pixelID.x + rnd());
                //     //float v = float(pixelID.y + rnd());
                //     owl::Ray ray = generateRay(*fs, u, v);
                //     col += owl::vec4f(traceRay(self, ray, rnd, prd), 1);
                // }
                // col = col / float(fs->samplesPerPixel);

                // uint32_t rgba = owl::make_rgba(col);
                // self.colorBufferPtr[pixelIdx] = rgba;

                if (frame_idx > 0) {
                    const float a = 1.0f / static_cast<float>(frame_idx + 1);
                    col = glm::vec4(lerp(glm::vec3(old_col), glm::vec3(col), a), col.w);
                    //col.w = frame_idx + 1;
                }
                self.colorBufferPtr[pixelIdx] = col;
            }
        } // namespace device
    }     // namespace optix
} // namespace hpg
} // namespace megamol
