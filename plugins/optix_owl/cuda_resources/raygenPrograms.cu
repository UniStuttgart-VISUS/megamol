#include <cuda_runtime.h>
#include <optix_device.h>

#include <owl/owl_device.h>
#include <owl/common/math/vec.h>
#include <owl/common/math/random.h>

#include "raygen.h"
#include "perraydata.h"

namespace megamol {
namespace optix_owl {
namespace device {
using namespace owl::common;
using Random = owl::common::LCG<16>;

extern "C" __global__ void __miss__miss() {}

inline __device__ vec3f safe_normalize(const vec3f& v) {
    const vec3f vv = vec3f(
        (fabsf(v.x) < 1e-8f) ? 1e-8f : v.x, (fabsf(v.y) < 1e-8f) ? 1e-8f : v.y, (fabsf(v.z) < 1e-8f) ? 1e-8f : v.z);
    return normalize(vv);
}

static __device__ owl::Ray generateRay(const FrameState& fs, float s, float t, Random& rnd) {
    const vec3f origin = fs.camera_lens_center;
    //const vec3f direction = fs.camera_screen_00 + s * fs.camera_screen_du + t * fs.camera_screen_dv;

    /*const vec3f p = s * fs.camera_screen_du + t * fs.camera_screen_dv + fs.camera_screen_dz;
    const vec3f direction = p - origin;*/

    const vec3f direction = fs.near_plane* fs.camera_screen_dz + s * fs.camera_screen_du + t * fs.camera_screen_dv;


    return owl::Ray(
        // return optix::make_Ray(
        /* origin   : */ origin,
        /* direction: */ safe_normalize(direction),
        /* tmin     : */ 1e-6f,
        /* tmax     : */ 1e20f); //RT_DEFAULT_MAX);
}

inline __device__ int32_t make_8bit(const float f) {
    return min(255, max(0, int(f * 256.f)));
}

inline __device__ int32_t make_rgba8(const vec4f color) {
    return (make_8bit(color.x) << 0) + (make_8bit(color.y) << 8) + (make_8bit(color.z) << 16) +
           (make_8bit(color.w) << 24);
}

#define RANDVEC3F vec3f(rnd(), rnd(), rnd())

inline __device__ vec3f random_in_unit_sphere(Random& rnd) {
    vec3f p;
    do {
        p = 2.0f * RANDVEC3F - vec3f(1, 1, 1);
    } while (dot(p, p) >= 1.0f);
    return p;
}
    

inline __device__ vec3f traceRay(const RayGenData& self, owl::Ray& ray, Random& rnd, PerRayData& prd) {
    const bool lastFieldOfParticleIsScalarValue = false;

    vec3f attenuation = 1.f;
    vec3f ambientLight(.8f);

    /* iterative version of recursion, up to depth 50 */
    for (int depth = 0; true; depth++) {
        prd.particleID = -1;

        owl::traceRay(/*accel to trace against*/ self.world,
            /*the ray to trace*/ ray,
            // /*numRayTypes*/1,
            /*prd*/ prd, OPTIX_RAY_FLAG_DISABLE_ANYHIT);
        if (prd.particleID == -1) {
            // miss...
            return attenuation * ambientLight;
        }

        //const Particle particle = self.particleBuffer[prd.particleID];
        vec3f N = (ray.origin + prd.t * ray.direction) - prd.pos;
        // printf("normal %f %f %f\n",N.x,N.y,N.z);
        if (dot(N, (vec3f) ray.direction) > 0.f)
            N = -N;
        N = normalize(N);

        // hardcoded albedo for now:
#if COLOR_CODING
        const vec3f albedo = randomColor(prd.treeletID);
#else
        //const vec3f albedo
        //  = //prd.primID == 0 ? vec3f(.1,.6,.3) :
        //  (lastFieldOfParticleIsScalarValue)
        //  ? transferFunction(.1f*sqrtf(particle.fieldValue))
        //  : randomColor(1+particle.matID);
        const vec3f albedo = randomColor(0);
#endif
        // hard-coded for the 'no path tracing' case:
        if (self.rec_depth == 0)
            return albedo * (.2f + .6f * fabsf(dot(N, (vec3f) ray.direction)));


        attenuation *= albedo;

        if (depth >= self.rec_depth) {
            // ambient term:
            return 0.1f;
        }

        const vec3f scattered_origin = ray.origin + prd.t * ray.direction;
        const vec3f scattered_direction = N + random_in_unit_sphere(rnd);
        ray = owl::Ray(/* origin   : */ scattered_origin,
            /* direction: */ safe_normalize(scattered_direction),
            /* tmin     : */ 1e-3f,
            /* tmax     : */ 1e+8f);
    }
}

extern "C" __global__ void __raygen__raygen() {

    const RayGenData& self = owl::getProgramData<RayGenData>();
    const vec2i pixelID = owl::getLaunchIndex();
    const vec2i launchDim = owl::getLaunchDims();

    if (pixelID.x >= self.fbSize.x)
        return;
    if (pixelID.y >= self.fbSize.y)
        return;
    const int pixelIdx = pixelID.x + self.fbSize.x * pixelID.y;
    const int pixel_index = pixelID.y * launchDim.x + pixelID.x;

    const FrameState* fs = self.frameStateBuffer;

    Random rnd(pixel_index, fs->accumID);

    PerRayData prd;

    vec4f col(0.f);

    for (int s = 0; s < fs->samplesPerPixel; ++s) {
        /*float u = float(pixelID.x + rnd() - self.fbSize.x / 2);
        float v = float(pixelID.y + rnd() - self.fbSize.y / 2);*/

        float u = -fs->rw + (fs->rw + fs->rw) * float(pixelID.x + rnd()) / self.fbSize.x;
        float v = -(fs->th + (-fs->th - fs->th) * float(pixelID.y + rnd()) / self.fbSize.y);

        owl::Ray ray = generateRay(*fs, u, v, rnd);
        col += vec4f(traceRay(self, ray, rnd, prd), 1);
    }
    col = col / float(fs->samplesPerPixel);

    if (fs->accumID > 0)
        col = col + (vec4f) self.accumBufferPtr[pixelIdx];
    self.accumBufferPtr[pixelIdx] = col;

    //printf("Rendering!\n");

    uint32_t rgba = make_rgba8(col / (fs->accumID + 1.f));
    //uint32_t rgba = make_rgba8(vec4f(1, 0, 0, 1));
    self.colorBufferPtr[pixelIdx] = rgba;
}
} // namespace device
} // namespace optix_owl
} // namespace megamol
