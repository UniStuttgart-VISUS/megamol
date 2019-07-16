#extension GL_ARB_compute_shader: enable
#define FLT_MAX 3.402823466e+38
#define FLT_MIN 1.175494351e-38

/* matrix that transforms a set of coordinates from world space to volume texture space */
uniform mat4 volume_model_mx;
/* camera inverse view projection matrix */
//uniform mat4 camera_inv_view_proj_mx;

uniform mat4 view_mx;
uniform mat4 proj_mx;

/* render targete resolution*/
uniform vec2 rt_resolution;
/**/
uniform vec3 boxMin;
/**/
uniform vec3 boxMax;
/**/
uniform float voxelSize;
/**/
uniform vec3 halfVoxelSize;
uniform float rayStepRatio;
/**/
uniform float opacityThreshold;

/*	texture that houses the volume data */
uniform highp sampler3D volume_tx3D;
/* texture containing scene depth */
uniform highp sampler2D depth_tx2D;
/* texture containing transfer function */
uniform highp sampler2D transfer_function_tx2D;

layout(rgba32f, binding = 0) writeonly uniform highp image2D render_target_tx2D;

struct Ray {
    vec3 o;
    vec3 d;
};

bool intersectBox(Ray r, vec3 boxmin, vec3 boxmax, out float tnear, out float tfar) {
    vec3 invR = vec3(1.0f) / r.d;
    vec3 tbot = invR * (boxmin - r.o);
    vec3 ttop = invR * (boxmax - r.o);

    // Special case for a ray lying in a bounding plane.
    if (r.d.x == 0.0f && r.o.x == boxmax.x) {
        ttop.x = -FLT_MAX;
        tbot.x = FLT_MAX;
    }
    if (r.d.y == 0.0f && r.o.y == boxmax.y) {
        ttop.y = -FLT_MAX;
        tbot.y = FLT_MAX;
    }
    if (r.d.z == 0.0f && r.o.z == boxmax.z) {
        ttop.z = -FLT_MAX;
        tbot.z = FLT_MAX;
    }

    vec3 tmin = min(ttop, tbot);
    vec3 tmax = max(ttop, tbot);

    float largest_tmin = max(max(tmin.x, tmin.y), max(tmin.x, tmin.z));
    float smallest_tmax = min(min(tmax.x, tmax.y), min(tmax.x, tmax.z));

    tnear = largest_tmin;
    tfar = smallest_tmax;

    return smallest_tmax > largest_tmin;
}

/////////////////////////////// Random Number Generator
float wang_hash(uint seed) {
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);

    return float(seed) / 4294967296.0;
}
/////////////////////////////// End - Random Number Generator

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

void main() {
    vec3 gID = gl_GlobalInvocationID.xyz;

    if (gID.x >= rt_resolution.x || gID.y >= rt_resolution.y) return;

    ivec2 pixel_coords = ivec2(gID.xy);

    vec2 clip_space_pixel_coords =
        vec2((gID.x / rt_resolution.x) * 2.0f - 1.0f, (gID.y / rt_resolution.y) * 2.0f - 1.0f);

    Ray ray;
    // Unproject a point on the near plane and use as an origin.
    mat4 inv_view_proj_mx = inverse(proj_mx * view_mx);
    vec4 unproj = inv_view_proj_mx * vec4(clip_space_pixel_coords, -1.0f, 1.0f);
    ray.o = unproj.xyz / unproj.w;
    // Unproject a point at the same pixel, but further away from the near plane
    // to compute a ray direction in world space.
    unproj = inv_view_proj_mx * vec4(clip_space_pixel_coords, 0.0f, 1.0f);
    ray.d = normalize((unproj.xyz / unproj.w) - ray.o);

    // Just for safety.
    // Box diagonal is sqrt(3) times longer.
    float rayStep = voxelSize * rayStepRatio;
    vec3 box_range = boxMax - boxMin;
    float max_bbox_length = max(max(box_range.x, box_range.y), box_range.z);
    //int maxSteps = int((1.0f / rayStep) * 1.74f * 2.0f * max_bbox_length); // todo

    // Generate a random value in [0, 1] range.
    // float randomTemp = sin(dot(vec2(pixel_coords.y * rt_resolution.x + pixel_coords.x), vec2(12.9898f, 78.233f))) *
    // 43758.5453f; float random = abs(randomTemp - floor(randomTemp));
    float random = wang_hash(pixel_coords.x + pixel_coords.y * uint(rt_resolution.x));

    float tnear, tfar;

    // Require tnear or tfar to be positive, so that we can renderer from inside the box,
    // but do not render if the box is completely behind the camera.
    if (intersectBox(ray, boxMin, boxMax, tnear, tfar) && (tnear > 0.0f || tfar > 0.0f)) {
        float t = tnear >= 0.0f ? tnear : 0.0f;
        t += random * rayStep; // Randomly offset the ray origin to prevent ringing artifacts
        vec4 result = vec4(0.0f);
        int steps = 0;

        while (t < tfar && result.w < opacityThreshold /*&& steps < maxSteps*/) {
            vec3 pos = ray.o + t * ray.d;
            // Compute volume tex coordinates in [0,1] range.
            vec3 texCoords = (pos - boxMin) / (boxMax - boxMin);
            texCoords *= 1.0 - 2.0 * halfVoxelSize;
            texCoords += halfVoxelSize;

            vec4 vol_sample = texture(transfer_function_tx2D, vec2(texture(volume_tx3D, texCoords).x, 1));
            // vec4 vol_sample = texture(volume_tx3D,texCoords);
            // vol_sample.w = vol_sample.x;

            // Opacity correction.
            vol_sample.w = (1.0f - pow(1.0f - vol_sample.w, rayStepRatio));
            // TF "Brightness". Make sure to not over-saturate the opacity.
            // (Which will lead to color oversaturation.)
            // vol_sample.w = min(vol_sample.w * cSeriesDesc.BrightnessPerSeries[series], 1.0f)
            vol_sample.xyz *= vol_sample.w;
            // if (useLighting)
            //{
            //    vec3 gradient = fetchGradientEstimate(cVolumesToRender.List[series],
            //                                            cSeriesDesc.ComponentsPerSeries[series],
            //                                            cTfs.List[series],
            // texCoords,
                //                                            rayStep * 8.0f);
                //    float3 normal = normalize(-gradient);
                //    float3 lightDir = normalize(cLightDesc.Pos - pos);
                //    float lambert = max(0.0f, dot(normal, lightDir));
                //    float3 lightColorContribution = cLightDesc.Color * cLightDesc.Intensity * lambert;
                //    float3 surfaceColor = make_float3(mappedSample)
                //    mappedSample += make_float4(surfaceColor * lightColorContribution, 0.0f);
                //}

                result += (1.0f - result.w) * vol_sample;

            steps++;
            t += rayStep;
        }

        // Blend with white background. (Helps to make the renderings look more consistent.)
        // todo Is this correct? What if bg was transparent? The result would change with this formula.
        // result = (result.w) * result + vec4(1.0f,0.0,0.0,0.0) * (1.0f - result.w);
        // result.w = 1.0f;
        imageStore(render_target_tx2D, pixel_coords, result);

        // debug
        // imageStore(render_target_tx2D,pixel_coords,vec4(1.0));
    } else {
        // Always write out to make sure that data from the previous frame is overwritten.
        imageStore(render_target_tx2D, pixel_coords, vec4(0.0));
    }
}