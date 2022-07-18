#version 430

#include "common/RaycastVolumeRenderer-Input.inc.glsl"
#include "common/RaycastVolumeRenderer-Output.inc.glsl"
#include "common/RaycastVolumeRenderer-Functions.inc.glsl"

/* opacity threshold for integration */
uniform float opacityThreshold;

/* texture containing transfer function */
uniform highp sampler1D tf_tx1D;

/* texture containing a depth buffer */
uniform highp sampler2D color_tx2D;
uniform highp sampler2D depth_tx2D;
uniform int use_depth_tx;

/* main function for computation */
void compute(float t, const float tfar, const Ray ray, const float rayStep, const ivec2 pixel_coords) {
    // Get pixel texture coordinates and depth value at original position
    vec2 pixel_tex_coords = vec2(pixel_coords.x / rt_resolution.x, pixel_coords.y / rt_resolution.y);
    const float input_depth = texture(depth_tx2D, pixel_tex_coords).x;

    // Initialize results
    vec4 result = vec4(0.0f);

    float att = 0.0f;

    while (t < tfar) {
        vec3 pos = ray.o + t * ray.d;

        // Compute volume tex coordinates in [0,1] range.
        vec3 texCoords = (pos - boxMin) / (boxMax - boxMin);
        texCoords *= 1.0 - 2.0 * halfVoxelSize;
        texCoords += halfVoxelSize;

        if (use_depth_tx != 0) {
            // Compare depth values and decide to abort
            const float depth = calculate_depth(pos);

            if (depth > input_depth) {
                const vec4 color = texture(color_tx2D, pixel_tex_coords);

                result = color;

                break;
            }
        }

        // Get sample
        att += (texture(volume_tx3D, texCoords).x - valRange.x) / (valRange.y - valRange.x);

        t += rayStep;
    }

    // Write results
    result = result.w * result + background * (1.0f - result.w);

    imageStore(render_target_tx2D, pixel_coords, vec4(result.xyz, att));
}

/* function for storing default output values */
void storeDefaults(const ivec2 pixel_coords) {
    imageStore(render_target_tx2D, pixel_coords, vec4(0.0f));
}

#include "common/RaycastVolumeRenderer-Main.inc.glsl"
