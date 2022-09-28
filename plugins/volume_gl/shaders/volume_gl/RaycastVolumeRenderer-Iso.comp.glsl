#version 430

#include "common/RaycastVolumeRenderer-Input.inc.glsl"
#include "common/RaycastVolumeRenderer-Output.inc.glsl"
#include "common/RaycastVolumeRenderer-Functions.inc.glsl"
#include "core/phong.inc.glsl"

/* isovalue used for isosurface reconstruction */
uniform float isoValue;

/* opacity */
uniform float opacity;

/* output normal */
layout(rgba32f, binding = 1) writeonly uniform highp image2D normal_target_tx2D;

/* output depth */
layout(r32f, binding = 2) writeonly uniform highp image2D depth_target_tx2D;

/* inputs */
uniform highp sampler2D color_tx2D;
uniform highp sampler2D depth_tx2D;
uniform int use_depth_tx;

/* main function for computation */
void compute(float t, const float tfar, const Ray ray, const float rayStep, const ivec2 pixel_coords) {
    // Initialize results
    vec4 result = vec4(0.0f);
    bool have_hit = false;

    // Initialize output depth and normal value
    float depth = FLT_MAX;
    vec3 normal = vec3(0.0f);

    vec2 pixel_tex_coords = vec2(pixel_coords.x / rt_resolution.x, pixel_coords.y / rt_resolution.y);
    pixel_tex_coords += vec2(0.5/rt_resolution.x, 0.5/rt_resolution.y);

    // input values
    const float input_depth = texture(depth_tx2D, pixel_tex_coords).x;
    const vec4 input_color = texture(color_tx2D, pixel_tex_coords);

    // Store value and position from previous step
    vec3 old_pos = ray.o + t * ray.d;
    float old_value = 0.0f;

    while (t < tfar) {
        vec3 pos = ray.o + t * ray.d;

        // Compute volume tex coordinates in [0,1] range.
        vec3 texCoords = (pos - boxMin) / (boxMax - boxMin);
        texCoords *= 1.0 - 2.0 * halfVoxelSize;
        texCoords += halfVoxelSize;

        // Get volume sample
        float vol_sample = texture(volume_tx3D, texCoords).x;

        if (vol_sample > isoValue) {
            // Compute relative position between sample positions
            const vec3 direction = pos - old_pos;
            const float distance = (isoValue - old_value) / (vol_sample - old_value);

            const vec3 surface_pos = old_pos + distance * direction;

            depth = calculate_depth(surface_pos);

            if (use_depth_tx != 0) {
                // Compare depth values and decide to abort
                if (depth > input_depth) {
                    break;
                }
            }

            // Compute normal
            normal = calculate_normal(texCoords);

            have_hit = true;

            // Compute illumination from fixed light
            if (use_lighting) {
                result = vec4(phong(material_col, normal, -ray.d, -light), opacity);
            } else {
                result = vec4(material_col, opacity);
            }
            if (use_depth_tx != 0) {
                result.rgb = result.rgb * result.a + (1.0f - result.a) * input_color.rgb;
                result.a += input_color.a;
            }

            break;
        }

        // Store value and position for new "previous" step
        old_pos = pos;
        old_value = vol_sample;

        // Adaptive step size
        if (vol_sample / isoValue < 0.5f) {
            t += rayStep;
        } else {
            t += rayStep * (1.0f + (rayStep / 10.0f) - vol_sample / isoValue);
        }
    }

    if (!have_hit) {
        result = input_color;
        normal = vec3(0.0f);
        depth = input_depth;
    }    

    // Write results
    imageStore(render_target_tx2D, pixel_coords, result);
    imageStore(normal_target_tx2D, pixel_coords, vec4(normal, 1.0f));
    imageStore(depth_target_tx2D, pixel_coords, vec4(depth));
}

/* function for storing default output values */
void storeDefaults(const ivec2 pixel_coords) {
    imageStore(render_target_tx2D, pixel_coords, vec4(0.0f));
    imageStore(normal_target_tx2D, pixel_coords, vec4(0.0f));
    imageStore(depth_target_tx2D, pixel_coords, vec4(FLT_MAX));
}

#include "common/RaycastVolumeRenderer-Main.inc.glsl"
