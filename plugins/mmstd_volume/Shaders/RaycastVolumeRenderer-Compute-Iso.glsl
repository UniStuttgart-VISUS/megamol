/* isovalue used for isosurface reconstruction */
uniform float isoValue;

/* output normal */
layout(rgba32f, binding = 1) writeonly uniform highp image2D normal_target_tx2D;

/* output depth */
layout(r32f, binding = 2) writeonly uniform highp image2D depth_target_tx2D;

/* main function for computation */
void compute(float t, const float tfar, const Ray ray, const float rayStep, const ivec2 pixel_coords) {
    // Initialize results
    vec4 result = vec4(0.0f);

    // Initialize output depth and normal value
    float depth = FLT_MAX;
    vec3 normal = vec3(0.0f);

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

            // Compute normal
            normal = calculate_normal(texCoords);

            // Compute illumination from fixed light
            if (use_lighting) {
                result = vec4(phong(material_col, normal, normalize(-surface_pos), normalize(-surface_pos)), 1.0f);
            } else {
                result = vec4(material_col, 1.0f);
            }

            break;
        }

        // Store value and position for new "previous" step
        old_pos = pos;
        old_value = vol_sample;

        t += rayStep;
    }

    // Write results
    result = result.w * result + vec4(background, 1.0f) * (1.0f - result.w);

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