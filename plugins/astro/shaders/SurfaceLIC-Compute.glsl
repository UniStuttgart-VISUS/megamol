#extension GL_ARB_compute_shader : enable

/* near and far clipping planes */
uniform float cam_near;
uniform float cam_far;

/* render target resolution */
uniform vec2 rt_resolution;

/* world space extents */
uniform vec3 origin;
uniform vec3 resolution;

/* noise properties */
uniform int noise_bands;
uniform float noise_scale;

/* streamline arc length */
uniform float arc_length;

/* number of advection steps */
uniform int num_advections;

/* depth threshold */
uniform float epsilon;

/* velocity coloring */
uniform int coloring;

/* maximum velocity magnitude */
uniform float max_magnitude;

/* input textures */
uniform highp sampler2D depth_tx2D;
uniform highp sampler2D velocity_tx2D;
uniform highp sampler2D position_tx2D;
uniform highp sampler2D normal_tx2D;
uniform highp sampler3D noise_tx3D;
uniform highp sampler1D tf_tx1D;

/* output image */
layout(rgba32f, binding = 0) writeonly uniform highp image2D render_target;

/* linearize depth value */
float linearize(float depth) {
    // Reconstruct clip space coordinates
    const vec3 clip_pos = vec3(0.0f, 0.0f, depth * 2.0f - 1.0f);

    // Inverse transform to view space
    vec4 view_pos = inverse(proj_mx) * vec4(clip_pos, 1.0f);
    view_pos /= view_pos.w;

    return (-view_pos.z - cam_near) / (cam_far - cam_near);
}

/* get velocity magnitude for selected coloring mode */
float velocity_magnitude(vec2 screen_pos) {
    const vec3 velocity = texture(velocity_tx2D, screen_pos).xyz;

    if (coloring == 0) {
        // Return stored original magnitude
        return velocity.z;
    } else if (coloring == 1) {
        // Return projected magnitude
        return length(velocity.xy);
    } else if (coloring == 2) {
        // Return difference between original and projected magnitude
        return velocity.z - length(velocity.xy);
    }

    return 0.0f;
}

/* blocks for computation */
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

void main() {
    // Get pixel coordinates
    vec3 gID = gl_GlobalInvocationID.xyz;

    if (gID.x >= rt_resolution.x || gID.y >= rt_resolution.y) return;

    const ivec2 pixel_coords = ivec2(gID.xy);
    vec2 pixel_tex_coords = pixel_coords / rt_resolution;

    // Check for surface at this pixel
    vec4 color = vec4(0.0f);
    float depth = texture(depth_tx2D, pixel_tex_coords).x;

    if (depth >= 0.0f && depth < 1.0f) {
        // Get position and normal at the current pixel
        const vec3 normal = texture(normal_tx2D, pixel_tex_coords).xyz;
        const vec3 coloring = texture(tf_tx1D, velocity_magnitude(pixel_tex_coords) / max_magnitude).xyz;
        const vec4 world_pos = screen_to_world_space(pixel_tex_coords, depth);

        // Advect and integrate
        float value = 0.0f;
        float weight = 0.0f;
        float last_depth = depth;

        const float step_size = arc_length / num_advections;

        for (int steps = 0; steps < num_advections; ++steps) {
            // Get screen-space velocity and perform advection
            vec2 velocity = texture(velocity_tx2D, pixel_tex_coords).xy;

            if (length(velocity) == 0.0f) {
                break;
            }

            pixel_tex_coords += step_size * normalize(velocity);

            // Check depth
            depth = texture(depth_tx2D, pixel_tex_coords).x;

            if (abs(linearize(depth) - linearize(last_depth)) > epsilon) {
                break;
            }

            last_depth = depth;

            // Compute noise for the current position
            const vec4 world_coords = screen_to_world_space(pixel_tex_coords, depth);
            const vec3 noise_coords = (world_coords.xyz - origin) / resolution;

            const float center_scale_exact = log2(linearize(depth));
            const float center_scale = ceil(center_scale_exact);
            const float center_scale_deviation = abs(center_scale_exact - center_scale);

            const int lower_scale = int(center_scale - floor(noise_bands / 2.0f));
            const int higher_scale = int(center_scale + floor((noise_bands - 1) / 2.0f));

            float noise = 0.0f;
            float noise_weight = 0.0f;

            for (int i = lower_scale; i <= higher_scale; ++i) {
                const float alpha = exp(-1.0f / (1.0f - pow((1.0f / (2.0f * (higher_scale - lower_scale))) *
                                                                (i - center_scale + center_scale_deviation),
                                                            2.0f)));

                noise += alpha * texture(noise_tx3D, noise_coords * noise_scale * (1.0f / exp2(float(i)))).x;
                noise_weight += alpha;
            }

            noise /= noise_weight;

            // Aggregate noise weighted by streamline length
            const float rbf_weight =
                exp(-1.0f / (1.0f - pow((1.0f / (2.0f * arc_length)) * (steps * step_size), 2.0f)));

            value += noise * rbf_weight;
            weight += rbf_weight;
        }

        value /= weight;

        // Set color
        const vec3 inv_view_dir = (inverse(view_mx) * vec4(0.0f, 0.0f, 0.0f, 1.0f) - world_pos).xyz;
        const vec3 light_dir = light - world_pos.xyz;

        color = vec4(phong(value * coloring, normal, inv_view_dir, light_dir), 1.0f);
    }

    imageStore(render_target, pixel_coords, color);
}
