#extension GL_ARB_compute_shader : enable

/* near and far clipping planes */
uniform float cam_near;
uniform float cam_far;

/* render target resolution */
uniform vec2 rt_resolution;

/* world space extents */
uniform vec3 origin;
uniform vec3 resolution;

/* LIC stencil */
uniform int stencil;

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
uniform highp sampler2D color_tx2D;
uniform highp sampler2D depth_tx2D;
uniform highp sampler2D velocity_tx2D;
uniform highp sampler2D noise_tx2D;
uniform highp sampler2D normal_tx2D;
uniform highp sampler1D tf_tx1D;

/* output image */
layout(rgba32f, binding = 0) writeonly uniform highp image2D render_target;

/* linearize depth value */
float linearize(float depth) {
    const vec3 clip_space = vec3(0.5f, 0.5f, depth) * 2.0f - vec3(1.0f);

    return (depth - cam_near) / (cam_far - cam_near);
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

    const vec2 offset = vec2(0.5f * stencil - 0.5f);

    const ivec2 pixel_coords = ivec2(gID.xy);
    vec2 pixel_tex_coords = (pixel_coords - (pixel_coords % stencil) + offset) / rt_resolution;

    // Check for surface at this pixel
    vec4 color = texture(color_tx2D, pixel_tex_coords);
    float depth = texture(depth_tx2D, pixel_tex_coords).x;

    if (depth >= 0.0f && depth < 1.0f) {
        // Advect and integrate
        float value = 0.0f;
        float weight = 0.0f;
        float last_depth = depth;

        //const float max_arc_length = arc_length * max(max(resolution[0], resolution[1]), resolution[2]);
        const float max_arc_length = arc_length;
        const float step_size = max_arc_length / num_advections;

        for (int steps = 0; steps < num_advections; ++steps) {
            // Get screen-space velocity and perform advection
            vec2 velocity = texture(velocity_tx2D, pixel_tex_coords).xy;

            if (length(velocity) != 0.0f) {
                pixel_tex_coords += step_size * normalize(velocity);
            }

            // Check depth
            depth = texture(depth_tx2D, pixel_tex_coords).x;

            if (abs(linearize(depth) - linearize(last_depth)) > epsilon) {
                break;
            }

            last_depth = depth;

            // Aggregate noise
            const float sigma = 1.0f;
            const float rbf_weight =
                exp(-1.0f / (1.0f - pow((1.0f / (2.0f * sigma * max_arc_length)) * (steps * step_size), 2.0f)));

            value += texture(noise_tx2D, pixel_tex_coords).x * rbf_weight;
            weight += rbf_weight;
        }

        value /= weight;

        // Set color
        //const vec3 normal = texture(normal_tx2D, pixel_tex_coords).xyz; // TODO
        const vec4 intensity = vec4(vec3(0.299 * color.x + 0.587 * color.y + 0.114 * color.z), 0.0f);

        color = intensity + value * texture(tf_tx1D, velocity_magnitude(pixel_tex_coords) / max_magnitude);
    }

    imageStore(render_target, pixel_coords, color);
}
