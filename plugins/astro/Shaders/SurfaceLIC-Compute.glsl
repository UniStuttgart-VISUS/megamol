#extension GL_ARB_compute_shader : enable

/* matrices */
uniform mat4 view_mx;
uniform mat4 proj_mx;

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
uniform highp sampler3D velocity_tx3D;
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

/* transform screen to world space coordinates */
vec4 screen_to_world_space(vec2 screen_pos, float depth) {
    // Reconstruct clip space coordinates
    const vec3 clip_space = vec3(screen_pos, depth) * 2.0f - vec3(1.0f);

    // Inverse transform to world space
    vec4 world_pos = inverse(proj_mx * view_mx) * vec4(clip_space, 1.0f);
    world_pos /= world_pos.w;

    return world_pos;
}

/* transform world to screen space coordinates */
vec3 world_to_screen_space(vec4 world_pos) {
    // Transform to clip space
    vec4 clip_pos = (proj_mx * view_mx) * world_pos;
    clip_pos /= clip_pos.w;

    // Construct screen pos
    vec3 screen_pos = (clip_pos.xyz + vec3(1.0f)) / 2.0f;

    return screen_pos;
}

/* get velocity magnitude for selected coloring mode */
float velocity_magnitude(vec4 world_pos, vec2 screen_pos) {
    if (coloring == 0) {
        return length(texture(velocity_tx3D, (world_pos.xyz - origin) / resolution).xyz);
    } else if (coloring == 1) {
        const vec3 normal = texture(normal_tx2D, screen_pos).xyz;

        vec3 velocity = texture(velocity_tx3D, (world_pos.xyz - origin) / resolution).xyz;
        velocity -= dot(velocity, normal) * normal;

        return length(velocity);
    } else if (coloring == 2) {
        const vec3 normal = texture(normal_tx2D, screen_pos).xyz;

        vec3 velocity = texture(velocity_tx3D, (world_pos.xyz - origin) / resolution).xyz;
        velocity -= dot(velocity, normal) * normal;

        return length(texture(velocity_tx3D, (world_pos.xyz - origin) / resolution).xyz) - length(velocity);
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
        // Get position in world space
        vec4 world_pos = screen_to_world_space(pixel_tex_coords, depth);

        // Advect and integrate
        float value = 0.0f;
        float distance = 0.0f;
        float weight = 0.0f;
        float last_depth = depth;

        const float max_arc_length = arc_length * max(max(resolution[0], resolution[1]), resolution[2]);
        const float step_size = max_arc_length / num_advections;

        for (int steps = 0; steps < num_advections; ++steps) {
            // Project velocity onto the surface
            const vec3 normal = texture(normal_tx2D, pixel_tex_coords).xyz;

            vec3 velocity = texture(velocity_tx3D, (world_pos.xyz - origin) / resolution).xyz;
            velocity -= dot(velocity, normal) * normal;

            if (length(velocity) == 0.0f) {
                break;
            }

            const vec4 new_world_pos = vec4(world_pos.xyz + step_size * normalize(velocity), 1.0f);

            // Map to screen space
            pixel_tex_coords = world_to_screen_space(new_world_pos).xy;

            // Check depth
            depth = texture(depth_tx2D, pixel_tex_coords).x;

            if (abs(linearize(depth) - linearize(last_depth)) > epsilon) {
                break;
            }

            // Integrate and project endpoint onto surface
            distance += length(new_world_pos - world_pos);

            const float sigma = 1.0f;
            const float gaussian_weight =
                exp(-1.0f / (1.0f - pow((1.0f / (2.0f * sigma * max_arc_length)) * distance, 2.0f)));

            value += texture(noise_tx2D, pixel_tex_coords).x * gaussian_weight;
            weight += gaussian_weight;

            world_pos = screen_to_world_space(pixel_tex_coords, depth);

            last_depth = depth;
        }

        value /= weight;

        // Set color
        const vec4 intensity = vec4(vec3(0.299 * color.x + 0.587 * color.y + 0.114 * color.z), 0.0f);

        color = intensity + value * texture(tf_tx1D, velocity_magnitude(world_pos, pixel_tex_coords) / max_magnitude);
    }

    imageStore(render_target, pixel_coords, color);
}
