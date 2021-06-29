layout(triangles) in;
layout(line_strip, max_vertices = 4) out;

uniform mat4 view_mx;
uniform mat4 proj_mx;

in vec4 colors[];

in vec4 clip_planes[];
in int use_clip_planes[];

out vec4 color;
out vec3 normal;
out vec4 world_pos;
out vec4 clip_plane;
flat out int use_clip_plane;

void main() {
    if (!(use_clip_planes[0] != 0
        && clip_halfspace(gl_in[0].gl_Position.xyz, clip_planes[0])
        && clip_halfspace(gl_in[1].gl_Position.xyz, clip_planes[0])
        && clip_halfspace(gl_in[2].gl_Position.xyz, clip_planes[0]))) {

        const vec3 calculated_normal =
            calculate_normal(view_mx, gl_in[0].gl_Position, gl_in[1].gl_Position, gl_in[2].gl_Position);

        gl_Position = proj_mx * view_mx * gl_in[0].gl_Position;
        color = colors[0];
        normal = calculated_normal;
        world_pos = gl_in[0].gl_Position;
        clip_plane = clip_planes[0];
        use_clip_plane = use_clip_planes[0];
        EmitVertex();

        gl_Position = proj_mx * view_mx * gl_in[1].gl_Position;
        color = colors[1];
        normal = calculated_normal;
        world_pos = gl_in[1].gl_Position;
        clip_plane = clip_planes[0];
        use_clip_plane = use_clip_planes[0];
        EmitVertex();

        gl_Position = proj_mx * view_mx * gl_in[2].gl_Position;
        color = colors[2];
        normal = calculated_normal;
        world_pos = gl_in[2].gl_Position;
        clip_plane = clip_planes[0];
        use_clip_plane = use_clip_planes[0];
        EmitVertex();

        gl_Position = proj_mx * view_mx * gl_in[0].gl_Position;
        color = colors[0];
        normal = calculated_normal;
        world_pos = gl_in[0].gl_Position;
        clip_plane = clip_planes[0];
        use_clip_plane = use_clip_planes[0];
        EmitVertex();

        EndPrimitive();
    }
}
