#version 430

#include "commons.inc.glsl"

layout(triangles) in;
layout(line_strip, max_vertices = 4) out;

uniform mat4 view_mx;
uniform mat4 proj_mx;

in vec4 colors[];
in vec3 normals[];
in vec4 clip_planes[];
in int use_clip_planes[];
in int culling[];

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

        const vec3 face_normal =
            calculate_normal(gl_in[0].gl_Position, gl_in[1].gl_Position, gl_in[2].gl_Position);

        const bool front_face = is_front_face(view_mx, face_normal);

        if ((culling[0] == 0)                    // no culling
            || (culling[0] == 1 && front_face)   // back face culling
            || (culling[0] == 2 && !front_face)) // front face culling
        {
            gl_Position = proj_mx * view_mx * gl_in[0].gl_Position;
            color = colors[0];
            normal = normals[0];
            world_pos = gl_in[0].gl_Position;
            clip_plane = clip_planes[0];
            use_clip_plane = use_clip_planes[0];
            EmitVertex();

            gl_Position = proj_mx * view_mx * gl_in[1].gl_Position;
            color = colors[1];
            normal = normals[1];
            world_pos = gl_in[1].gl_Position;
            clip_plane = clip_planes[0];
            use_clip_plane = use_clip_planes[0];
            EmitVertex();

            gl_Position = proj_mx * view_mx * gl_in[2].gl_Position;
            color = colors[2];
            normal = normals[2];
            world_pos = gl_in[2].gl_Position;
            clip_plane = clip_planes[0];
            use_clip_plane = use_clip_planes[0];
            EmitVertex();

            gl_Position = proj_mx * view_mx * gl_in[0].gl_Position;
            color = colors[0];
            normal = normals[0];
            world_pos = gl_in[0].gl_Position;
            clip_plane = clip_planes[0];
            use_clip_plane = use_clip_planes[0];
            EmitVertex();

            EndPrimitive();
        }
    }
}
