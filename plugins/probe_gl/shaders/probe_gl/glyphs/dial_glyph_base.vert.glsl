#version 450

#include "probe_gl/glyphs/extensions.inc.glsl"
#include "probe_gl/glyphs/per_frame_data_struct.inc.glsl"
#include "probe_gl/glyphs/base_probe_struct.inc.glsl"
#include "probe_gl/glyphs/dial_glyph_utility.inc.glsl"
#include "probe_gl/glyphs/dial_glyph_constants.inc.glsl"

layout(std430, binding = 0) readonly buffer MeshShaderParamsBuffer { MeshShaderParams[] mesh_shader_params; };

uniform mat4 view_mx;
uniform mat4 proj_mx;

layout(location = 0) flat out int draw_id;
layout(location = 1) out vec2 uv_coords;
layout(location = 2) out vec3 pixel_vector;

void main()
{
    draw_id = gl_DrawIDARB;
    uv_coords = vertices[gl_VertexID].zw;

    vec3 probe_direction = normalize( mesh_shader_params[draw_id].probe_direction.xyz );
    vec3 cam_front = normalize(transpose(mat3(view_mx)) * vec3(0.0,0.0,-1.0));

    // initialize glyph right and up to camera plane
    vec3 glyph_up = normalize(transpose(mat3(view_mx)) * vec3(0.0,1.0,0.0));
    vec3 glyph_right= normalize(transpose(mat3(view_mx)) * vec3(sign(dot(probe_direction, cam_front)) * 1.0,0.0,0.0));

    // compute world space pixel vector
    vec2 pixel_coords = uv_coords * 2.0 - 1.0;
    pixel_vector = normalize( pixel_coords.x * glyph_right + pixel_coords.y * glyph_up );

    // tilt glyph to be orthognal to probe direction
    tiltGlyph(glyph_right,glyph_up,probe_direction,cam_front);

    vec4 glyph_pos = vec4(mesh_shader_params[gl_DrawIDARB].glpyh_position.xyz, 1.0);
    vec2 bboard_vertex = vertices[gl_VertexID].xy;
    glyph_pos.xyz = glyph_pos.xyz + (glyph_up * bboard_vertex.y * mesh_shader_params[gl_DrawIDARB].scale) + (glyph_right * bboard_vertex.x * mesh_shader_params[gl_DrawIDARB].scale);

    gl_Position = (proj_mx * view_mx * glyph_pos);
}
