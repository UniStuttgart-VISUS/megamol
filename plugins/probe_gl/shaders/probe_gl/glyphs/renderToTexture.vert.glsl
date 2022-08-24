#version 450

#include "probe_gl/glyphs/extensions.inc.glsl"
#include "probe_gl/glyphs/scalar_probe_struct.inc.glsl"

layout(std430, binding = 0) readonly buffer MeshShaderParamsBuffer { MeshShaderParams[] mesh_shader_params; };

uniform mat4 view_mx;
uniform mat4 proj_mx;

layout(location = 0) flat out int draw_id;
layout(location = 1) out vec2 uv_coords;
layout(location = 2) out vec3 pixel_vector;
layout(location = 3) out vec3 cam_vector;


void main()
{
    const vec4 vertices[6] = vec4[6]( vec4( -1.0,-1.0,0.0,0.0 ),
                                      vec4( 1.0,1.0,1.0,1.0 ),
                                      vec4( -1.0,1.0,0.0,1.0 ),
                                      vec4( 1.0,1.0,1.0,1.0 ),
                                      vec4( -1.0,-1.0,0.0,0.0 ),
                                      vec4( 1.0,-1.0,1.0,0.0 ) );

    vec4 vertex = vertices[gl_VertexID];
    gl_Position =  vec4(vertex.xy, -1.0, 1.0);

    draw_id = 0;
    uv_coords = vertex.zw;
    cam_vector = mesh_shader_params[gl_DrawIDARB].probe_direction.xyz;

}
