//layout(quads, equal_spacing, ccw) in;
layout(quads) in;

layout(location = 0) in vec3 world_pos[];
layout(location = 1) in vec3 normal[];

layout(location = 0) out vec3 world_pos_out;
layout(location = 1) out vec3 normal_out;
layout(location = 2) out vec3 debug_col_out;

uniform mat4 view_mx;
uniform mat4 proj_mx;

void main()
{
    vec3 p_0 = mix(world_pos[1],world_pos[0],gl_TessCoord.x);
    vec3 p_1 = mix(world_pos[2],world_pos[3],gl_TessCoord.x);
    world_pos_out = mix(p_0, p_1, gl_TessCoord.y);

    vec3 n_0 = mix(normal[1],normal[0],gl_TessCoord.x);
    vec3 n_1 = mix(normal[2],normal[3],gl_TessCoord.x);
    normal_out = normalize(mix(n_0, n_1, gl_TessCoord.y));

    debug_col_out = vec3(gl_TessCoord.x,gl_TessCoord.y,0.0);

    vec4 cs_p1 = mix(gl_in[1].gl_Position,gl_in[0].gl_Position,gl_TessCoord.x);
    vec4 cs_p2 = mix(gl_in[2].gl_Position,gl_in[3].gl_Position,gl_TessCoord.x);
    gl_Position = proj_mx * view_mx * mix(cs_p1, cs_p2, gl_TessCoord.y);
}