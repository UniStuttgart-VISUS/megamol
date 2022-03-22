//layout(quads, equal_spacing, ccw) in;
layout(quads) in;

layout(location = 0) in vec3 world_pos[];
layout(location = 1) in float sample_value[];
layout(location = 2) flat in int draw_id[];

layout(location = 0) out vec3 normal_out;
layout(location = 1) out vec3 world_pos_out;
layout(location = 2) out float sample_value_out;
layout(location = 3) flat out int draw_id_out;

uniform mat4 view_mx;
uniform mat4 proj_mx;

void main()
{
    vec3 p_0 = mix(world_pos[1],world_pos[0],gl_TessCoord.x);
    vec3 p_1 = mix(world_pos[2],world_pos[3],gl_TessCoord.x);
    world_pos_out = mix(p_0, p_1, gl_TessCoord.y);

    vec3 tangent = normalize(world_pos[0]-world_pos[1]);
    vec3 bitangent = normalize(world_pos[2]-world_pos[1]);
    normal_out = normalize(cross(tangent,bitangent));

    float v_0 = mix(sample_value[1],sample_value[0],gl_TessCoord.x);
    float v_1 = mix(sample_value[2],sample_value[3],gl_TessCoord.x);
    sample_value_out = mix(v_0, v_1, gl_TessCoord.y);

    draw_id_out = draw_id[0];

    vec4 cs_p1 = mix(gl_in[1].gl_Position,gl_in[0].gl_Position,gl_TessCoord.x);
    vec4 cs_p2 = mix(gl_in[2].gl_Position,gl_in[3].gl_Position,gl_TessCoord.x);
    gl_Position = proj_mx * view_mx * mix(cs_p1, cs_p2, gl_TessCoord.y);
}
