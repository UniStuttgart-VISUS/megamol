layout (vertices = 4) out;

layout(location = 0) in vec3 world_pos[];
layout(location = 1) in float sample_value[];
layout(location = 2) flat in int draw_id[];

layout(location = 0) out vec3 world_pos_out[];
layout(location = 1) out float sample_value_out[];
layout(location = 2) flat out int draw_id_out[];

void main(void)
{
    if (gl_InvocationID == 0) // to not do same stuff 4 times
    {
        gl_TessLevelInner[0] = 4;
        gl_TessLevelInner[1] = 4;
        gl_TessLevelOuter[0] = 4;
        gl_TessLevelOuter[1] = 4;
        gl_TessLevelOuter[2] = 4;
        gl_TessLevelOuter[3] = 4;
    }

    world_pos_out[gl_InvocationID] = world_pos[gl_InvocationID];

    sample_value_out[gl_InvocationID] = sample_value[gl_InvocationID];

    draw_id_out[gl_InvocationID] = draw_id[gl_InvocationID];

    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
}
