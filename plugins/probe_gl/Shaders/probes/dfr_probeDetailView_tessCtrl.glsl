layout (vertices = 4) out;

layout(location = 0) in vec3 world_pos[];

layout(location = 0) out vec3 world_pos_out[];

void main(void)
{
    if (gl_InvocationID == 0) // to not do same stuff 4 times
    {
        gl_TessLevelInner[0] = 1;
        gl_TessLevelInner[1] = 1;
        gl_TessLevelOuter[0] = 1;
        gl_TessLevelOuter[1] = 1;
        gl_TessLevelOuter[2] = 1;
        gl_TessLevelOuter[3] = 1;
    }

    world_pos_out[gl_InvocationID] = world_pos[gl_InvocationID];

    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
}