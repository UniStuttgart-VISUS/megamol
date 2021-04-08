layout (vertices = 4) out;

layout(location = 0) in vec3 world_pos[];
layout(location = 1) in vec3 normal[];
layout(location = 2) in vec4 color[];
layout(location = 3) flat in int cluster_id[];

layout(location = 0) out vec3 world_pos_out[];
layout(location = 1) out vec3 normal_out[];
layout(location = 2) out vec4 color_out[];
layout(location = 3) flat out int cluster_id_out[];

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
    normal_out[gl_InvocationID] = normal[gl_InvocationID];
    color_out[gl_InvocationID] = color[gl_InvocationID];
    cluster_id_out[gl_InvocationID] = cluster_id[gl_InvocationID];

    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
}
