#version 430

layout (local_size_x = 1024) in;

layout(std430, binding = 0) buffer ClusterInfos
{
    uint ci[];  // count, start, ... , data[count]
};

layout(std430, binding = 1) buffer VBO
{
    float vbo[];
};

layout(std430, binding = 2) buffer CBO
{
    float cbo[];
};

uniform uint count;
uniform uint pos_stride;
uniform uint col_stride;

void main()
{
    uint id = gl_GlobalInvocationID.y * (gl_NumWorkGroups.x * gl_WorkGroupSize.x) + gl_GlobalInvocationID.x;
    if(id >= count)
    return;

    uint particle_count = ci[2*id + 0];
    uint particle_start = ci[2*id + 1];

    uint X = 0, Y = 1, Z = 2;
    uint R = 0, G = 1, B = 2, A = 3;
    if(pos_stride == col_stride) //interleaved
    {
        X = 0;
        Y = 1;
        Z = 2;
        R = 3;
        G = 4;
        B = 5;
        A = 6;
    }

    vec4 col = vec4(vec3(id)/count, 1.0);
    for(uint i = particle_start; i < particle_start + particle_count; ++i)
    {
        cbo[ci[i]*col_stride + R] = col.r;
        cbo[ci[i]*col_stride + G] = col.g;
        cbo[ci[i]*col_stride + B] = col.b;
        cbo[ci[i]*col_stride + A] = col.a;
    }
}
