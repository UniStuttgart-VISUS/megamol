uint getInvocationID()
{
    uvec3 grid = gl_NumWorkGroups * gl_WorkGroupSize;

    return grid.x * (gl_GlobalInvocationID.z * grid.y + gl_GlobalInvocationID.y) + gl_GlobalInvocationID.x;
}

uint getInvocationID(uvec2 dimension)
{
    return dimension.x * (gl_GlobalInvocationID.z * dimension.y + gl_GlobalInvocationID.y) + gl_GlobalInvocationID.x;
}

uint maxInvocationID()
{
    uvec3 grid = gl_NumWorkGroups * gl_WorkGroupSize;

    return grid.x * grid.y * grid.z;
}
