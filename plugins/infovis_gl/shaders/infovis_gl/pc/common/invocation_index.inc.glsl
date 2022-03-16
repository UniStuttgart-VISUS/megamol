uint globalInvocationIndex() {
    const uvec3 globalSize = gl_NumWorkGroups * gl_WorkGroupSize;
    return globalSize.x * (globalSize.y * gl_GlobalInvocationID.z + gl_GlobalInvocationID.y) + gl_GlobalInvocationID.x;
}
