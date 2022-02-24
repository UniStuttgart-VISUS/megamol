#version 430

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

#include "pc_common/pc_fragment_count_buffers.inc.glsl"
#include "pc_common/pc_fragment_count_uniforms.inc.glsl"
#include "pc_common/pc_cs_id_getInvocationID.inc.glsl"

uniform uvec2 resolution = uvec2(0);
uniform uvec2 fragmentCountStepSize = uvec2(16);

void main()
{
    if (any(greaterThanEqual(gl_GlobalInvocationID.xy, resolution)))
    {
        return;
    }

    uint invocationID = getInvocationID();
    fragmentMinMax[invocationID] = uvec2(4294967295u, 0u);
    
    memoryBarrierBuffer();
    uvec2 texCoord = gl_GlobalInvocationID.xy;

    uint thisMin = 4294967295u;
    uint thisMax = 0;

    texCoord.y = gl_GlobalInvocationID.y;
    while (texCoord.y < resolution.y) {
        texCoord.x = gl_GlobalInvocationID.x;
        while (texCoord.x < resolution.x) {
            vec4 texel = texelFetch(fragmentCount, ivec2(texCoord), 0) - clearColor;
            uint count = uint(texel.r);
            if (count >= 0 && count < thisMin) {
                thisMin = count;
            }
            if (count > thisMax) {
                thisMax = count;
            }
            texCoord.x += fragmentCountStepSize.x;
        }
        texCoord.y += fragmentCountStepSize.y;
    }

    // Technically, it is no longer need to store the value for each innvocation
    // as the min/max is currently gather from local variable using atomic operations
    // (see below)
    // I will keep this for now in case we want to go back to a different gathering method
    fragmentMinMax[invocationID] = uvec2(thisMin, thisMax);

    atomicMin(fragmentMinMax[0].x,thisMin);
    atomicMax(fragmentMinMax[0].y,thisMax);
}
