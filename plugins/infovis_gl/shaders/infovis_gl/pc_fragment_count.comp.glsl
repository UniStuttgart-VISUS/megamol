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

    memoryBarrierBuffer();
    uvec2 texCoord = gl_GlobalInvocationID.xy;

    uint thisMin = 4294967295u;
    uint thisMax = 0;

    texCoord.y = gl_GlobalInvocationID.y;
    while (texCoord.y < resolution.y) {
        texCoord.x = gl_GlobalInvocationID.x;
        while (texCoord.x < resolution.x) {
            float texel = texelFetch(fragmentCount, ivec2(texCoord), 0).r;
            uint count = uint(texel);
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

    atomicMin(fragmentMinMax[0].x,thisMin);
    atomicMax(fragmentMinMax[0].y,thisMax);
}
