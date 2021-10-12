///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016, Intel Corporation
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
// documentation files (the "Software"), to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of
// the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

layout(local_size_x = 8, local_size_y = 8) in;

//vec4 PSApply( in vec4 inPos : SV_POSITION/*, in vec2 inUV : TEXCOORD0*/ ) : SV_Target
void main()
{
    vec3 inPos = gl_GlobalInvocationID;

    float ao;
    uvec2 pixPos     = uvec2(inPos.xy);
    uvec2 pixPosHalf = pixPos / uvec2(2, 2);

    // calculate index in the four deinterleaved source array texture
    int mx = int(pixPos.x % 2);
    int my = int(pixPos.y % 2);
    int ic = mx + my * 2;       // center index
    int ih = (1 - mx) + my * 2;   // neighbouring, horizontal
    int iv = mx + (1 - my) * 2;   // neighbouring, vertical
    int id = (1 - mx) + (1 - my) * 2; // diagonal

    vec2 centerVal = texelFetch(g_FinalSSAO, ivec3( pixPosHalf, ic ), 0).xy;

    ao = centerVal.x;

#if 1   // change to 0 if you want to disable last pass high-res blur (for debugging purposes, etc.)
    vec4 edgesLRTB = UnpackEdges( centerVal.y );

    // convert index shifts to sampling offsets
    float fmx   = float(mx);
    float fmy   = float(my);

    // in case of an edge, push sampling offsets away from the edge (towards pixel center)
    float fmxe  = (edgesLRTB.y - edgesLRTB.x);
    float fmye  = (edgesLRTB.w - edgesLRTB.z);

    // calculate final sampling offsets and sample using bilinear filter
    vec2  uvH = (inPos.xy + vec2( fmx + fmxe - 0.5, 0.5 - fmy ) ) * 0.5 * g_ASSAOConsts.HalfViewportPixelSize;
    float   aoH = textureLod(g_FinalSSAOLinearClamp, vec3( uvH, ih ), 0 ).x;
    vec2  uvV = (inPos.xy + vec2( 0.5 - fmx, fmy - 0.5 + fmye ) ) * 0.5 * g_ASSAOConsts.HalfViewportPixelSize;
    float   aoV = textureLod( g_FinalSSAOLinearClamp, vec3( uvV, iv ), 0 ).x;
    vec2  uvD = (inPos.xy + vec2( fmx - 0.5 + fmxe, fmy - 0.5 + fmye ) ) * 0.5 * g_ASSAOConsts.HalfViewportPixelSize;
    float   aoD = textureLod( g_FinalSSAOLinearClamp, vec3( uvD, id ), 0 ).x;

    // reduce weight for samples near edge - if the edge is on both sides, weight goes to 0
    vec4 blendWeights;
    blendWeights.x = 1.0;
    blendWeights.y = (edgesLRTB.x + edgesLRTB.y) * 0.5;
    blendWeights.z = (edgesLRTB.z + edgesLRTB.w) * 0.5;
    blendWeights.w = (blendWeights.y + blendWeights.z) * 0.5;

    // calculate weighted average
    float blendWeightsSum   = dot( blendWeights, vec4( 1.0, 1.0, 1.0, 1.0 ) );
    ao = dot( vec4( ao, aoH, aoV, aoD ), blendWeights ) / blendWeightsSum;
#endif

    //return vec4( ao.xxx, 1.0 );
    imageStore(g_FinalOutput, ivec2(pixPos), vec4(ao.xxx, 1.f));
}
