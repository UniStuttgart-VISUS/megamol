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

void PrepareDepthMip( const vec4 inPos/*, const vec2 inUV*/, int mipLevel, out float outD0, out float outD1, out float outD2, out float outD3 )
{
    ivec2 baseCoords = ivec2(inPos.xy) * 2;

    vec4 depthsArr[4];
    float depthsOutArr[4];

    // how to Gather a specific mip level?
    //depthsArr[0].x = g_ViewspaceDepthSource[baseCoords + ivec2( 0, 0 )].x ;// * g_ASSAOConsts.MaxViewspaceDepth;
    //depthsArr[0].y = g_ViewspaceDepthSource[baseCoords + ivec2( 1, 0 )].x ;// * g_ASSAOConsts.MaxViewspaceDepth;
    //depthsArr[0].z = g_ViewspaceDepthSource[baseCoords + ivec2( 0, 1 )].x ;// * g_ASSAOConsts.MaxViewspaceDepth;
    //depthsArr[0].w = g_ViewspaceDepthSource[baseCoords + ivec2( 1, 1 )].x ;// * g_ASSAOConsts.MaxViewspaceDepth;
    //depthsArr[1].x = g_ViewspaceDepthSource1[baseCoords + ivec2( 0, 0 )].x;// * g_ASSAOConsts.MaxViewspaceDepth;
    //depthsArr[1].y = g_ViewspaceDepthSource1[baseCoords + ivec2( 1, 0 )].x;// * g_ASSAOConsts.MaxViewspaceDepth;
    //depthsArr[1].z = g_ViewspaceDepthSource1[baseCoords + ivec2( 0, 1 )].x;// * g_ASSAOConsts.MaxViewspaceDepth;
    //depthsArr[1].w = g_ViewspaceDepthSource1[baseCoords + ivec2( 1, 1 )].x;// * g_ASSAOConsts.MaxViewspaceDepth;
    //depthsArr[2].x = g_ViewspaceDepthSource2[baseCoords + ivec2( 0, 0 )].x;// * g_ASSAOConsts.MaxViewspaceDepth;
    //depthsArr[2].y = g_ViewspaceDepthSource2[baseCoords + ivec2( 1, 0 )].x;// * g_ASSAOConsts.MaxViewspaceDepth;
    //depthsArr[2].z = g_ViewspaceDepthSource2[baseCoords + ivec2( 0, 1 )].x;// * g_ASSAOConsts.MaxViewspaceDepth;
    //depthsArr[2].w = g_ViewspaceDepthSource2[baseCoords + ivec2( 1, 1 )].x;// * g_ASSAOConsts.MaxViewspaceDepth;
    //depthsArr[3].x = g_ViewspaceDepthSource3[baseCoords + ivec2( 0, 0 )].x;// * g_ASSAOConsts.MaxViewspaceDepth;
    //depthsArr[3].y = g_ViewspaceDepthSource3[baseCoords + ivec2( 1, 0 )].x;// * g_ASSAOConsts.MaxViewspaceDepth;
    //depthsArr[3].z = g_ViewspaceDepthSource3[baseCoords + ivec2( 0, 1 )].x;// * g_ASSAOConsts.MaxViewspaceDepth;
    //depthsArr[3].w = g_ViewspaceDepthSource3[baseCoords + ivec2( 1, 1 )].x;// * g_ASSAOConsts.MaxViewspaceDepth;

    // texelFetch seems equivalent to Texture2D::Operator function[]
    depthsArr[0].x = texelFetch(g_ViewspaceDepthSource,  baseCoords + ivec2( 0, 0 ), 0).x ;// * g_ASSAOConsts.MaxViewspaceDepth;
    depthsArr[0].y = texelFetch(g_ViewspaceDepthSource,  baseCoords + ivec2( 1, 0 ), 0).x ;// * g_ASSAOConsts.MaxViewspaceDepth;
    depthsArr[0].z = texelFetch(g_ViewspaceDepthSource,  baseCoords + ivec2( 0, 1 ), 0).x ;// * g_ASSAOConsts.MaxViewspaceDepth;
    depthsArr[0].w = texelFetch(g_ViewspaceDepthSource,  baseCoords + ivec2( 1, 1 ), 0).x ;// * g_ASSAOConsts.MaxViewspaceDepth;
    depthsArr[1].x = texelFetch(g_ViewspaceDepthSource1, baseCoords + ivec2( 0, 0 ), 0).x;// * g_ASSAOConsts.MaxViewspaceDepth;
    depthsArr[1].y = texelFetch(g_ViewspaceDepthSource1, baseCoords + ivec2( 1, 0 ), 0).x;// * g_ASSAOConsts.MaxViewspaceDepth;
    depthsArr[1].z = texelFetch(g_ViewspaceDepthSource1, baseCoords + ivec2( 0, 1 ), 0).x;// * g_ASSAOConsts.MaxViewspaceDepth;
    depthsArr[1].w = texelFetch(g_ViewspaceDepthSource1, baseCoords + ivec2( 1, 1 ), 0).x;// * g_ASSAOConsts.MaxViewspaceDepth;
    depthsArr[2].x = texelFetch(g_ViewspaceDepthSource2, baseCoords + ivec2( 0, 0 ), 0).x;// * g_ASSAOConsts.MaxViewspaceDepth;
    depthsArr[2].y = texelFetch(g_ViewspaceDepthSource2, baseCoords + ivec2( 1, 0 ), 0).x;// * g_ASSAOConsts.MaxViewspaceDepth;
    depthsArr[2].z = texelFetch(g_ViewspaceDepthSource2, baseCoords + ivec2( 0, 1 ), 0).x;// * g_ASSAOConsts.MaxViewspaceDepth;
    depthsArr[2].w = texelFetch(g_ViewspaceDepthSource2, baseCoords + ivec2( 1, 1 ), 0).x;// * g_ASSAOConsts.MaxViewspaceDepth;
    depthsArr[3].x = texelFetch(g_ViewspaceDepthSource3, baseCoords + ivec2( 0, 0 ), 0).x;// * g_ASSAOConsts.MaxViewspaceDepth;
    depthsArr[3].y = texelFetch(g_ViewspaceDepthSource3, baseCoords + ivec2( 1, 0 ), 0).x;// * g_ASSAOConsts.MaxViewspaceDepth;
    depthsArr[3].z = texelFetch(g_ViewspaceDepthSource3, baseCoords + ivec2( 0, 1 ), 0).x;// * g_ASSAOConsts.MaxViewspaceDepth;
    depthsArr[3].w = texelFetch(g_ViewspaceDepthSource3, baseCoords + ivec2( 1, 1 ), 0).x;// * g_ASSAOConsts.MaxViewspaceDepth;

    const uvec2 SVPosui         = uvec2( inPos.xy );
    const uint pseudoRandomA    = (SVPosui.x ) + 2 * (SVPosui.y );

    float dummyUnused1;
    float dummyUnused2;
    float falloffCalcMulSq, falloffCalcAdd;

    //[unroll]
    // should unroll automatically in glsl
    for( int i = 0; i < 4; i++ )
    {
        vec4 depths = depthsArr[i];

        float closest = min( min( depths.x, depths.y ), min( depths.z, depths.w ) );

        CalculateRadiusParameters( abs( closest ), vec2(1.0), dummyUnused1, dummyUnused2, falloffCalcMulSq );

        vec4 dists = depths - closest.xxxx;

        vec4 weights = clamp( dists * dists * falloffCalcMulSq + 1.0, 0.0, 1.0 );

        float smartAvg = dot( weights, depths ) / dot( weights, vec4( 1.0, 1.0, 1.0, 1.0 ) );

        const uint pseudoRandomIndex = ( pseudoRandomA + i ) % 4;

        //depthsOutArr[i] = closest;
        //depthsOutArr[i] = depths[ pseudoRandomIndex ];
        depthsOutArr[i] = smartAvg;
    }

    outD0 = depthsOutArr[0];
    outD1 = depthsOutArr[1];
    outD2 = depthsOutArr[2];
    outD3 = depthsOutArr[3];
}
