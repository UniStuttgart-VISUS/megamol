void PrepareDepthMip( const vec4 inPos/*, const vec2 inUV*/, int mipLevel, out float outD0, out float outD1, out float outD2, out float outD3 )
{
    ivec2 baseCoords = ivec2(inPos.xy) * 2;

    vec4 depthsArr[4];
    float depthsOutArr[4];

    // how to Gather a specific mip level?
    depthsArr[0].x = g_ViewspaceDepthSource[baseCoords + ivec2( 0, 0 )].x ;// * g_ASSAOConsts.MaxViewspaceDepth;
    depthsArr[0].y = g_ViewspaceDepthSource[baseCoords + ivec2( 1, 0 )].x ;// * g_ASSAOConsts.MaxViewspaceDepth;
    depthsArr[0].z = g_ViewspaceDepthSource[baseCoords + ivec2( 0, 1 )].x ;// * g_ASSAOConsts.MaxViewspaceDepth;
    depthsArr[0].w = g_ViewspaceDepthSource[baseCoords + ivec2( 1, 1 )].x ;// * g_ASSAOConsts.MaxViewspaceDepth;
    depthsArr[1].x = g_ViewspaceDepthSource1[baseCoords + ivec2( 0, 0 )].x;// * g_ASSAOConsts.MaxViewspaceDepth;
    depthsArr[1].y = g_ViewspaceDepthSource1[baseCoords + ivec2( 1, 0 )].x;// * g_ASSAOConsts.MaxViewspaceDepth;
    depthsArr[1].z = g_ViewspaceDepthSource1[baseCoords + ivec2( 0, 1 )].x;// * g_ASSAOConsts.MaxViewspaceDepth;
    depthsArr[1].w = g_ViewspaceDepthSource1[baseCoords + ivec2( 1, 1 )].x;// * g_ASSAOConsts.MaxViewspaceDepth;
    depthsArr[2].x = g_ViewspaceDepthSource2[baseCoords + ivec2( 0, 0 )].x;// * g_ASSAOConsts.MaxViewspaceDepth;
    depthsArr[2].y = g_ViewspaceDepthSource2[baseCoords + ivec2( 1, 0 )].x;// * g_ASSAOConsts.MaxViewspaceDepth;
    depthsArr[2].z = g_ViewspaceDepthSource2[baseCoords + ivec2( 0, 1 )].x;// * g_ASSAOConsts.MaxViewspaceDepth;
    depthsArr[2].w = g_ViewspaceDepthSource2[baseCoords + ivec2( 1, 1 )].x;// * g_ASSAOConsts.MaxViewspaceDepth;
    depthsArr[3].x = g_ViewspaceDepthSource3[baseCoords + ivec2( 0, 0 )].x;// * g_ASSAOConsts.MaxViewspaceDepth;
    depthsArr[3].y = g_ViewspaceDepthSource3[baseCoords + ivec2( 1, 0 )].x;// * g_ASSAOConsts.MaxViewspaceDepth;
    depthsArr[3].z = g_ViewspaceDepthSource3[baseCoords + ivec2( 0, 1 )].x;// * g_ASSAOConsts.MaxViewspaceDepth;
    depthsArr[3].w = g_ViewspaceDepthSource3[baseCoords + ivec2( 1, 1 )].x;// * g_ASSAOConsts.MaxViewspaceDepth;

    const uvec2 SVPosui         = uvec2( inPos.xy );
    const uint pseudoRandomA    = (SVPosui.x ) + 2 * (SVPosui.y );

    float dummyUnused1;
    float dummyUnused2;
    float falloffCalcMulSq, falloffCalcAdd;

    [unroll]
    for( int i = 0; i < 4; i++ )
    {
        vec4 depths = depthsArr[i];

        float closest = min( min( depths.x, depths.y ), min( depths.z, depths.w ) );

        CalculateRadiusParameters( abs( closest ), 1.0, dummyUnused1, dummyUnused2, falloffCalcMulSq );

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
